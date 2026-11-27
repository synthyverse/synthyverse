import functools
import time

import numpy as np
import pandas as pd

from synthyverse.evaluation import fidelity, privacy
from synthyverse.evaluation.distances import (
    FaissGowerNN,
    FastGowerNN,
    faiss_gpu_available,
)
from synthyverse.evaluation.fidelity import AlphaPrecisionBetaRecall
from synthyverse.evaluation.preprocessing import fast_gower_transform
from synthyverse.evaluation.privacy import DCR, DPI

GPU = faiss_gpu_available()
DEVICES = ["cpu", "gpu", "auto"] if GPU else ["cpu", "auto"]
print(f"faiss GPU available: {GPU} -> testing devices {DEVICES}")

num_cols = [f"num_{i}" for i in range(20)]
cat_cols = [f"cat_{i}" for i in range(10)]
k = 3


def make_data(n, rng, nan_frac=0.0):
    """Synthetic mixed data, min-max scaled and integer-coded for Gower."""
    df = pd.DataFrame(rng.normal(size=(n, len(num_cols))), columns=num_cols)
    if nan_frac:
        df = df.mask(rng.random(df.shape) < nan_frac)
    df[cat_cols] = rng.integers(0, 5, size=(n, len(cat_cols)))
    return fast_gower_transform({"x": df}, reference_data=df, discrete_features=cat_cols)["x"]


def knn(cls, ref, query=None, normalize=False, **kwargs):
    nn = cls(numeric_cols=num_cols, categorical_cols=cat_cols, normalize=normalize, **kwargs)
    return nn.fit(ref).kneighbors(X=query, k=k)


rng = np.random.default_rng(42)

# Warm-up: exclude numba JIT and CUDA initialisation from timings.
warm = make_data(1_000, rng)
knn(FastGowerNN, warm)
for dev in DEVICES:
    knn(FaissGowerNN, warm, device=dev)

# --- 1. k-NN benchmark + equality with FastGowerNN ----------------------------
rows = []
for n in [10_000, 50_000]:
    df = make_data(n, rng)
    for normalize in (False, True):
        t0 = time.perf_counter()
        D_ref, I_ref = knn(FastGowerNN, df, normalize=normalize)
        rows.append({"n": n, "normalize": normalize, "method": "FastGowerNN (numba)",
                     "seconds": time.perf_counter() - t0, "identical_idx_rows": 1.0})
        for dev in DEVICES:
            t0 = time.perf_counter()
            D, I = knn(FaissGowerNN, df, normalize=normalize, device=dev)
            rows.append({"n": n, "normalize": normalize, "method": f"FaissGowerNN ({dev})",
                         "seconds": time.perf_counter() - t0,
                         "identical_idx_rows": (I == I_ref).all(axis=1).mean()})
            np.testing.assert_allclose(D, D_ref, rtol=1e-5, atol=1e-5,
                                       err_msg=f"{dev}, n={n}, normalize={normalize}")

bench = pd.DataFrame(rows)
bench["speedup_vs_numba"] = bench.groupby(["n", "normalize"])["seconds"].transform(
    lambda s: s.iloc[0] / s
)
print(f"\nSelf-query Gower k-NN (k={k}, {len(num_cols)} numeric + {len(cat_cols)} categorical cols)")
print(bench.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# External query (non-self path), data scaled jointly.
df = make_data(10_000, rng)
ref, query = df.iloc[:5_000], df.iloc[5_000:]
D_ref, I_ref = knn(FastGowerNN, ref, query)
for dev in DEVICES:
    D, I = knn(FaissGowerNN, ref, query, device=dev)
    np.testing.assert_allclose(D, D_ref, rtol=1e-5, atol=1e-5, err_msg=f"ext-query {dev}")
print("ext-query: distances match on", DEVICES)

# Chunking arguments are accepted (API compatibility) and change nothing.
for dev in DEVICES:
    D0, I0 = knn(FaissGowerNN, ref, query, device=dev)
    nn = FaissGowerNN(numeric_cols=num_cols, categorical_cols=cat_cols,
                      ref_chunk_size=1_000, device=dev).fit(ref)
    D1, I1 = nn.kneighbors(query, k=k, query_chunk_size=500)
    assert np.array_equal(D0, D1) and np.array_equal(I0, I1), dev
print("chunking arguments accepted, results unchanged on", DEVICES)

# device="auto": small searches go to the CPU (GPU setup would dominate),
# large ones to the GPU when available. Results match device="cpu".
for n_rows, expected in [(1_000, "cpu"), (10_000, "gpu" if GPU else "cpu")]:
    df_r = make_data(n_rows, rng)  # 30 cols: work = n_rows**2 * 30
    nn = FaissGowerNN(numeric_cols=num_cols, categorical_cols=cat_cols).fit(df_r)
    D_auto, I_auto = nn.kneighbors(k=k)
    assert nn.last_device_ == expected, (n_rows, nn.last_device_)
    D_cpu, I_cpu = knn(FaissGowerNN, df_r, device="cpu")
    assert np.array_equal(I_auto, I_cpu) and np.allclose(D_auto, D_cpu, atol=1e-6)
    print(f"auto routing: n={n_rows} x 30 cols -> {nn.last_device_}")

# GPU memory is released after each search (only the CUDA context remains).
if GPU:
    import os
    import subprocess

    def gpu_mib():
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        return sum(int(m) for p, m in (l.split(",") for l in out.strip().splitlines() if l)
                   if int(p) == os.getpid())

    before = gpu_mib()
    knn(FaissGowerNN, make_data(50_000, rng), device="gpu")
    after = gpu_mib()
    print(f"GPU memory of this process before/after a 50k GPU search: {before} / {after} MiB")
    assert after - before <= 100, "faiss kept GPU memory after the search"

# --- 2. Missing numerics --------------------------------------------------------
# normalize=True: both rank by the mean over jointly observed features.
df = make_data(5_000, rng, nan_frac=0.05)
D_ref, _ = knn(FastGowerNN, df, normalize=True)
for dev in DEVICES:
    D, _ = knn(FaissGowerNN, df, normalize=True, device=dev)
    np.testing.assert_allclose(D, D_ref, rtol=1e-5, atol=1e-5, err_msg=f"NaN {dev}")
print("NaN numerics (normalize=True): distances match on", DEVICES)

# --- 3. Real-data min-max scaling + clipping -----------------------------------
real = pd.DataFrame(rng.normal(size=(1_000, 3)), columns=["a", "b", "c"])
syn = real * 3 + 5  # far outside the real range
scaled = fast_gower_transform({"real": real, "syn": syn}, reference_data=real)
assert all(((d >= 0) & (d <= 1)).all().all() for d in scaled.values())
assert np.allclose(scaled["real"].min(), 0) and np.allclose(scaled["real"].max(), 1)
FaissGowerNN().fit(scaled["real"]).kneighbors(scaled["syn"], k=1)
print("real-range scaling: real spans [0, 1], synthetic clipped into [0, 1]; FaissGowerNN runs")

# Outlier robustness: one extreme synthetic row must not rescale all distances.
train = pd.DataFrame(rng.normal(size=(2_000, 4)), columns=list("abcd"))
test = pd.DataFrame(rng.normal(size=(1_000, 4)), columns=list("abcd"))
syn = train + rng.normal(0, 0.3, size=train.shape)
syn_outlier = syn.copy()
syn_outlier.iloc[0] = 1e6
dcr = DCR(subsample_test_size=False)
base = dcr.evaluate(train, syn, X_test=test)
outl = dcr.evaluate(train, syn_outlier, X_test=test)
for key in ("dcr.score", "dcr.002"):
    print(f"{key} without / with one 1e6 synthetic row: {base[key]:.4f} / {outl[key]:.4f}")
    assert abs(outl[key] - base[key]) <= 0.01 * max(abs(base[key]), 1e-3) + 1e-3, (
        f"outlier changed {key}"
    )

# --- 4. Metrics: FaissGowerNN vs FastGowerNN backend ----------------------------


def make_metric_data(kind, rng, n_train=2_000, n_test=1_000, n_syn=2_000):
    """Raw (unscaled) train/test/synthetic data; synthetic = noisy train copies."""
    n_num, n_cat = {"mixed": (6, 4), "mixed_nan": (6, 4), "categorical": (0, 8),
                    "numeric": (8, 0)}[kind]
    def real(n):
        df = pd.DataFrame(rng.normal(10, 3, size=(n, n_num)), columns=[f"x{i}" for i in range(n_num)])
        for j in range(n_cat):
            df[f"c{j}"] = rng.choice(list("abcdef"), size=n)
        return df
    train, test = real(n_train), real(n_test)
    syn = train.sample(n_syn, replace=True, random_state=0).reset_index(drop=True)
    noisy = rng.random(n_syn) > 0.1  # 10% exact copies
    for c in syn.columns[:n_num]:
        syn.loc[noisy, c] += rng.normal(0, 1, noisy.sum())
    for c in syn.columns[n_num:]:
        flip = noisy & (rng.random(n_syn) < 0.2)
        syn.loc[flip, c] = rng.choice(list("abcdef"), size=flip.sum())
    if kind == "mixed_nan":
        for d in (train, test, syn):
            num = d.columns[:n_num]
            d[num] = d[num].mask(rng.random((len(d), n_num)) < 0.05)
    return train, test, syn, list(train.columns[n_num:])


def run_metrics(backend, train, test, syn, discrete):
    old = privacy.FaissGowerNN, fidelity.FaissGowerNN
    privacy.FaissGowerNN = fidelity.FaissGowerNN = backend
    try:
        out = {}
        out.update(DCR().evaluate(train, syn, X_test=test, discrete_features=discrete))
        out.update(DPI().evaluate(train, syn, X_test=test, discrete_features=discrete))
        out.update(AlphaPrecisionBetaRecall().evaluate(train, syn, X_test=test,
                                                       discrete_features=discrete))
        return out
    finally:
        privacy.FaissGowerNN, fidelity.FaissGowerNN = old


metric_rows, failures = [], []
for kind in ["mixed", "mixed_nan", "categorical", "numeric"]:
    train, test, syn, discrete = make_metric_data(kind, np.random.default_rng(7))
    t0 = time.perf_counter()
    ref_res = run_metrics(FastGowerNN, train, test, syn, discrete)
    t_ref = time.perf_counter() - t0
    for dev in DEVICES:
        t0 = time.perf_counter()
        res = run_metrics(functools.partial(FaissGowerNN, device=dev), train, test, syn, discrete)
        t = time.perf_counter() - t0
        assert res.keys() == ref_res.keys(), (kind, dev)
        diff = {key: (ref_res[key], res[key]) for key in ref_res
                if not np.allclose(res[key], ref_res[key], rtol=1e-5, atol=1e-6, equal_nan=True)}
        if diff:
            failures.append((kind, dev, diff))
        metric_rows.append({"data": kind, "device": dev, "n_scores": len(res),
                            "mismatches": len(diff), "s_numba": t_ref, "s_faiss": t})

print("\nMetrics (DCR, DPI, AlphaPrecisionBetaRecall): FaissGowerNN vs FastGowerNN")
print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
# With missing numerics, FaissGowerNN ranks neighbours by the mean Gower
# distance over jointly observed features (Gower's definition), FastGowerNN
# with normalize=False by the sum, so DCR/DPI may differ there by design.
expected = [f for f in failures if f[0] == "mixed_nan"]
unexpected = [f for f in failures if f[0] != "mixed_nan"]
for kind, dev, diff in expected:
    print(f"expected difference (NaN numerics, mean vs sum ranking) {kind}/{dev}: "
          f"{sorted(diff)}")
for kind, dev, diff in unexpected:
    print(f"MISMATCH {kind}/{dev}:")
    for key, (a, b) in diff.items():
        print(f"   {key}: FastGowerNN={a} FaissGowerNN={b}")
assert not unexpected, "metric results differ between backends"
print("Metric results identical for all data without missing numerics.")
