import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from numba import njit, prange
from scipy.special import kl_div, rel_entr
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import pairwise_distances


def l1(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(np.abs(np.asarray(x) - np.asarray(y)), axis=axis)


def l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x)))


def js_robust(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    p = np.asarray(p, dtype=float).copy()
    q = np.asarray(q, dtype=float).copy()
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    js = np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m))
    js /= np.log(base)
    return float(np.sqrt(np.clip(js / 2, 0, None)))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(
        0.5 * np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum()
    )


def kld(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    eps = np.finfo(float).eps
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(kl_div(p, q)) / np.log(2))


def ksd(x: np.ndarray, y: np.ndarray) -> float:
    return float(
        ks_2samp(
            np.asarray(x),
            np.asarray(y),
            alternative="two-sided",
            mode="auto",
        ).statistic
    )


def wsd(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    min_ = y.min()
    max_ = y.max()
    if max_ == min_:
        scale = max_ if max_ != 0 else 1
        return float(wasserstein_distance(x / scale, y / scale))

    x_z = (x - min_) / (max_ - min_)
    y_z = (y - min_) / (max_ - min_)
    return float(wasserstein_distance(x_z, y_z))


def pairwise_l1(
    x: np.ndarray,
    y: np.ndarray = None,
    n_jobs: int = -1,
) -> np.ndarray:
    if y is None:
        y = x
    return pairwise_distances(x, y, metric="cityblock", n_jobs=n_jobs)


def kth_smallest(values: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    kth_index = k - 1
    if kth_index < 0 or kth_index >= values.shape[axis]:
        raise ValueError("k is outside the selected axis.")
    return np.take(np.partition(values, kth_index, axis=axis), kth_index, axis=axis)


def knn_distances(
    x: np.ndarray,
    k: int,
    n_jobs: int = -1,
) -> np.ndarray:
    distances = pairwise_l1(x, n_jobs=n_jobs)
    return kth_smallest(distances, k + 1, axis=-1)


@njit(cache=True, fastmath=True)
def _insert_topk(best_dist, best_idx, d, j):
    """
    Maintain sorted top-k arrays in ascending distance order.
    """
    k = best_dist.shape[0]

    if d >= best_dist[k - 1]:
        return

    pos = k - 1
    while pos > 0 and d < best_dist[pos - 1]:
        best_dist[pos] = best_dist[pos - 1]
        best_idx[pos] = best_idx[pos - 1]
        pos -= 1

    best_dist[pos] = d
    best_idx[pos] = j


@njit(cache=True, fastmath=True, parallel=True)
def _gower_topk_numba(
    Q_num,
    Q_num_nan,
    Q_cat,
    Q_cat_nan,
    R_num,
    R_num_nan,
    R_cat,
    R_cat_nan,
    k,
    exclude_self,
    query_offset,
    ref_offset,
    normalize,
):
    """
    Exact blocked Gower top-k search.

    Numeric features are assumed pre-scaled to [0, 1].
    Categorical features are integer-coded.
    Values marked in the missing-value masks are ignored featurewise. If
    normalize is true, the distance is renormalized by the number of jointly
    observed features.
    """
    nq = Q_num.shape[0]
    nr = R_num.shape[0]
    p_num = Q_num.shape[1]
    p_cat = Q_cat.shape[1]

    out_dist = np.empty((nq, k), dtype=np.float32)
    out_idx = np.empty((nq, k), dtype=np.int64)

    for i in prange(nq):
        best_dist = np.empty(k, dtype=np.float32)
        best_idx = np.empty(k, dtype=np.int64)

        for kk in range(k):
            best_dist[kk] = np.inf
            best_idx[kk] = -1

        global_i = query_offset + i

        for j in range(nr):
            global_j = ref_offset + j

            if exclude_self and global_i == global_j:
                continue

            s = 0.0
            cnt = 0

            # numeric contribution
            for c in range(p_num):
                if not Q_num_nan[i, c] and not R_num_nan[j, c]:
                    diff = Q_num[i, c] - R_num[j, c]
                    if diff < 0:
                        diff = -diff
                    s += diff
                    cnt += 1

            # categorical contribution
            for c in range(p_cat):
                if not Q_cat_nan[i, c] and not R_cat_nan[j, c]:
                    if Q_cat[i, c] != R_cat[j, c]:
                        s += 1.0
                    cnt += 1

            if cnt == 0:
                d = np.inf
            elif normalize:
                d = s / cnt
            else:
                d = s

            _insert_topk(best_dist, best_idx, np.float32(d), global_j)

        for kk in range(k):
            out_dist[i, kk] = best_dist[kk]
            out_idx[i, kk] = best_idx[kk]

    return out_dist, out_idx


@njit(cache=True, fastmath=True)
def _merge_topk(existing_dist, existing_idx, cand_dist, cand_idx, k):
    """
    Merge current top-k with candidate top-k for each query row.
    """
    nq = existing_dist.shape[0]

    for i in range(nq):
        for c in range(k):
            d = cand_dist[i, c]
            j = cand_idx[i, c]

            if j < 0:
                continue

            if d >= existing_dist[i, k - 1]:
                continue

            pos = k - 1
            while pos > 0 and d < existing_dist[i, pos - 1]:
                existing_dist[i, pos] = existing_dist[i, pos - 1]
                existing_idx[i, pos] = existing_idx[i, pos - 1]
                pos -= 1

            existing_dist[i, pos] = d
            existing_idx[i, pos] = j


class FastGowerNN:
    """
    Exact, blocked, memory-light Gower nearest-neighbour search.

    Distance:
        numeric: abs(x - y)
        categorical: 0 if equal else 1
        missing numeric: feature ignored
        missing categorical: encoded as a separate category

    Parameters:
        normalize:
            If True, return the average per-observed-feature distance. If
            False, return the summed mixed-type distance.

    Notes:
        - Exact Gower NN is still fundamentally O(n_query * n_ref * d).
        - This implementation is fast mainly because it:
            1. pre-encodes everything,
            2. uses float32 / int32 arrays,
            3. avoids the full pairwise matrix,
            4. uses Numba parallel loops,
            5. streams reference rows in chunks.
    """

    def __init__(
        self,
        numeric_cols=None,
        categorical_cols=None,
        ref_chunk_size=50_000,
        normalize=False,
    ):
        """
        Configure a Gower nearest-neighbour index.

        Parameters
        ----------
        numeric_cols, categorical_cols:
            Optional column lists used to split mixed-type input. Leaving either
            as None lets the estimator infer the missing side during ``fit``.
        ref_chunk_size:
            Number of fitted reference rows scanned at once. Larger chunks can
            be faster, while smaller chunks reduce peak memory use.
        normalize:
            If True, return average per-observed-feature Gower distances. If
            False, return summed feature contributions.
        """
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.ref_chunk_size = int(ref_chunk_size)
        self.normalize = bool(normalize)

        self.X_num_ = None
        self.X_num_nan_ = None
        self.X_cat_ = None
        self.X_cat_nan_ = None

    def fit(self, X):
        """
        Fit the reference data used for subsequent nearest-neighbour queries.

        Parameters
        ----------
        X:
            DataFrame whose rows become the searchable reference set. The
            neighbour indices returned by ``kneighbors`` point into this data.

        Returns
        -------
        self:
            The fitted nearest-neighbour index.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if self.numeric_cols is None:
            if self.categorical_cols is None:
                self.numeric_cols = list(X.columns)
                self.categorical_cols = []
            else:
                self.categorical_cols = list(self.categorical_cols)
                self.numeric_cols = [
                    c for c in X.columns if c not in self.categorical_cols
                ]
        elif self.categorical_cols is None:
            self.numeric_cols = list(self.numeric_cols)
            self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        else:
            self.numeric_cols = list(self.numeric_cols)
            self.categorical_cols = [
                c for c in self.categorical_cols if c not in self.numeric_cols
            ]

        self.X_num_, self.X_num_nan_ = self._transform_numeric(X)
        self.X_cat_, self.X_cat_nan_ = self._transform_categorical(X)

        self.n_samples_ = len(X)
        return self

    def _transform_numeric(self, X):
        if len(self.numeric_cols) == 0:
            return (
                np.empty((len(X), 0), dtype=np.float32),
                np.empty((len(X), 0), dtype=np.bool_),
            )

        arr = X[self.numeric_cols].to_numpy(dtype=np.float32, copy=True)
        nan = np.isnan(arr)
        arr[nan] = 0.0

        return np.ascontiguousarray(arr, dtype=np.float32), np.ascontiguousarray(nan)

    def _transform_categorical(self, X):
        if len(self.categorical_cols) == 0:
            return (
                np.empty((len(X), 0), dtype=np.int32),
                np.empty((len(X), 0), dtype=np.bool_),
            )

        invalid_cols = [
            col for col in self.categorical_cols if not is_integer_dtype(X[col])
        ]
        if invalid_cols:
            raise ValueError(
                "FastGowerNN categorical columns must be ordinally encoded as "
                "integer dtype before fitting or querying. Non-encoded columns: "
                f"{invalid_cols}. Use fast_gower_transform(..., "
                "categorical_fit_data=[...]) to create consistent category codes "
                "across datasets, or pass pre-encoded categorical columns."
            )

        arr = X[self.categorical_cols].to_numpy(
            dtype=np.float32, na_value=np.nan, copy=True
        )
        arr[np.isnan(arr)] = -1
        return (
            np.ascontiguousarray(arr.astype(np.int32)),
            np.zeros(arr.shape, dtype=np.bool_),
        )

    def kneighbors(
        self,
        X=None,
        k=1,
        exclude_self=None,
        query_chunk_size=None,
    ):
        """
        Return nearest-neighbour distances and fitted-reference indices.

        Parameters
        ----------
        X:
            Query DataFrame to compare against the fitted reference set. If
            None, the fitted reference data is queried against itself.
        k:
            Number of nearest neighbours to retain for each query row.
        exclude_self:
            Whether to ignore the same fitted row during self-query. The
            default is True when X is None, so each row does not return itself,
            and False for external query data.
        query_chunk_size:
            Number of query rows processed at once. Tune this to trade memory
            use against overhead when querying large data sets.

        Returns
        -------
        distances: float32 array, shape (n_query, k)
        indices: int64 array, shape (n_query, k)
            Indices into the fitted reference data.
        """
        if X is None:
            Q_num = self.X_num_
            Q_num_nan = self.X_num_nan_
            Q_cat = self.X_cat_
            Q_cat_nan = self.X_cat_nan_
            same_data = True
        else:
            Q_num, Q_num_nan = self._transform_numeric(X)
            Q_cat, Q_cat_nan = self._transform_categorical(X)
            same_data = False

        if exclude_self is None:
            exclude_self = same_data
        exclude_self = bool(exclude_self and same_data)

        k = int(k)
        if k <= 0:
            raise ValueError("k must be positive.")

        if exclude_self and k >= self.n_samples_:
            raise ValueError("When exclude_self=True, k must be < n_samples.")

        nq = Q_num.shape[0]
        if query_chunk_size is None:
            query_chunk_size = min(self.ref_chunk_size, nq)
        distances = np.full((nq, k), np.inf, dtype=np.float32)
        indices = np.full((nq, k), -1, dtype=np.int64)

        for qs in range(0, nq, query_chunk_size):
            qe = min(qs + query_chunk_size, nq)

            best_dist = np.full((qe - qs, k), np.inf, dtype=np.float32)
            best_idx = np.full((qe - qs, k), -1, dtype=np.int64)

            for rs in range(0, self.n_samples_, self.ref_chunk_size):
                re = min(rs + self.ref_chunk_size, self.n_samples_)

                cand_dist, cand_idx = _gower_topk_numba(
                    Q_num[qs:qe],
                    Q_num_nan[qs:qe],
                    Q_cat[qs:qe],
                    Q_cat_nan[qs:qe],
                    self.X_num_[rs:re],
                    self.X_num_nan_[rs:re],
                    self.X_cat_[rs:re],
                    self.X_cat_nan_[rs:re],
                    k,
                    exclude_self,
                    qs,
                    rs,
                    self.normalize,
                )

                _merge_topk(best_dist, best_idx, cand_dist, cand_idx, k)

            distances[qs:qe] = best_dist
            indices[qs:qe] = best_idx

        return distances, indices


# faiss caps top-k on GPU (k <= 2048); larger searches fall back to CPU.
_FAISS_GPU_MAX_K = 2048
# With device="auto", searches below this many n_query * n_ref * n_features
# operations run on the CPU: the per-search GPU setup (~80 ms) costs more than
# the faiss CPU search there. Machine-dependent; measured on a laptop GPU
# (RTX A1000) vs a 20-thread i7-12700H, where CPU and GPU break even at ~3e8.
_FAISS_GPU_MIN_WORK = 3e8
# Slack for float round-off when checking that numerics are min-max scaled.
_UNIT_RANGE_TOL = 1e-6


@lru_cache(maxsize=None)
def faiss_gpu_available() -> bool:
    """Return whether faiss can run on a GPU in this process.

    Requires Linux, the GPU build of faiss (``faiss-gpu``) and at least one
    visible GPU. The result is cached for the lifetime of the process.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        import faiss

        return hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    except Exception:
        return False


class FaissGowerNN:
    """Exact Gower nearest-neighbour search backed by faiss ``METRIC_GOWER``.

    Drop-in alternative to ``FastGowerNN`` with the same API. Runs on the GPU
    when available (Linux, ``faiss-gpu``, visible GPU) and on the faiss CPU
    brute-force index otherwise. GPU memory is allocated per search and
    released when the search returns (only the CUDA context stays). With
    ``device="auto"``, small searches run on the CPU because the per-search
    GPU setup would dominate; ``last_device_`` records the device used by the
    most recent search.

    Distance:
        numeric: abs(x - y), values must be min-max scaled to [0, 1]
        categorical: 0 if equal else 1
        missing numeric: feature ignored
        missing categorical: encoded as a separate category

    Notes:
        - Numeric features must lie in [0, 1] for both the fitted and the
          query data. Scale them with ``fast_gower_transform``, which uses the
          real-data min/max and clips other data into [0, 1].
        - faiss ranks neighbours by the mean Gower distance over jointly
          observed features. Without missing numerics this gives the same
          neighbours as ``FastGowerNN`` for both ``normalize`` settings. With
          missing numerics and ``normalize=False``, the returned summed
          distances are exact but neighbours are ranked by the mean, which
          can differ from ``FastGowerNN``'s ranking by the sum.
    """

    def __init__(
        self,
        numeric_cols=None,
        categorical_cols=None,
        ref_chunk_size=50_000,
        normalize=False,
        device="auto",
    ):
        """Configure a faiss-backed Gower nearest-neighbour index.

        Args:
            numeric_cols: Numeric columns. If None, inferred during ``fit``.
            categorical_cols: Integer-coded categorical columns. If None,
                inferred during ``fit``.
            ref_chunk_size: Accepted for API compatibility with
                ``FastGowerNN``; faiss splits searches into memory-sized
                pieces internally, so this has no effect.
            normalize: If True, return average per-observed-feature Gower
                distances. If False, return summed feature contributions.
            device: "auto" uses the GPU if available and the search is large
                enough to amortise the GPU setup (n_query * n_ref *
                n_features >= 3e8), else the CPU. "gpu" forces the GPU and
                "cpu" forces the CPU.
        """
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', got {device!r}.")
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.ref_chunk_size = ref_chunk_size
        self.normalize = bool(normalize)
        self.device = device

    def fit(self, X):
        """Fit the reference data used for subsequent nearest-neighbour queries.

        Args:
            X: DataFrame whose rows become the searchable reference set. The
                neighbour indices returned by ``kneighbors`` point into it.

        Returns:
            The fitted nearest-neighbour index.
        """
        import faiss

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if self.numeric_cols is None:
            if self.categorical_cols is None:
                self.numeric_cols = list(X.columns)
                self.categorical_cols = []
            else:
                self.categorical_cols = list(self.categorical_cols)
                self.numeric_cols = [
                    c for c in X.columns if c not in self.categorical_cols
                ]
        elif self.categorical_cols is None:
            self.numeric_cols = list(self.numeric_cols)
            self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        else:
            self.numeric_cols = list(self.numeric_cols)
            self.categorical_cols = [
                c for c in self.categorical_cols if c not in self.numeric_cols
            ]

        if len(self.numeric_cols) + len(self.categorical_cols) == 0:
            raise ValueError("FaissGowerNN requires at least one feature.")

        if self.device == "auto":
            self.device_ = "gpu" if faiss_gpu_available() else "cpu"
        elif self.device == "gpu" and not faiss_gpu_available():
            raise RuntimeError(
                "device='gpu' requires Linux, faiss-gpu and a visible GPU."
            )
        else:
            self.device_ = self.device
        # With "auto", small searches are routed to the CPU in _search.
        self.auto_device_ = self.device == "auto"

        self.X_, self.X_num_nan_ = self._encode(X)
        self.n_samples_ = len(X)

        # The CPU index is also the fallback for GPU searches with large k.
        self.index_ = faiss.IndexFlat(self.X_.shape[1], faiss.METRIC_GOWER)
        self.index_.add(self.X_)
        return self

    def _encode(self, X):
        """Encode a frame for faiss: numerics in [0, 1], categoricals < 0."""
        num = X[self.numeric_cols].to_numpy(dtype=np.float32, copy=True)
        num_nan = np.isnan(num)
        if num.size and not num_nan.all():
            lo, hi = np.nanmin(num), np.nanmax(num)
            if lo < -_UNIT_RANGE_TOL or hi > 1.0 + _UNIT_RANGE_TOL:
                raise ValueError(
                    "FaissGowerNN numeric columns must be min-max scaled to "
                    f"[0, 1] (found range [{lo:.4g}, {hi:.4g}]). Use "
                    "fast_gower_transform, which scales on real data and clips."
                )
            # Only absorbs float round-off; NaN is kept so faiss skips it.
            np.clip(num, 0.0, 1.0, out=num)

        invalid_cols = [
            col for col in self.categorical_cols if not is_integer_dtype(X[col])
        ]
        if invalid_cols:
            raise ValueError(
                "FaissGowerNN categorical columns must be ordinally encoded as "
                "integer dtype before fitting or querying. Non-encoded columns: "
                f"{invalid_cols}. Use fast_gower_transform(..., "
                "categorical_fit_data=[...]) to create consistent category codes "
                "across datasets, or pass pre-encoded categorical columns."
            )
        cat = X[self.categorical_cols].to_numpy(
            dtype=np.float32, na_value=np.nan, copy=True
        )
        cat[np.isnan(cat)] = -1
        # faiss treats strictly negative values as categories. Codes 0..K,
        # missing -1 and unknown -2 map to distinct values <= -1.
        cat = -(cat + 3.0)

        encoded = np.ascontiguousarray(np.hstack((num, cat)), dtype=np.float32)
        return encoded, num_nan

    def _search(self, Q, k):
        """Top-k search ordered by (distance, index), like ``FastGowerNN``.

        The CPU index already breaks distance ties by the lower index. GPU
        top-k selection does not, so the GPU fetches extra candidates, sorts
        them by (distance, index), and rows whose tie group at the k-th
        neighbour may extend past the fetched candidates are recomputed on
        the CPU.
        """
        k_ext = min(2 * k + 16, self.n_samples_, _FAISS_GPU_MAX_K)
        small = Q.shape[0] * self.n_samples_ * self.X_.shape[1] < _FAISS_GPU_MIN_WORK
        use_gpu = (
            self.device_ == "gpu"
            and k <= k_ext
            and not (self.auto_device_ and small)
        )
        self.last_device_ = "gpu" if use_gpu else "cpu"
        if not use_gpu:
            return self.index_.search(Q, k)

        import faiss

        # Per-search GPU resources: their memory pool is freed on return, so
        # faiss does not hold GPU memory between metrics or generator runs.
        res = faiss.StandardGpuResources()
        try:
            D, I = faiss.knn_gpu(res, Q, self.X_, k_ext, metric=faiss.METRIC_GOWER)
        finally:
            del res
        # Invalid slots (index -1) sort last.
        D = np.where(I < 0, np.inf, D)
        order = np.lexsort((np.where(I < 0, np.iinfo(np.int64).max, I), D), axis=-1)
        D = np.take_along_axis(D, order, axis=1)
        I = np.take_along_axis(I, order, axis=1)

        if k_ext < self.n_samples_:
            ambiguous = np.isfinite(D[:, k - 1]) & (D[:, k - 1] == D[:, -1])
            if ambiguous.any():
                D_cpu, I_cpu = self.index_.search(Q[ambiguous], k)
                D[ambiguous, :k] = D_cpu
                I[ambiguous, :k] = I_cpu
        return D[:, :k], I[:, :k]

    def kneighbors(self, X=None, k=1, exclude_self=None, query_chunk_size=None):
        """Return nearest-neighbour distances and fitted-reference indices.

        Args:
            X: Query DataFrame to compare against the fitted reference set. If
                None, the fitted reference data is queried against itself.
            k: Number of nearest neighbours to retain for each query row.
            exclude_self: Whether to ignore the same fitted row during
                self-query. Defaults to True when X is None and False for
                external query data.
            query_chunk_size: Accepted for API compatibility with
                ``FastGowerNN``; faiss splits searches into memory-sized
                pieces internally, so this has no effect.

        Returns:
            Tuple ``(distances, indices)``: float32 and int64 arrays of shape
            (n_query, k). Indices point into the fitted reference data.
        """
        if X is None:
            Q, Q_nan = self.X_, self.X_num_nan_
            same_data = True
        else:
            Q, Q_nan = self._encode(X)
            same_data = False

        if exclude_self is None:
            exclude_self = same_data
        exclude_self = bool(exclude_self and same_data)

        k = int(k)
        if k <= 0:
            raise ValueError("k must be positive.")
        if exclude_self and k >= self.n_samples_:
            raise ValueError("When exclude_self=True, k must be < n_samples.")

        nq = Q.shape[0]
        if exclude_self:
            D, I = self._search(Q, k + 1)
            drop = I == np.arange(nq)[:, None]
            # Ties may push self out of the top k+1: drop the last column.
            drop[~drop.any(axis=1), -1] = True
            D = D[~drop].reshape(nq, k)
            I = I[~drop].reshape(nq, k)
        else:
            D, I = self._search(Q, k)

        D = np.asarray(D, dtype=np.float32)
        I = np.asarray(I, dtype=np.int64)

        if not self.normalize:
            # faiss returns the mean; scale back by the jointly observed count.
            p_cat = len(self.categorical_cols)
            if Q_nan.any() or self.X_num_nan_.any():
                valid = np.where(I >= 0, I, 0)
                observed = ~(Q_nan[:, None, :] | self.X_num_nan_[valid])
                counts = p_cat + observed.sum(axis=-1)
            else:
                counts = len(self.numeric_cols) + p_cat
            # inf * 0 for pairs without observed features; reset to inf below.
            with np.errstate(invalid="ignore"):
                D = (D * counts).astype(np.float32)

        # Pairs without jointly observed features: same as FastGowerNN.
        D[I < 0] = np.inf
        return D, I
