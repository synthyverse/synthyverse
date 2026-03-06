from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import (
    LabelEncoder,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn_extra.cluster import KMedoids
from scipy.stats import truncnorm


from .eqf import EmpiricalInterpolatedQuantile


def get_bootstrap_repo(
    X: pd.DataFrame, X_enc: pd.DataFrame, numerical_columns: list, discretizers: dict
):
    bootstrap_repo = {}
    for col in numerical_columns:
        bin_ids = X_enc[col].astype(int).to_numpy()
        orig_vals = X[col].to_numpy()
        repo = {}
        n_bins_col = int(discretizers[col].n_bins_[0])
        for b in range(n_bins_col):
            repo[b] = orig_vals[bin_ids == b]
        bootstrap_repo[col] = repo
    return bootstrap_repo


def get_eqf_repo(X: pd.DataFrame, X_enc: pd.DataFrame, numerical_columns: list):
    eqf_repo = {}
    for col in numerical_columns:
        eqf_repo[col] = {}
        for bin in X_enc[col].unique():
            eqf = EmpiricalInterpolatedQuantile(
                n_knots=-1,  # use all training samples as knots
                use_spline=False,  # whether to use monotonic cubic spline interpolation
            )
            eqf.fit(X[col][X_enc[col] == bin].values)
            eqf_repo[col][bin] = eqf
    return eqf_repo


def label_encode(
    X,
    val_X,
    X_enc,
    val_X_enc,
    discrete_columns: list,
    use_early_stopping: bool,
):
    label_encoders = {}
    for col in X_enc.columns:
        le = LabelEncoder()
        if val_X is not None and use_early_stopping:
            le.fit(pd.concat([X_enc[col], val_X_enc[col]]))
            val_X_enc[col] = le.transform(val_X_enc[col])
        else:
            le.fit(X_enc[col])
        X_enc[col] = le.transform(X_enc[col])
        if col in discrete_columns:
            X[col] = le.transform(X[col])
            if val_X is not None and use_early_stopping:
                val_X[col] = le.transform(val_X[col])
        label_encoders[col] = le
    return X, val_X, X_enc, val_X_enc, label_encoders


def discretize(
    X_enc: pd.DataFrame,
    val_X_enc: pd.DataFrame,
    numerical_columns: list,
    n_bins: int,
    discretization: str,
    random_state: int,
    rng: np.random.RandomState,
    use_early_stopping: bool,
):

    discretizers = {}
    for col in numerical_columns:
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            strategy=discretization,
            encode="ordinal",
            random_state=random_state,
            quantile_method="averaged_inverted_cdf",
        )
        if discretization == "quantile":
            # add jitter to break quantile ties to ensure a minimum number of samples per bin
            try:
                eps = np.min(np.diff(np.sort(np.unique(X_enc[col].values)))) / 2
                noise = rng.uniform(-eps, eps, len(X_enc))
            except:
                noise = 0

            X_enc[col] = X_enc[col] + noise
            if val_X_enc is not None and use_early_stopping:
                try:
                    eps = np.min(np.diff(np.sort(np.unique(val_X_enc[col].values)))) / 2
                    noise = rng.uniform(-eps, eps, len(val_X_enc))
                except:
                    noise = 0
                val_X_enc[col] = val_X_enc[col] + noise

        X_enc[col] = discretizer.fit_transform(X_enc[col].values.reshape(-1, 1)).astype(
            int
        )

        if val_X_enc is not None and use_early_stopping:
            val_X_enc[col] = discretizer.transform(
                val_X_enc[col].values.reshape(-1, 1)
            ).astype(int)

        discretizers[col] = discretizer
    return X_enc, val_X_enc, discretizers


def sample_from_posterior(
    probs,
    colname,
    n,
    temperature: float,
    discrete_columns: list,
    rng: np.random.RandomState,
    per_bin_sampling: str,
    label_encoders: dict,
    discretizers: dict,
    repo: dict = {},
):
    # temperature scaling
    probs = np.exp(np.log(probs + 1e-10) / temperature)
    probs /= probs.sum(axis=1, keepdims=True)

    # inverse-CDF vectorized sampling
    cum_probs = np.cumsum(probs, axis=1)
    # Ensure cum_probs[:, -1] is exactly 1.0 to avoid out-of-bounds sampling
    cum_probs[:, -1] = 1.0
    r = rng.random(size=n)
    sampled = np.argmax(cum_probs >= r[:, None], axis=1)

    # return sampled

    if colname in discrete_columns:
        return sampled
    else:
        # map back to bin centers + noise
        if per_bin_sampling.endswith("noise"):
            classes_ = label_encoders[colname].classes_
            sampled_cls = classes_[sampled].astype(int)
        else:
            # repo's are built from label encoded values
            sampled_cls = sampled

        if per_bin_sampling.endswith("noise"):
            edges = discretizers[colname].bin_edges_[0]
            left = edges[sampled_cls]
            right = edges[sampled_cls + 1]
            centers = 0.5 * (left + right)
            widths = right - left
            # add truncated gaussian noise to bin center
            if per_bin_sampling == "gaussian_noise":
                # per-sample sigma; clamp to avoid div-by-zero
                sigmas = np.maximum(widths / 6, 1e-12)
                # truncated around zero, within the bin
                a = (left - centers) / sigmas
                b = (right - centers) / sigmas
                # draw zero-mean noise, no 'size' so shapes broadcast to (n,)
                eps = truncnorm.rvs(a, b, loc=0.0, scale=sigmas)
                vals = centers + eps
            elif per_bin_sampling == "uniform_noise":
                vals = left + rng.random(n) * widths

        elif per_bin_sampling == "bootstrap":
            vals = np.empty(n, dtype=float)
            repo = repo.get(colname, {})

            for b_id in np.unique(sampled_cls):
                rows = np.where(sampled_cls == b_id)[0]
                pool = repo.get(int(b_id), np.array([], dtype=float))

                idx = rng.randint(low=0, high=pool.size, size=len(rows))
                vals[rows] = pool[idx]
        elif per_bin_sampling == "eqf":
            vals = np.empty(n, dtype=float)
            for b_id in np.unique(sampled_cls):
                eqf = repo[colname][int(b_id)]
                # get relevant rows and sample from eqf
                rows = np.where(sampled_cls == b_id)[0]
                vals[rows] = eqf.rvs(size=len(rows), rng=rng)

    return vals


@dataclass
class _ColState:
    strategy: str
    keep: List[str]
    merged_labels: List[str]
    cat2merged: Dict[str, str]
    merged2dist: Dict[str, Dict[str, float]]
    other_cat_cols: List[str] = None
    scaler: Any = None
    ohe: Any = None
    kmedoids: Any = None
    medoid_labels: List[str] = None


class CategoryMerger:

    def __init__(
        self,
        K: int = 30,
        merge_type: str = "naive",  # "naive" or "cluster"
        discrete_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        random_state: int = 0,
        n_rare_clusters: int = 5,  # if None, picked automatically
        merged_prefix: str = "__MERGED__",
    ):
        if K < 2:
            raise ValueError("K must be >= 2.")
        if merge_type not in ("naive", "cluster"):
            raise ValueError("merge_type must be 'naive' or 'cluster'.")

        self.K = int(K)
        self.merge_type = merge_type
        self.discrete_columns = list(discrete_columns or [])
        self.numerical_columns = list(numerical_columns or [])
        self.random_state = int(random_state)
        self.n_rare_clusters = n_rare_clusters
        self.merged_prefix = merged_prefix

        self.col_state_: Dict[str, _ColState] = {}

    def fit_transform(
        self, X: pd.DataFrame, X_val: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], "CategoryMerger"]:
        if len(self.discrete_columns) == 0:
            return X, X_val
        self.fit(X, X_val=X_val)
        X_out = self.transform(X)
        X_val_out = None if X_val is None else self.transform(X_val)
        return X_out, X_val_out

    def fit(
        self, X: pd.DataFrame, X_val: Optional[pd.DataFrame] = None
    ) -> "CategoryMerger":

        disc = [c for c in self.discrete_columns if c in X.columns]
        num = [c for c in self.numerical_columns if c in X.columns]

        self.col_state_.clear()

        for col in disc:
            sX = X[col].astype(str)
            vc = sX.value_counts(dropna=False)
            uniques = vc.index.astype(str).tolist()

            if len(uniques) <= self.K:
                self.col_state_[col] = _ColState(
                    strategy="none",
                    keep=uniques,
                    merged_labels=[],
                    cat2merged={},
                    merged2dist={},
                )
                continue

            if self.merge_type == "naive":
                state = self._fit_naive(col, sX, vc)
            else:
                state = self._fit_cluster(col, X, X_val, disc, num, vc)

            self.col_state_[col] = state

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.col_state_:
            raise RuntimeError("Call fit() first.")
        X = X.copy()

        for col, st in self.col_state_.items():
            if col not in X.columns:
                continue

            if st.strategy == "none":
                X[col] = X[col].astype(str)
                continue

            s = X[col].astype(str)

            if st.strategy == "naive":
                merged_label = st.merged_labels[0]
                s = s.where(s.isin(st.keep), merged_label)
                X[col] = s
                continue

            if st.strategy == "cluster":
                mapped = s.map(st.cat2merged)
                out = s.copy()
                known_rare_mask = mapped.notna()
                out.loc[known_rare_mask] = mapped.loc[known_rare_mask]

                unseen_mask = ~out.isin(st.keep) & ~known_rare_mask
                if unseen_mask.any():
                    unseen_rows = X.loc[unseen_mask]
                    assign_map = self._assign_cat_to_cluster(
                        col=col,
                        rows=unseen_rows,
                        other_cat_cols=st.other_cat_cols,
                        scaler=st.scaler,
                        ohe=st.ohe,
                        kmedoids=st.kmedoids,
                        medoid_labels=st.medoid_labels,
                        num_cols=self.numerical_columns,
                    )
                    out.loc[unseen_mask] = out.loc[unseen_mask].map(assign_map)

                X[col] = out
                continue

            raise ValueError(f"Unknown strategy: {st.strategy}")

        return X

    def expand(self, X: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
        """
        Expand merged labels back to original categories using the training distributions.
        """
        if len(self.discrete_columns) == 0:
            return X
        if not self.col_state_:
            raise RuntimeError("Call fit() first.")
        X = X.copy()

        for col, st in self.col_state_.items():
            if col not in X.columns:
                continue

            if not st.merged2dist:
                X[col] = X[col].astype(str)
                continue

            s = X[col].astype(str)
            for merged_label, dist in st.merged2dist.items():
                if not dist:
                    continue
                mask = s == merged_label
                n = int(mask.sum())
                if n == 0:
                    continue

                cats = np.array(list(dist.keys()), dtype=object)
                probs = np.asarray(list(dist.values()), dtype=float)
                probs = probs / probs.sum()

                s.loc[mask] = rng.choice(cats, size=n, p=probs)

            X[col] = s

        return X

    def _fit_naive(self, col: str, vc: pd.Series) -> _ColState:
        keep = vc.index[: self.K - 1].astype(str).tolist()
        rare = vc.index[self.K - 1 :].astype(str).tolist()

        merged_label = f"{self.merged_prefix}{col}__RARE"
        cat2merged = {r: merged_label for r in rare}

        rare_counts = vc.loc[rare].astype(float)
        dist = (rare_counts / rare_counts.sum()).to_dict()

        return _ColState(
            strategy="naive",
            keep=keep,
            merged_labels=[merged_label],
            cat2merged=cat2merged,
            merged2dist={merged_label: dist},
        )

    def _fit_cluster(
        self,
        col: str,
        X: pd.DataFrame,
        X_val: Optional[pd.DataFrame],
        discrete_cols: List[str],
        num_cols: List[str],
        vc: pd.Series,
    ) -> _ColState:
        n_unique = len(vc)
        n_rare = n_unique

        n_clusters = int(self.n_rare_clusters)
        n_clusters = max(1, min(n_clusters, self.K - 1))  # at least 1 cluster label
        keep_n = self.K - n_clusters
        keep = vc.index[:keep_n].astype(str).tolist()
        rare = vc.index[keep_n:].astype(str).tolist()
        n_rare = len(rare)

        if n_rare == 0:
            return _ColState(
                strategy="none",
                keep=vc.index.astype(str).tolist(),
                merged_labels=[],
                cat2merged={},
                merged2dist={},
            )

        n_clusters = min(n_clusters, n_rare)

        other_cat_cols = [c for c in discrete_cols if c != col]

        X_fit_ohe = X.copy()
        if X_val is not None:
            X_fit_ohe = pd.concat([X_fit_ohe, X_val.copy()], axis=0, ignore_index=True)

        if num_cols:
            scaler = MinMaxScaler()
            scaler.fit(X_fit_ohe[num_cols].astype(float))
        else:
            scaler = None

        if other_cat_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe.fit(X_fit_ohe[other_cat_cols].astype(str))
        else:
            ohe = None

        source = X.copy()
        if X_val is not None:
            source = pd.concat([source, X_val.copy()], axis=0, ignore_index=True)

        source_col = source[col].astype(str)
        source = source.loc[source_col.isin(rare)].copy()
        source_col = source[col].astype(str)

        E = self._embed_rows(
            df=source,
            num_cols=num_cols,
            other_cat_cols=other_cat_cols,
            scaler=scaler,
            ohe=ohe,
        )

        rare_labels = []
        rare_vecs = []
        for r in rare:
            mask = (source_col == r).to_numpy()
            if mask.sum() == 0:
                continue
            rare_labels.append(r)
            rare_vecs.append(E[mask].mean(axis=0))

        if len(rare_labels) == 0:
            return self._fit_naive(col, X[col].astype(str), vc)

        rare_vecs = np.vstack(rare_vecs)
        n_clusters = min(n_clusters, len(rare_labels))

        kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric="manhattan",
            init="k-medoids++",
            max_iter=300,
            random_state=self.random_state,
        ).fit(rare_vecs)

        merged_labels = [f"{self.merged_prefix}{col}__C{i}" for i in range(n_clusters)]
        medoid_labels = merged_labels[:]  # aligned with cluster index

        cat2merged = {}
        for lab, cid in zip(rare_labels, kmedoids.labels_):
            cat2merged[lab] = merged_labels[int(cid)]

        merged2dist: Dict[str, Dict[str, float]] = {ml: {} for ml in merged_labels}
        train_counts = vc.astype(float)  # counts from training X only
        for ml in merged_labels:
            members = [c for c, m in cat2merged.items() if m == ml]
            if not members:
                continue
            counts = train_counts.loc[members]
            probs = (counts / counts.sum()).to_dict()
            merged2dist[ml] = probs

        return _ColState(
            strategy="cluster",
            keep=keep,
            merged_labels=merged_labels,
            cat2merged=cat2merged,
            merged2dist=merged2dist,
            other_cat_cols=other_cat_cols,
            scaler=scaler,
            ohe=ohe,
            kmedoids=kmedoids,
            medoid_labels=medoid_labels,
        )

    def _embed_rows(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        other_cat_cols: List[str],
        scaler: Optional[MinMaxScaler],
        ohe: Optional[OneHotEncoder],
    ) -> np.ndarray:
        parts = []

        if num_cols:
            Xn = df[num_cols].astype(float).to_numpy()
            if scaler is not None:
                Xn = scaler.transform(Xn)
            parts.append(Xn)

        if other_cat_cols:
            Xc = df[other_cat_cols].astype(str)
            if ohe is not None:
                Xc = ohe.transform(Xc) / 2.0
                parts.append(Xc)
            else:
                parts.append(np.empty((len(df), 0), dtype=float))

        if not parts:
            return np.zeros((len(df), 1), dtype=float)

        return np.concatenate(parts, axis=1)

    def _assign_cat_to_cluster(
        self,
        col: str,
        rows: pd.DataFrame,
        other_cat_cols: List[str],
        scaler: MinMaxScaler,
        ohe: OneHotEncoder,
        kmedoids: KMedoids,
        medoid_labels: List[str],
        num_cols: List[str],
    ) -> Dict[str, str]:
        s = rows[col].astype(str)
        unseen_cats = s.unique().tolist()

        E = self._embed_rows(
            df=rows,
            num_cols=[c for c in num_cols if c in rows.columns],
            other_cat_cols=[c for c in (other_cat_cols or []) if c in rows.columns],
            scaler=scaler,
            ohe=ohe,
        )

        assign_map: Dict[str, str] = {}
        for cat in unseen_cats:
            mask = (s == cat).to_numpy()
            v = E[mask].mean(axis=0, keepdims=True)
            cid = int(kmedoids.predict(v)[0])
            assign_map[cat] = medoid_labels[cid]

        return assign_map


def get_visit_order(
    df: pd.DataFrame,
    mode: str = "descending",
    method: str = "centrality",
):
    """
    Compute a dependency-based feature ordering for a discretized DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Discretized features, each column is integer-coded categorical.
    mode : {"descending", "ascending"}
        - "descending": most-dependent-first
        - "ascending":  least-dependent-first
    method : {"centrality", "chow-liu"}
        - "centrality": order by sum of normalized mutual information to others
        - "chow-liu":  build Chow–Liu tree and DFS it from a central root
        - "naive": order by column index

    Returns
    -------
    List[str]
        Column names ordered according to the chosen dependency criterion.
    """
    assert mode in ("descending", "ascending")
    assert method in ("centrality", "chow-liu", "naive")
    if method == "naive":
        return list(df.columns)

    cols = df.columns.to_numpy()
    D = len(cols)

    # ---------- 1. Pairwise MI ----------
    MI = np.zeros((D, D))
    for i, ci in enumerate(cols):
        xi = df[ci].to_numpy()
        for j in range(i + 1, D):
            cj = cols[j]
            xj = df[cj].to_numpy()
            mi = mutual_info_score(xi, xj)
            MI[i, j] = MI[j, i] = mi

    # ---------- 2. Entropy per feature ----------
    H = np.zeros(D)
    for i, ci in enumerate(cols):
        x = df[ci].to_numpy()
        # assume non-negative ints; if not, map to 0..K-1 first
        # (for discretized data this should already hold)
        counts = np.bincount(x)
        p = counts / counts.sum()
        p = p[p > 0]
        H[i] = -(p * np.log(p)).sum()

    # ---------- 3. Normalized MI ----------
    NMI = np.zeros_like(MI)
    for i in range(D):
        for j in range(D):
            denom = np.sqrt(H[i] * H[j])
            if denom > 0:
                NMI[i, j] = MI[i, j] / denom

    # ---------- 4. Dependency centrality ----------
    # centrality_j = sum_k NMI_jk
    centrality = NMI.sum(axis=1)

    if method == "centrality":
        # simple global ranking
        if mode == "descending":
            order_idx = np.argsort(-centrality)  # most-dependent-first
        else:
            order_idx = np.argsort(centrality)  # least-dependent-first

        return list(cols[order_idx])

    # ---------- 5. Chow–Liu tree construction ----------
    # Build maximum spanning tree on NMI weights (Kruskal)

    # collect edges (i < j)
    edges = []
    for i in range(D):
        for j in range(i + 1, D):
            w = NMI[i, j]
            edges.append((w, i, j))

    # sort edges by weight descending for max spanning tree
    edges.sort(key=lambda t: t[0], reverse=True)

    # Union-Find (Disjoint Set Union) for Kruskal
    parent = np.arange(D)
    rank = np.zeros(D, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    # adjacency list for MST
    adj = [[] for _ in range(D)]
    num_edges = 0
    for w, i, j in edges:
        if union(i, j):
            adj[i].append(j)
            adj[j].append(i)
            num_edges += 1
            if num_edges == D - 1:
                break

    # ---------- 6. Choose root based on dependency mode ----------
    if mode == "descending":
        root = int(np.argmax(centrality))  # most central as root
        neighbor_sort_key = lambda u, v: -NMI[u, v]  # visit strongest deps first
    else:
        root = int(np.argmin(centrality))  # least central as root
        neighbor_sort_key = lambda u, v: NMI[u, v]  # visit weakest deps first

    # ---------- 7. DFS traversal to get order ----------
    visited = np.zeros(D, dtype=bool)
    order = []

    stack = [root]
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)

        # sort neighbors by weight according to mode
        neighbors = [v for v in adj[u] if not visited[v]]
        neighbors.sort(key=lambda v: neighbor_sort_key(u, v))
        # DFS: push in reverse order so highest priority is popped first
        for v in reversed(neighbors):
            stack.append(v)

    # In (rare) case of disconnected graph (all MI=0), some nodes may be unvisited:
    if len(order) < D:
        remaining = [i for i in range(D) if not visited[i]]
        order.extend(remaining)

    return list(cols[np.array(order)])
