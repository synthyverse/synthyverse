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
