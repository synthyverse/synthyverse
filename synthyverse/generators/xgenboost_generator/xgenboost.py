from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    StandardScaler,
    QuantileTransformer,
)
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import check_random_state
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
from scipy.stats import truncnorm

from ..base import BaseGenerator
from ...utils.xgb_utils import get_xgb_tree_method
from time import time


class XGenBoostGenerator(BaseGenerator):
    name = "xgenboost"

    # TBD:
    # - speed-up preprocessing and inference through cuML library
    # - speed up fitting through XGB params (e.g. max_bin)
    # probably best to first compute the benchmarks, then decide how much speed-ups/improvements are required

    # - allow to pass causal column order when causal masking
    # - test imputation capabilites
    # - allow more complex types of discretization
    # - add other backends, e.g., LightGBM

    def __init__(
        self,
        duplicate_K: int = 1,  # number of times to replicate the random masking procedure (extends training set)
        val_size: float = 0.2,  # no early stopping if <=0
        early_stopping_rounds: int = 50,
        temperature: float = 1.0,
        n_estimators: int = 100,
        max_depth: int = 6,
        mask_style: str = "random",  # "random" or "causal"
        discretization: str = "uniform",  # uniform (equal-width) or quantile (equal-height) -> quantile can be better for privacy
        max_bins: int = 30,  # maximum number of bins -> used to constrain token space (and preserve privacy for quantile bins)
        min_samples_per_bin: int = 10,  # ensures minimum number of samples per bin if discretization==quantile
        uniform_bins: str = "auto",  # how to compute histogram bins when discretization==uniform: "max", "auto", "fd", "doane", "scott", etc.
        bootstrap_first: bool = True,  # whether to bootstrap first column to be generated
        add_target_idx: bool = True,  # whether to add target feature index during modelling
        batch_size: int = 100_000,  # inference batch size
        merge_infrequent_categories: bool = True,  # whether to merge infrequent categories (cardinality>max_bins)
        backend: str = "xgboost",  # "xgboost" or "rf" or "catboost"
        max_tree_bins: int = 256,  # number of bins within tree ensembles
        numerical_scaling: str = "quantile",  # "quantile" or "none" (or more to be added...)
        per_bin_noise: str = "gaussian",  # "uniform" or "gaussian"
        n_gibbs_sweeps: int = 2,  # number of Gibbs sweeps for generation
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)
        self.duplicate_K = duplicate_K
        self.n_gibbs_sweeps = n_gibbs_sweeps
        self.val_size = val_size
        self.backend = backend
        self.early_stopping_rounds = early_stopping_rounds
        self.max_tree_bins = max_tree_bins
        self.max_bins = max_bins
        self.temperature = temperature
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.numerical_scaling = numerical_scaling
        self.mask_style = mask_style
        self.bootstrap_first = bootstrap_first
        self.add_target_idx = add_target_idx
        self.batch_size = batch_size
        self.per_bin_noise = per_bin_noise
        self.discretization = discretization
        self.min_samples_per_bin = min_samples_per_bin
        self.merge_infrequent_categories = merge_infrequent_categories
        self.uniform_bins = uniform_bins
        self.random_state = random_state

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        self.discrete_columns = discrete_features
        self.numerical_columns = [
            x for x in X.columns if x not in self.discrete_columns
        ]
        self.X = X.copy()
        if self.merge_infrequent_categories:
            self.X[self.discrete_columns], self.merge_info = (
                self._merge_infrequent_categories(
                    self.X[self.discrete_columns], self.max_bins - 1
                )
            )
        X_enc = self.X.copy()

        # scale numerical features
        if self.numerical_scaling == "quantile":
            self.scaler = QuantileTransformer(
                output_distribution="normal",
                random_state=self.random_state,
                subsample=100_000,
            )
            self.X[self.numerical_columns] = self.scaler.fit_transform(
                self.X[self.numerical_columns]
            )
            X_enc[self.numerical_columns] = self.scaler.transform(
                X_enc[self.numerical_columns]
            )

        # discretize numerical features
        self.discretizers = {}
        for col in self.numerical_columns:
            # determine number of bins
            if self.uniform_bins == "max" and self.discretization == "uniform":
                n_bins = self.max_bins
            elif self.discretization == "uniform":
                n_bins = np.histogram_bin_edges(X_enc[col], bins=self.uniform_bins)
                n_bins = min(self.max_bins, len(n_bins) - 1)
            elif self.discretization == "quantile":
                n_bins = min(
                    self.max_bins,
                    int(np.floor(self.X.shape[0] / self.min_samples_per_bin)),
                )
            else:
                raise ValueError(f"Discretization {self.discretization} not supported")

            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                strategy=self.discretization,
                encode="ordinal",
                random_state=self.random_state,
            )
            X_enc[col] = discretizer.fit_transform(X_enc[col].values.reshape(-1, 1))
            self.discretizers[col] = discretizer

        # label encode all features
        self.label_encoders = {}
        for col in X_enc.columns:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col])
            # required for sampling later
            if col in self.discrete_columns:
                self.X[col] = le.transform(self.X[col])
            self.label_encoders[col] = le

        # Split train/val if needed
        if self.val_size > 0:
            self.X, self.X_val, X_enc, X_enc_val = train_test_split(
                self.X, X_enc, test_size=self.val_size, random_state=self.random_state
            )
            rng = check_random_state(self.random_state)
            masked_X_val, y_val = self._build_masked_dataset(
                self.X_val,
                X_enc_val,
                rng,
                self.mask_style,
                self.duplicate_K,
                self.add_target_idx,
            )
            # free some memory
            del X_enc_val
            del self.X_val
            masked_X_val[self.discrete_columns] = masked_X_val[
                self.discrete_columns
            ].astype("category")
            if self.backend == "catboost":
                masked_X_val[self.discrete_columns] = masked_X_val[
                    self.discrete_columns
                ].astype(str)
            masked_X_val[self.numerical_columns] = masked_X_val[
                self.numerical_columns
            ].astype(float)

        # Build masked dataset for training
        rng = check_random_state(self.random_state)
        masked_X, y = self._build_masked_dataset(
            self.X, X_enc, rng, self.mask_style, self.duplicate_K, self.add_target_idx
        )
        # free some memory
        del X_enc
        masked_X[self.discrete_columns] = masked_X[self.discrete_columns].astype(
            "category"
        )
        if self.backend == "catboost":
            masked_X[self.discrete_columns] = masked_X[self.discrete_columns].astype(
                str
            )
        masked_X[self.numerical_columns] = masked_X[self.numerical_columns].astype(
            float
        )

        self.model = self._make_model()
        eval_set = [(masked_X_val, y_val)] if self.val_size > 0 else None
        self.model.fit(masked_X, y, eval_set=eval_set)

    def _generate_data(self, n: int):
        if self.batch_size > n:
            self.batch_size = n
        syn_batches = []
        for _ in range(0, n, self.batch_size):
            syn_batches.append(self._generate_batch(self.batch_size))
        syn = pd.concat(syn_batches, axis=0)
        return syn

    def _generate_batch(self, n: int):
        d = self.X.shape[1]
        cols = self.X.columns.tolist()

        # one permutation per row
        if self.mask_style == "random":
            orders = np.vstack([np.random.permutation(d) for _ in range(n)])
        else:
            orders = np.tile(np.arange(d), (n, 1))

        # synthetic dataset init
        syn = pd.DataFrame(np.nan, index=range(n), columns=cols)

        for pos in tqdm(range(d), desc="Generating synthetic data"):
            target_cols = orders[:, pos]

            if pos == 0 and self.bootstrap_first:
                for col_idx in np.unique(target_cols):
                    colname = cols[col_idx]
                    rows = np.where(target_cols == col_idx)[0]
                    syn.loc[rows, colname] = (
                        self.X[colname]
                        .sample(n=len(rows), replace=True)
                        .reset_index(drop=True)
                        .to_numpy()
                    )
                continue

            curr_X = syn.copy()

            if self.backend == "catboost":
                curr_X[self.discrete_columns] = curr_X[self.discrete_columns].astype(
                    str
                )
            if self.add_target_idx:
                curr_X["idx_feature"] = target_cols

            start_time = time()
            probs_all = self.model.predict_proba(curr_X)

            # ---- vectorized sampling ----
            for col_idx in np.unique(target_cols):
                colname = cols[col_idx]
                rows = np.where(target_cols == col_idx)[0]

                classes = np.arange(len(self.label_encoders[colname].classes_))
                probs = probs_all[rows, : len(classes)]

                syn.loc[rows, colname] = self._sample_from_posterior(
                    probs, colname, len(rows)
                )

        # --- systematic Gibbs sweeps (mask target col across all rows) ---
        if getattr(self, "n_gibbs_sweeps", 0) > 0:
            n_rows, d = syn.shape
            cols = self.X.columns.tolist()

            for sweep in range(self.n_gibbs_sweeps):
                # randomize column order each sweep; fixed is also fine
                col_order = np.random.permutation(d)

                for col_idx in col_order:
                    colname = cols[col_idx]

                    # copy and MASK the target column for ALL rows
                    curr_X = syn.copy()
                    curr_X.iloc[:, col_idx] = np.nan

                    # match inference dtypes to training dtypes
                    if self.backend == "catboost":
                        # CatBoost saw strings for categoricals during training (including 'nan')
                        curr_X[self.discrete_columns] = curr_X[
                            self.discrete_columns
                        ].astype(str)
                    else:
                        # XGBoost with enable_categorical expects pandas 'category'
                        curr_X[self.discrete_columns] = curr_X[
                            self.discrete_columns
                        ].astype("category")

                    if self.add_target_idx:
                        # constant target column id for all rows
                        curr_X["idx_feature"] = col_idx

                    # single predict_proba for this column (vectorized over rows)
                    probs = self.model.predict_proba(curr_X)

                    # slice to this column's class space and sample
                    classes = np.arange(len(self.label_encoders[colname].classes_))
                    probs = probs[:, : len(classes)]

                    syn[colname] = self._sample_from_posterior(probs, colname, n_rows)
        # reinstate categories, numerical scaling, etc.
        for col in self.discrete_columns:
            syn[col] = self.label_encoders[col].inverse_transform(syn[col].astype(int))
        if self.merge_infrequent_categories:
            syn[self.discrete_columns] = self._expand_merged_categories(
                syn[self.discrete_columns], self.merge_info
            )

        if self.numerical_scaling == "quantile":
            syn[self.numerical_columns] = self.scaler.inverse_transform(
                syn[self.numerical_columns]
            )

        syn = syn.reset_index(drop=True)
        return syn

    def _sample_from_posterior(self, probs, colname, n):
        # temperature scaling
        probs = np.exp(np.log(probs + 1e-10) / self.temperature)
        probs /= probs.sum(axis=1, keepdims=True)

        # inverse-CDF vectorized sampling
        cum_probs = np.cumsum(probs, axis=1)
        r = np.random.rand(n, 1)
        sampled = (cum_probs < r).sum(axis=1)

        if colname in self.discrete_columns:
            return sampled
        else:
            # map back to bin centers + noise
            classes_ = self.label_encoders[colname].classes_
            sampled_cls = classes_[sampled].astype(int)

            edges = self.discretizers[colname].bin_edges_[0]
            left = edges[sampled_cls]
            right = edges[sampled_cls + 1]
            centers = 0.5 * (left + right)
            widths = right - left

            if self.per_bin_noise == "uniform":
                # zero-mean uniform, per-sample width
                eps = (np.random.rand(n) - 0.5) * widths
                vals = centers + eps

            elif self.per_bin_noise == "gaussian":
                # per-sample sigma; clamp to avoid div-by-zero
                sigmas = np.maximum(widths / 6, 1e-12)
                # truncated around zero, within the bin
                a = (left - centers) / sigmas
                b = (right - centers) / sigmas
                # draw zero-mean noise, no 'size' so shapes broadcast to (n,)
                eps = truncnorm.rvs(a, b, loc=0.0, scale=sigmas)
                vals = centers + eps

            return vals

    def _generate_column(self, X: pd.DataFrame, colname: str, col_idx: int):
        curr_X = X.copy()
        if self.add_target_idx:
            curr_X["idx_feature"] = col_idx

        probs = self.model.predict_proba(curr_X)
        classes = np.arange(len(self.label_encoders[colname].classes_))
        probs = probs[:, : len(classes)]
        x_pred = self._sample_from_posterior(probs, colname, len(curr_X))

        return x_pred

    def _build_masked_dataset(
        self, X, X_enc, rng, mask_style, duplicate_K, add_target_idx
    ):
        N, D = X.shape
        cols = X.columns.tolist()
        Xv = X.values
        Xev = X_enc.values

        masked_blocks = []
        target_blocks = []
        idx_blocks = []

        pbar = tqdm(range(duplicate_K), desc="Building masked dataset")

        for k_ in pbar:
            pbar.set_description(f"Building masked dataset: {k_}")
            if mask_style == "causal":
                # Predict feature j using features < j (j = 1..D-1)
                row_idx = np.repeat(np.arange(N), D - 1)  # (N*(D-1),)
                target_idx = np.tile(np.arange(1, D), N)  # (N*(D-1),)
                M = Xv[row_idx].copy()  # (N*(D-1), D)
                mask = np.arange(D)[None, :] >= target_idx[:, None]
                M[mask] = np.nan
                yk = Xev[row_idx, target_idx]  # (N*(D-1),)
                idxk = target_idx

            elif mask_style == "random":
                # Row-wise permutation; at step t keep rank < t, predict rank == t
                keys = rng.random((N, D))
                P = np.argsort(keys, axis=1)  # (N, D)
                R = np.empty_like(P)
                R[np.arange(N)[:, None], P] = np.arange(D)[
                    None, :
                ]  # inverse perm (ranks)

                steps = D - 1
                row_idx = np.repeat(np.arange(N), steps)  # (N*(D-1),)
                tvec = np.tile(np.arange(1, D), N)  # (N*(D-1),)
                target_idx = P[:, 1:].reshape(-1)  # (N*(D-1),)

                M = Xv[row_idx].copy()
                R_rep = R[row_idx]
                mask = R_rep >= tvec[:, None]
                M[mask] = np.nan

                yk = Xev[row_idx, target_idx]
                idxk = target_idx
            else:
                raise ValueError("mask_style must be 'causal' or 'random'")

            masked_blocks.append(M)
            target_blocks.append(yk)
            idx_blocks.append(idxk)

        masked_data = np.vstack(masked_blocks) if masked_blocks else np.empty((0, D))
        y_all = np.concatenate(target_blocks) if target_blocks else np.empty((0,))
        idx_all = (
            np.concatenate(idx_blocks) if idx_blocks else np.empty((0,), dtype=int)
        )

        masked_X = pd.DataFrame(masked_data, columns=cols)
        if add_target_idx:
            masked_X["idx_feature"] = idx_all
        y = pd.DataFrame(y_all, columns=["target"])

        # drop rows in X and y where y is nan
        masked_X["TEMP_TARGET_Y"] = y
        masked_X = masked_X.dropna(subset=["TEMP_TARGET_Y"])
        y = y.dropna()
        masked_X = masked_X.drop(columns=["TEMP_TARGET_Y"])

        return masked_X, y

    def _impute(self, X: pd.DataFrame, n_imputations: int = 1):
        if self.mask_style == "causal":
            raise Exception(
                "Causal masking style not supported for imputation. Use random masking style instead."
            )

        # preprocess X like expected by the XGBClassifier
        X_enc = X.copy()
        for col in self.discrete_columns:
            X_enc[col] = self.label_encoders[col].transform(X[col])

        # generate random column order for each imputation batch
        imputed_X = []
        for _ in range(n_imputations):
            column_order = np.random.permutation(X.shape[1])
            for col in column_order:
                colname = X.columns.tolist()[col]
                # only predict nans in imputation
                mask = X_enc[colname].isna()
                X_enc.loc[mask, colname] = self._generate_column(
                    X_enc[mask], colname, col
                )
            # reverse preprocessing
            for col in self.discrete_columns:
                X_enc[col] = self.label_encoders[col].inverse_transform(X_enc[col])
            imputed_X.append(X_enc)

        return imputed_X if len(imputed_X) > 1 else imputed_X[0]

    def _merge_infrequent_categories(
        self,
        X: pd.DataFrame,
        max_categories: int,
        merged_label: str = "PIWCREVIUHWEOC",  # random string
    ):
        """
        Merge infrequent categories across all columns in X into a single merged_label.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with categorical/discrete features.
        max_categories : int
            Maximum number of categories to keep per column. The least frequent
            categories exceeding this count are merged.
        merged_label : str
            Label name for merged categories.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns modified.
        dict
            Merge info per column for inversion.
        """
        X = X.copy()
        merge_info = {}

        for col in X.columns:
            vc = X[col].value_counts(dropna=False)
            keep = vc.index[:max_categories]
            merge = vc.index[max_categories:]

            if len(merge) > 0:
                merge_dist = vc[merge] / vc[merge].sum()
                X.loc[~X[col].isin(keep), col] = merged_label
                merge_info[col] = {
                    "merged_label": merged_label,
                    "distribution": merge_dist.to_dict(),
                }
            else:
                # nothing merged in this column
                merge_info[col] = {
                    "merged_label": merged_label,
                    "distribution": {},
                }

        return X, merge_info

    def _expand_merged_categories(
        self,
        X: pd.DataFrame,
        merge_info: dict,
        random_state: int = None,
    ):
        """
        Expand merged categories back into original categories by random sampling.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with merged categories.
        merge_info : dict
            Dict from merge_infrequent_categories() with per-column distributions.
        random_state : int
            Random state for reproducibility.

        Returns
        -------
        pd.DataFrame
            Dataframe with expanded categories.
        """
        rng = np.random.default_rng(random_state)
        X = X.copy()

        for col in X.columns:
            info = merge_info.get(col, None)
            if info is None:
                continue

            merged_label = info["merged_label"]
            dist = info["distribution"]

            if not dist:
                continue  # nothing was merged

            categories = list(dist.keys())
            probs = np.array(list(dist.values()))

            mask = X[col] == merged_label
            n_replace = mask.sum()
            if n_replace > 0:
                sampled = rng.choice(categories, size=n_replace, p=probs)
                X.loc[mask, col] = sampled

        return X

    def _make_model(self):
        common_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            early_stopping_rounds=(
                self.early_stopping_rounds if self.val_size > 0 else None
            ),
            max_bin=self.max_tree_bins,
        )
        if self.backend == "xgboost":
            return XGBClassifier(
                **common_params,
                enable_categorical=True,
                tree_method=get_xgb_tree_method(),
            )
        elif self.backend == "rf":
            return XGBRFClassifier(
                **common_params,
                enable_categorical=True,
                tree_method=get_xgb_tree_method(),
            )
        elif self.backend == "catboost":
            return CatBoostClassifier(
                **common_params,
                cat_features=self.discrete_columns,
                task_type="GPU" if get_xgb_tree_method() == "gpu_hist" else "CPU",
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")
