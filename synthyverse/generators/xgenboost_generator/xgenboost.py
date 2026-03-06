import pandas as pd
import numpy as np

from sklearn.utils import check_random_state

from typing import Union, Tuple

ArrayLike = Union[np.ndarray, list, Tuple[float, ...]]

from ..base import TabularBaseGenerator

from .utils import (
    get_bootstrap_repo,
    get_eqf_repo,
    discretize,
    label_encode,
    CategoryMerger,
    get_visit_order,
)


class XGenBoost(TabularBaseGenerator):

    def __init__(
        self,
        target_column: str,
        conditioning: str = "inference",  # "generation", "inference"
        use_early_stopping: bool = True,  # whether to use early stopping (if a validation set is provided)
        discretization: str = "quantile",  # uniform (equal-width), quantile (equal-height), kmeans (k-means clustering)
        n_bins: int = 30,  # maximum number of bins -> used to constrain token space and preserve privacy
        per_bin_sampling: str = "bootstrap",  # "gaussian_noise" or "uniform_noise" or "eqf" or "bootstrap"
        cat_merge_type: str = "cluster",  # "naive" or "clustering"
        cat_merge_n_infrequent: int = 5,  # number of infrequent categories to merge into
        visit_order_method: str = "centrality",  # "centrality", "chow-liu", "naive"
        visit_order_mode: str = "ascending",  # "descending", "ascending"
        random_state: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        self.__dict__.update(locals())
        self.rng = check_random_state(self.random_state)

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):

        self.ori_col_order = X.columns.tolist()

        self.discrete_columns = discrete_features
        self.numerical_columns = [
            x for x in X.columns if x not in self.discrete_columns
        ]

        self.X = X.copy()

        self.cat_merger = CategoryMerger(
            K=self.n_bins,
            merge_type=self.cat_merge_type,  # "naive" or "cluster"
            discrete_columns=self.discrete_columns,
            numerical_columns=self.numerical_columns,
            random_state=self.random_state,
            n_rare_clusters=self.cat_merge_n_infrequent,  # if None, picked automatically
            merged_prefix="__MERGED__",
        )
        self.X, val_X = self.cat_merger.fit_transform(
            self.X,
            X_val=X_val if (X_val is not None and self.use_early_stopping) else None,
        )

        X_enc = self.X.copy()

        if X_val is not None and self.use_early_stopping:
            val_X_enc = val_X.copy()
        else:
            val_X_enc = None

        X_enc, val_X_enc, self.discretizers = discretize(
            X_enc,
            val_X_enc,
            self.numerical_columns,
            self.n_bins,
            self.discretization,
            self.random_state,
            self.rng,
            self.use_early_stopping,
        )

        # label encode all features
        self.X, val_X, X_enc, val_X_enc, self.label_encoders = label_encode(
            self.X,
            val_X,
            X_enc,
            val_X_enc,
            self.discrete_columns,
            self.use_early_stopping,
        )

        self.repo = (
            get_bootstrap_repo(self.X, X_enc, self.numerical_columns)
            if self.per_bin_sampling == "bootstrap"
            else (
                get_eqf_repo(self.X, X_enc, self.numerical_columns)
                if self.per_bin_sampling == "eqf"
                else {}
            )
        )

        # apply visit order
        order = get_visit_order(
            X_enc[[c for c in X_enc.columns if c != self.target_column]],
            method=self.visit_order_method,
            mode=self.visit_order_mode,
        )
        order_ = (
            [self.target_column] + order
            if self.conditioning == "generation"
            else order + [self.target_column]
        )

        self.X = self.X[order_]
        X_enc = X_enc[order_]
        if val_X is not None:
            val_X = val_X[order_]
            val_X_enc = val_X_enc[order_]

        self._train_model(self.X, X_enc, val_X, val_X_enc)

    def _generate_data(self, n: int):
        syn = self._sample_data(n)

        for col in self.discrete_columns:
            syn[col] = self.label_encoders[col].inverse_transform(syn[col].astype(int))

        syn = self.cat_merger.expand(syn, self.rng)

        syn = syn.reset_index(drop=True)
        syn = syn[self.ori_col_order]

        return syn

    def _train_model(
        self,
        X: pd.DataFrame,
        X_enc: pd.DataFrame,
        val_X: pd.DataFrame = None,
        val_X_enc: pd.DataFrame = None,
    ):
        raise NotImplementedError("Subclasses must implement _train_model")

    def _sample_data(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _sample_data")
