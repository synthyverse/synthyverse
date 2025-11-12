from ..base import TabularBaseGenerator
import pandas as pd
from .fd_dir.fd import ForestDiffusionModel
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


class ForestDiffusionGenerator(TabularBaseGenerator):
    name = "forestdiffusion"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        duplicate_K: int = 100,
        noise_level: int = 50,
        diffusion_type: str = "flow",
        n_jobs: int = -1,
        max_depth: int = 7,
        n_estimators: int = 100,
        eta: float = 0.3,
        tree_method: str = "hist",
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        subsample: float = 1.0,
        num_leaves: int = 31,
        eps: float = 1e-3,
        beta_min: float = 0.1,
        beta_max: float = 8,
        n_z: int = 10,
        gpu_hist: bool = False,
        random_state: int = 0,
        max_rows_in_memory: int = 100_000,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.target_column = target_column
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta
        self.gpu_hist = gpu_hist
        self.tree_method = tree_method
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_z = n_z
        self.n_jobs = n_jobs
        self.duplicate_K = duplicate_K
        self.diffusion_type = diffusion_type
        self.noise_level = noise_level
        self.max_rows_in_memory = max_rows_in_memory

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        # dynamically set n_batch based on max_rows_in_memory
        # >= 1 so always use data iterator
        self.n_batch = int(np.ceil(X.shape[0] / self.max_rows_in_memory))

        self.ori_col_order = X.columns
        self.discrete_features = discrete_features.copy()
        self.X = X.copy()

        # ordinally encode all features
        self.ordinal_encoder = OrdinalEncoder()
        self.X[self.discrete_features] = self.ordinal_encoder.fit_transform(
            self.X[self.discrete_features]
        )

        # separate target column in case of classification
        if self.target_column in self.discrete_features:
            self.y = self.X[self.target_column]
            self.X = self.X.drop(columns=[self.target_column])
            self.y = self.y.to_numpy()
        else:
            self.y = None

        bin_features = []
        cat_features = []
        for col in self.X.columns:
            if col in self.discrete_features:
                if self.X[col].nunique() == 2:
                    bin_features.append(col)
                else:
                    cat_features.append(col)

        bin_indexes = [self.X.columns.get_loc(x) for x in bin_features]
        cat_indexes = [self.X.columns.get_loc(x) for x in cat_features]
        int_indexes = []  # already handled by basegenerator

        self.X = self.X.to_numpy()

        self.model = ForestDiffusionModel(
            self.X,  # Numpy dataset
            X_covs=None,  # Numpy dataset of additional covariates/features in order to sample X | X_covs (Optional); note that these variables will not be transformed, please apply your own z-scoring or min-max scaling if desired.
            label_y=self.y,  # must be a categorical/binary variable; if provided will learn multiple models for each label y
            n_t=self.noise_level,  # number of noise level
            model="xgboost",  # xgboost, random_forest, lgbm, catboost
            diffusion_type=self.diffusion_type,  # vp, flow (flow is better, but only vp can be used for imputation)
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            eta=self.eta,  # xgboost hyperparameters
            tree_method=self.tree_method,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,  # xgboost hyperparameters
            num_leaves=self.num_leaves,  # lgbm hyperparameters
            duplicate_K=self.duplicate_K,  # number of different noise sample per real data sample
            bin_indexes=bin_indexes,  # vector which indicates which column is binary
            cat_indexes=cat_indexes,  # vector which indicates which column is categorical (>=3 categories)
            int_indexes=int_indexes,  # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
            remove_miss=False,  # If True, we remove the missing values, this allow us to train the XGBoost using one model for all predictors; otherwise we cannot do it
            p_in_one=True,  # When possible (when there are no missing values), will train the XGBoost using one model for all predictors
            true_min_max_values=None,  # Vector or None of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
            gpu_hist=self.gpu_hist,  # using GPU or not with xgboost
            n_z=self.n_z,  # number of noise to use in zero-shot classification
            eps=self.eps,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            n_jobs=self.n_jobs,  # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
            n_batch=self.n_batch,  # If >0 use the data iterator with the specified number of batches
            seed=self.random_state,
        )
        # free some memory
        del self.X
        del self.y

    def _generate_data(self, n: int):
        syn = self.model.generate(batch_size=n)

        # put back in original column order
        if self.target_column in self.discrete_features:
            syn_X = pd.DataFrame(
                syn[:, :-1],
                columns=[x for x in self.ori_col_order if x != self.target_column],
            )
            syn_y = pd.DataFrame(
                np.expand_dims(syn[:, -1], axis=1), columns=[self.target_column]
            )
            syn = pd.concat([syn_X, syn_y], axis=1)
            syn = syn[self.ori_col_order]
        else:
            syn = pd.DataFrame(syn, columns=self.ori_col_order)

        # remove ordinal encoding
        syn[self.discrete_features] = self.ordinal_encoder.inverse_transform(
            syn[self.discrete_features]
        )

        return syn
