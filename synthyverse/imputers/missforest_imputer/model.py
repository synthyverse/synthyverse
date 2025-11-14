### Taken from https://github.com/epsilon-machine/missingpy/blob/master/missingpy/missforest.py

"""MissForest Imputer for Missing Data"""
# Author: Ashim Bhattarai
# License: GNU General Public License v3 (GPLv3)

import warnings

import numpy as np
from scipy.stats import mode

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

__all__ = [
    "MissForest",
]


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


class MissForest(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        max_iter=10,
        decreasing=False,
        missing_values=np.nan,
        copy=True,
        n_estimators=100,
        criterion=("squared_error", "gini"),
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
    ):

        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

    def _miss_forest(self, Ximp, mask):
        """The missForest algorithm"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get("col_means")
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols
            )

            # Reg criterion
            reg_criterion = (
                self.criterion if type(self.criterion) == str else self.criterion[0]
            )

            # Instantiate regression model
            rf_regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=reg_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
            )

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get("col_modes")
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(
                col_modes, missing_cat_cols
            )

            # Classfication criterion
            clf_criterion = (
                self.criterion if type(self.criterion) == str else self.criterion[1]
            )

            # Instantiate classification model
            rf_classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=clf_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                class_weight=self.class_weight,
            )

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])

        while (
            gamma_new < gamma_old or gamma_newcat < gamma_oldcat
        ) and self.iter_count_ < self.max_iter:

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    rf_classifier.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_classifier.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis
                else:
                    rf_regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                if n_catmissing == 0:
                    gamma_newcat = 0
                else:
                    gamma_newcat = (
                        np.sum((Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_]))
                        / n_catmissing
                    )
            if self.num_vars_ is not None:
                gamma_new = np.sum(
                    (Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2
                ) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            print("Iteration:", self.iter_count_)
            self.iter_count_ += 1

        return Ximp_old

    def fit(self, X, y=None, cat_vars=None):

        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN", np.nan] else True

        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite=force_all_finite,
            copy=self.copy,
        )

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Check cat_vars type and convert if necessary
        if cat_vars is not None:
            if type(cat_vars) == int:
                cat_vars = [cat_vars]
            elif type(cat_vars) == list or type(cat_vars) == np.ndarray:
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array " "of ints."
                    )
            else:
                raise ValueError(
                    "cat_vars needs to be either an int or an array " "of ints."
                )

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else None

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ["NaN", np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = np.nanmean(X[:, num_vars], axis=0) if num_vars is not None else None
        col_modes = (
            mode(X[:, cat_vars], axis=0, nan_policy="omit", keepdims=True)[0]
            if cat_vars is not None
            else None
        )

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}

        return self

    def transform(self, X):

        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_", "statistics_"])

        # Check data integrity
        force_all_finite = False if self.missing_values in ["NaN", np.nan] else True
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite=force_all_finite,
            copy=self.copy,
        )

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (0 if self.num_vars_ is None else len(self.num_vars_)) + (
            0 if self.cat_vars_ is None else len(self.cat_vars_)
        )
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError(
                "Incompatible dimension between the fitted "
                "dataset and the one to be transformed."
            )

        # Check if anything is actually missing and if not return original X
        mask = _get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original " "dataset.")
            return X

        # row_total_missing = mask.sum(axis=1)
        # if not np.any(row_total_missing):
        #     return X

        # Call missForest function to impute missing
        X = self._miss_forest(X, mask)

        # Return imputed dataset
        return X

    def fit_transform(self, X, y=None, **fit_params):

        return self.fit(X, **fit_params).transform(X)
