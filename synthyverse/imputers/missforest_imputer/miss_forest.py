import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from .model import MissForest
from ..base import BaseImputer


class MissForestImputer(BaseImputer):
    """Iterative imputation using random forests.

    Using implementation from https://github.com/epsilon-machine/missingpy/blob/master/missingpy/missforest.py

    Paper: "MissForest: Non-parametric missing value imputation for mixed-type data" by Stekhoven and Bühlmann (2012).

    Args:
        max_iter (int): Maximum number of imputation rounds. Default: 10.
        decreasing (bool): Whether to sort features by amount of missing data. Default: False.
        missing_values (float): Placeholder for missing values. Default: np.nan.
        copy (bool): Whether to create a copy of the input data. Default: True.
        n_estimators (int): Number of trees in the random forest. Default: 100.
        criterion (tuple): Splitting criterion tuple (for regression, classification). Default: ("squared_error", "gini").
        max_depth (int, optional): Maximum depth of trees. Default: None.
        min_samples_split (int): Minimum samples required to split a node. Default: 2.
        min_samples_leaf (int): Minimum samples required in a leaf node. Default: 1.
        min_weight_fraction_leaf (float): Minimum weighted fraction in a leaf node. Default: 0.0.
        max_features (float): Number of features to consider for best split. Default: 1.0.
        max_leaf_nodes (int, optional): Maximum number of leaf nodes. Default: None.
        min_impurity_decrease (float): Minimum impurity decrease for split. Default: 0.0.
        bootstrap (bool): Whether to use bootstrap sampling. Default: True.
        oob_score (bool): Whether to compute out-of-bag score. Default: False.
        n_jobs (int): Number of parallel jobs (-1 for all cores). Default: -1.
        verbose (int): Verbosity level. Default: 0.
        warm_start (bool): Whether to reuse previous solution. Default: False.
        class_weight (dict, optional): Class weights for classification. Default: None.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to BaseImputer.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.imputers import MissForestImputer
        >>>
        >>> # Load data with missing values
        >>> X = pd.read_csv("data_with_missing.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create and fit imputer
        >>> imputer = MissForestImputer(
        ...     n_estimators=100,
        ...     n_jobs=-1,
        ...     random_state=42
        ... )
        >>> imputer.fit(X, discrete_features)
        >>>
        >>> # Transform data
        >>> X_imputed = imputer.transform(X)
    """

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
        verbose=0,
        warm_start=False,
        class_weight=None,
        random_state: int = 0,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.__dict__.update(locals())

    def _fit(self, X: pd.DataFrame):
        self.ori_cols = X.columns.tolist()

        # missforest expects label encoded categoricals
        self.encoder = OrdinalEncoder()
        X[self.discrete_features] = self.encoder.fit_transform(
            X[self.discrete_features]
        )

        self.imputer = MissForest(
            max_iter=10,
            decreasing=self.decreasing,
            missing_values=self.missing_values,
            copy=self.copy,
            n_estimators=self.n_estimators,
            criterion=self.criterion,
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

        cat_vars = [X.columns.get_loc(x) for x in self.discrete_features]

        self.imputer.fit(X, cat_vars=cat_vars)

    def _transform(self, X: pd.DataFrame):
        imputed = self.imputer.transform(X)
        imputed = pd.DataFrame(imputed, columns=self.ori_cols)
        imputed[self.discrete_features] = self.encoder.inverse_transform(
            imputed[self.discrete_features]
        )
        return imputed
