import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from .model import MissForest
from ..base import BaseImputer


class MissForestImputer(BaseImputer):

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
