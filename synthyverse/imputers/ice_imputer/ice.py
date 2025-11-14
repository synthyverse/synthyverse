from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
from ..base import BaseImputer


class ICEImputer(BaseImputer):
    """Imputation by Chained Equations (ICE).

    Uses scikit-learn's IterativeImputer.

    Args:
        max_iter (int): Maximum number of imputation rounds. Default: 10.
        sample_posterior (bool): Whether to sample from the posterior distribution
            for imputation. Default: True.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to BaseImputer.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.imputers import ICEImputer
        >>>
        >>> # Load data with missing values
        >>> X = pd.read_csv("data_with_missing.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create and fit imputer
        >>> imputer = ICEImputer(
        ...     max_iter=10,
        ...     sample_posterior=True,
        ...     random_state=42
        ... )
        >>> imputer.fit(X, discrete_features)
        >>>
        >>> # Transform data
        >>> X_imputed = imputer.transform(X)
    """

    def __init__(
        self,
        max_iter: int = 10,
        sample_posterior: bool = True,
        random_state: int = 0,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.__dict__.update(locals())

    def _fit(self, X: pd.DataFrame):
        x = X.copy()
        # ensure numerically encoded data
        self.encoder = OrdinalEncoder()
        x[self.discrete_features] = self.encoder.fit_transform(
            x[self.discrete_features]
        ).astype(float)

        self.imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=self.sample_posterior,
        )

        self.ranges = {}
        for col in X.columns:
            self.ranges[col] = {
                "min": x[col].dropna().min(),
                "max": x[col].dropna().max(),
            }

        self.imputer.fit(x)

    def _transform(self, X: pd.DataFrame):
        x = X.copy()
        x[self.discrete_features] = self.encoder.transform(
            x[self.discrete_features]
        ).astype(float)
        imputed = self.imputer.transform(x)
        imputed = pd.DataFrame(imputed, columns=self.ori_cols)
        for col in X.columns:
            imputed[col] = np.clip(
                imputed[col], self.ranges[col]["min"], self.ranges[col]["max"]
            )
        imputed[self.discrete_features] = self.encoder.inverse_transform(
            imputed[self.discrete_features]
        )

        return imputed
