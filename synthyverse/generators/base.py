import pandas as pd
from ..utils.utils import calculate_column_precision


class BaseGenerator:

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, discrete_features: list):
        self.discrete_features = discrete_features
        self.numerical_features = [
            col for col in X.columns if col not in discrete_features
        ]
        self.X = X.copy()
        self.X = self.X.dropna(subset=self.numerical_features)
        self.ori_dtypes = X.dtypes
        self.ori_columns = X.columns
        self.ori_precision = {}
        for col in self.numerical_features:
            self.ori_precision[col] = calculate_column_precision(X[col])

        self._fit_model(X=self.X, discrete_features=self.discrete_features)

        return self

    def generate(self, n: int):
        syn_X = self._generate_data(n)
        syn_X = syn_X[self.ori_columns]

        # align precision of numerical columns
        for col in self.numerical_features:
            syn_X[col] = syn_X[col].round(self.ori_precision[col])

        # align dtypes with original datatypes
        syn_X = syn_X.astype(self.ori_dtypes)
        return syn_X

    def _fit_model(self, X: pd.DataFrame, discrete_features: list = None):
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _generate_data(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _generate_data")
