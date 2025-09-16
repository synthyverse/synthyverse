import pandas as pd
from typing import Union
from ..preprocessing.tabular import TabularPreprocessor


class TabularBaseGenerator:

    def __init__(
        self,
        constraints: Union[list, str] = [],
        missing_imputation_method: str = "drop",
        retain_missingness: bool = False,
        encode_mixed_numerical_features: bool = False,
        quantile_transform_numericals: bool = False,
        random_state: int = 0,
    ):
        self.random_state = random_state
        if isinstance(constraints, str):
            self.constraints = [constraints]
        else:
            self.constraints = constraints
        self.missing_imputation_method = missing_imputation_method
        self.retain_missingness = retain_missingness
        self.quantile_transform_numericals = quantile_transform_numericals
        self.encode_mixed_numerical_features = encode_mixed_numerical_features

    def fit(self, X: pd.DataFrame, discrete_features: list):
        self.base_discrete_features = discrete_features.copy()
        X_prep = X.copy()

        # execute a preprocessing pipeline
        self.preprocessor = TabularPreprocessor(
            discrete_features=self.base_discrete_features,
            random_state=self.random_state,
        )
        X_prep = self.preprocessor.pipeline(
            X=X_prep,
            missing_imputation_method=self.missing_imputation_method,
            retain_missingness=self.retain_missingness,
            encode_mixed_numerical_features=self.encode_mixed_numerical_features,
            quantile_transform_numericals=self.quantile_transform_numericals,
            constraints=self.constraints,
        )
        # update which features are discrete
        self.base_discrete_features = self.preprocessor.discrete_features.copy()

        self._fit_model(X=X_prep, discrete_features=self.base_discrete_features)

        return self

    def generate(self, n: int):
        syn_X = self._generate_data(n)

        # inverse the preprocessing pipeline
        syn_X = self.preprocessor.inverse_pipeline(syn_X)

        return syn_X

    def _fit_model(self, X: pd.DataFrame, discrete_features: list = []):
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _generate_data(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _generate_data")
