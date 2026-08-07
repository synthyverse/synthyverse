from abc import ABC, abstractmethod

import pandas as pd


class BaseMetric(ABC):
    """Base class for tabular evaluation metrics.

    Args:
        random_state (int): Random seed for reproducible metric behavior.
            Default: 0.
    """

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def _validate_feature_mix(
        self, X: pd.DataFrame, discrete_features: list, name: str
    ) -> None:
        missing_features = [col for col in discrete_features if col not in X.columns]
        if missing_features:
            missing = ", ".join(missing_features)
            raise ValueError(f"discrete_features are not present in {name}: {missing}")

        non_numerical_features = [
            col
            for col in X.columns
            if col not in discrete_features
            and not pd.api.types.is_numeric_dtype(X[col])
        ]
        if non_numerical_features:
            non_numerical = ", ".join(non_numerical_features)
            raise ValueError(
                f"Non-discrete features must be numerical in {name}: {non_numerical}"
            )

    def _validate_matching_columns(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
    ) -> None:
        expected = list(X.columns)
        for name, data in (("X_syn", X_syn), ("X_test", X_test)):
            if data is None:
                continue
            missing = [col for col in expected if col not in data.columns]
            extra = [col for col in data.columns if col not in expected]
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"missing columns: {missing}")
                if extra:
                    parts.append(f"extra columns: {extra}")
                raise ValueError(
                    f"{name} must have the same columns as X ({'; '.join(parts)})."
                )

    def evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ) -> dict:
        """Evaluate synthetic data against real data.

        Args:
            X: Real data used as the main reference dataset.
            X_syn: Synthetic data to evaluate.
            X_test: Optional independent real test data. Metrics that need a
                held-out real dataset require this argument.
            discrete_features (list): Column names that should be treated as
                discrete/categorical features. Default: None.

        Returns:
            dict: Metric results keyed by metric and score names.
        """
        discrete_features = list(discrete_features or [])
        self._validate_matching_columns(X, X_syn, X_test)
        self._validate_feature_mix(X, discrete_features, "X")
        self._validate_feature_mix(X_syn, discrete_features, "X_syn")
        if X_test is not None:
            self._validate_feature_mix(X_test, discrete_features, "X_test")

        x = X.copy().reset_index(drop=True)
        x_syn = X_syn[x.columns].copy().reset_index(drop=True)
        x_test = (
            None if X_test is None else X_test[x.columns].copy().reset_index(drop=True)
        )
        return self._evaluate(
            x,
            x_syn,
            X_test=x_test,
            discrete_features=discrete_features,
        )

    @abstractmethod
    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ) -> dict:
        """Metric-specific implementation called by evaluate()."""
