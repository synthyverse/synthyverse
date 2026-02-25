from typing import Union
import warnings

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from . import get_metric


class TabularMetricEvaluator:
    """Evaluator for tabular synthetic data quality metrics.

    This class provides a unified interface for evaluating synthetic data quality
    across the dimensions fidelity, utility, and privacy, using various metrics.

    Args:
        metrics (Union[dict, list]): Dictionary mapping metric names to their parameters, or list of metric names (will use default parameters). Dictionaries can be used to specify metric hyperparameters, and compute different configurations of the same metric.
        discrete_features (list): List of column names that are discrete/categorical. Default: [].
        target_column (str): Name of the target column for supervised metrics. Default: "target".
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import TabularMetricEvaluator
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>> target_column = "target"
        >>>
        >>> # Define metrics
        >>> metrics = ["mle"]
        >>>
        >>> # Compute different configurations of the same metric by adding a dash to the metric name
        >>> metrics = {"mle-trts": {"train_set":"real"}, "mle-tstr": {"train_set":"synthetic", "tune":True}}
        >>>
        >>> # Create evaluator
        >>> evaluator = TabularMetricEvaluator(
        ...     metrics=metrics,
        ...     discrete_features=discrete_features,
        ...     target_column=target_column
        ... )
        >>>
        >>> # Evaluate synthetic data
        >>> results = evaluator.evaluate(X_train, X_test, X_syn)
    """

    def __init__(
        self,
        metrics: Union[dict, list],
        discrete_features: list = None,
        target_column: str = "target",
        random_state: int = 0,
    ):

        if isinstance(metrics, list):
            self.metrics = {metric: {} for metric in metrics}
        else:
            self.metrics = metrics
        self.discrete_features = list(discrete_features) if discrete_features is not None else []
        self.target_column = target_column
        self.random_state = random_state

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_val: pd.DataFrame = None,
    ):
        """Evaluate synthetic data quality using specified metrics.

        Args:
            X_train: Training data as a pandas DataFrame.
            X_test: Test data as a pandas DataFrame.
            X_syn: Synthetic data as a pandas DataFrame.
            X_val: Optional validation data as a pandas DataFrame.

        Returns:
            dict: Dictionary mapping metric names to their evaluation results.
        """
        X_train, X_test, X_syn = X_train.copy(), X_test.copy(), X_syn.copy()
        if X_val is not None:
            X_val = X_val.copy()

        # drop missings in numericals
        numerical_features = [
            col for col in X_train.columns if col not in self.discrete_features
        ]
        X_train, X_test, X_syn = (
            X_train.dropna(subset=numerical_features),
            X_test.dropna(subset=numerical_features),
            X_syn.dropna(subset=numerical_features),
        )
        if X_val is not None:
            X_val = X_val.dropna(subset=numerical_features)

        X_train, X_test, X_syn = (
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            X_syn.reset_index(drop=True),
        )

        if X_val is not None:
            X_val = X_val.reset_index(drop=True)

        # ensure discrete features are integer encoded
        ordinal_encoder = OrdinalEncoder()
        (
            ordinal_encoder.fit(
                pd.concat(
                    [
                        X_train[self.discrete_features],
                        X_test[self.discrete_features],
                        X_syn[self.discrete_features],
                    ]
                )
            )
            if X_val is None
            else ordinal_encoder.fit(
                pd.concat(
                    [
                        X_train[self.discrete_features],
                        X_test[self.discrete_features],
                        X_syn[self.discrete_features],
                        X_val[self.discrete_features],
                    ]
                )
            )
        )
        X_train[self.discrete_features] = ordinal_encoder.transform(
            X_train[self.discrete_features]
        )
        X_test[self.discrete_features] = ordinal_encoder.transform(
            X_test[self.discrete_features]
        )
        X_syn[self.discrete_features] = ordinal_encoder.transform(
            X_syn[self.discrete_features]
        )
        if X_val is not None:
            X_val[self.discrete_features] = ordinal_encoder.transform(
                X_val[self.discrete_features]
            )

        dict_ = {}
        for metric__ in self.metrics.keys():
            print(f"Evaluating metric: {metric__}")
            metric_ = metric__.split("-")[0].strip().lower()
            metric_cls = get_metric(metric_)

            params = dict(self.metrics[metric__])
            if getattr(metric_cls, "needs_discrete_features", False):
                params["discrete_features"] = self.discrete_features
            if getattr(metric_cls, "needs_target_column", False):
                params["target_column"] = self.target_column
            if getattr(metric_cls, "needs_random_state", False):
                params["random_state"] = self.random_state
            if getattr(metric_cls, "needs_val_set", False):
                params["X_val"] = X_val

            metric = metric_cls(**params)

            data_req = getattr(metric_cls, "data_requirement", None)
            if data_req == "test":
                metric_result = metric.evaluate(
                    X_test,
                    X_syn[-len(X_test) :],
                )
            elif data_req == "train":
                metric_result = metric.evaluate(
                    X_train,
                    X_syn[: len(X_train)],
                )
            elif data_req == "train_and_test":
                metric_result = metric.evaluate(X_train, X_test, X_syn)
            else:
                raise ValueError(
                    f"Metric {metric_} not (fully) implemented or missing data_requirement property"
                )

            if isinstance(metric_result, dict):
                dict_.update(metric_result)
            else:
                warnings.warn(
                    f"Metric '{metric__}' returned {type(metric_result).__name__} "
                    f"instead of dict — result discarded."
                )

        return dict_
