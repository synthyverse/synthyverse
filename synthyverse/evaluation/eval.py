from typing import Union
import warnings
import inspect

import pandas as pd
from . import get_metric


def _split_metric_config(metric_key: str):
    metric_name, separator, config_slug = str(metric_key).partition("-")
    metric_name = metric_name.strip().lower()
    config_slug = config_slug.strip() if separator else None

    if not metric_name:
        raise ValueError("Metric name cannot be empty.")
    if config_slug == "":
        config_slug = None

    return metric_name, config_slug


def _with_metric_config_slug_in_key(
    metric_name: str, result_key: str, config_slug: str = None
) -> str:
    """Place the user-provided config slug between metric and score names.

    The returned key starts with the registry metric name and preserves the
    result key tail so dotted metric names and score names remain intact, e.g.
    ``mle.tstr.train_synthetic_test_real.auc``.
    """
    result_key = str(result_key)
    key_tail = (
        result_key[len(metric_name) + 1 :]
        if result_key.startswith(f"{metric_name}.")
        else result_key
    )
    key_parts = [metric_name]
    if config_slug is not None:
        key_parts.append(config_slug)
    if key_tail:
        key_parts.append(key_tail)
    return ".".join(key_parts)


class TabularMetricEvaluator:
    """Evaluator for tabular synthetic data quality metrics.

    This class provides a unified interface for evaluating synthetic data quality
    across the dimensions fidelity, utility, and privacy, using various metrics.

    Args:
        metrics (Union[dict, list]): Dictionary mapping metric names to their parameters, or list of metric names (will use default parameters). Dictionaries can be used to specify metric hyperparameters, and compute different configurations of the same metric (see the example below).
        discrete_features (list): List of column names that are discrete/categorical. Default: [].
        target_column (str): Name of the target column for supervised metrics. Default: "target".
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation.eval import TabularMetricEvaluator
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
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            raise ValueError("metrics must be a list or a dictionary")
        self.discrete_features = (
            list(discrete_features) if discrete_features is not None else []
        )
        self.target_column = target_column
        self.random_state = random_state

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_syn_test: pd.DataFrame = None,
        X_val: pd.DataFrame = None,
    ):
        """Evaluate synthetic data quality using specified metrics.

        Args:
            X_train: Training data as a pandas DataFrame.
            X_test: Test data as a pandas DataFrame.
            X_syn: Synthetic data as a pandas DataFrame.
            X_syn_test: Optional synthetic test data as a pandas DataFrame.
            X_val: Optional validation data as a pandas DataFrame.

        Returns:
            dict: Dictionary mapping metric names to their evaluation results.
        """
        x_train, x_test, x_syn = X_train.copy(), X_test.copy(), X_syn.copy()
        x_syn_test = X_syn_test.copy() if X_syn_test is not None else None
        x_val = X_val.copy() if X_val is not None else None

        # reset indices for proper indexing/slicing
        x_train, x_test, x_syn = (
            x_train.reset_index(drop=True),
            x_test.reset_index(drop=True),
            x_syn.reset_index(drop=True),
        )
        if x_val is not None:
            x_val = x_val.reset_index(drop=True)
        if x_syn_test is not None:
            x_syn_test = x_syn_test.reset_index(drop=True)

        dict_ = {}
        for metric__ in self.metrics.keys():
            print(f"Evaluating metric: {metric__}")
            metric_, config_slug = _split_metric_config(metric__)
            metric_cls = get_metric(metric_)

            # add necessary fixed parameters to the metrics:
            metric_params = dict(self.metrics[metric__])
            params = dict(metric_params)
            required_params = inspect.signature(metric_cls.__init__).parameters.keys()
            if "discrete_features" in required_params:
                params["discrete_features"] = self.discrete_features
            if "target_column" in required_params:
                params["target_column"] = self.target_column
            if "random_state" in required_params:
                params["random_state"] = self.random_state
            data = {
                "X_train": x_train,
                "X_syn": x_syn,
            }
            required_data = inspect.signature(metric_cls.evaluate).parameters.keys()
            if "X_test" in required_data:
                data["X_test"] = x_test
            if "X_val" in required_data:
                data["X_val"] = x_val
            if "X_syn_test" in required_data:
                data["X_syn_test"] = x_syn_test

            metric = metric_cls(**params)
            metric_result = metric.evaluate(**data)

            if isinstance(metric_result, dict):
                for key, value in metric_result.items():
                    result_key = _with_metric_config_slug_in_key(
                        metric_, key, config_slug
                    )
                    if result_key in dict_:
                        raise ValueError(
                            f"Duplicate metric result key '{result_key}' while "
                            f"evaluating '{metric__}'. Use distinct metric names "
                            "or config slugs so result keys remain unique."
                        )
                    dict_[result_key] = value
            else:
                warnings.warn(
                    f"Metric '{metric__}' returned {type(metric_result).__name__} "
                    "instead of dict - result discarded."
                )

        return dict_
