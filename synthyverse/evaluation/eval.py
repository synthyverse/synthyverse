from typing import Union
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ..utils.missingness import (
    apply_fitted_numeric_imputer,
    apply_random_imputation,
    drop_rows_with_missing_numericals,
    fit_random_imputation_samples,
    fit_transform_numeric_imputer,
    validate_missing_imputation_method,
)
from . import get_metric


class TabularMetricEvaluator:
    """Evaluator for tabular synthetic data quality metrics.

    This class provides a unified interface for evaluating synthetic data quality
    across the dimensions fidelity, utility, and privacy, using various metrics.

    Args:
        metrics (Union[dict, list]): Dictionary mapping metric names to their parameters, or list of metric names (will use default parameters). Dictionaries can be used to specify metric hyperparameters, and compute different configurations of the same metric.
        discrete_features (list): List of column names that are discrete/categorical. Default: [].
        target_column (str): Name of the target column for supervised metrics. Default: "target".
        missing_imputation_method (str): Method for handling missing values. "drop" removes missing rows, other options perform imputation: "random", "mean", "median", "most_frequent", "missforest". Default: "drop".
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
        missing_imputation_method: str = "drop",
        random_state: int = 0,
    ):

        if isinstance(metrics, list):
            self.metrics = {metric: {} for metric in metrics}
        else:
            self.metrics = metrics
        self.discrete_features = (
            list(discrete_features) if discrete_features is not None else []
        )
        self.target_column = target_column
        self.missing_imputation_method = missing_imputation_method
        self.random_state = random_state

    def _apply_missingness_preprocessing(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_val: pd.DataFrame = None,
    ):
        numerical_features = [
            col for col in X_train.columns if col not in self.discrete_features
        ]
        if len(numerical_features) == 0:
            return X_train, X_test, X_syn, X_val

        validate_missing_imputation_method(self.missing_imputation_method)

        if self.missing_imputation_method == "drop":
            X_train = drop_rows_with_missing_numericals(X_train, numerical_features)
            X_test = drop_rows_with_missing_numericals(X_test, numerical_features)
            X_syn = drop_rows_with_missing_numericals(X_syn, numerical_features)
            if X_val is not None:
                X_val = drop_rows_with_missing_numericals(X_val, numerical_features)
            return X_train, X_test, X_syn, X_val

        if self.missing_imputation_method == "keep":
            return X_train, X_test, X_syn, X_val

        if self.missing_imputation_method == "random":
            rng = np.random.default_rng(self.random_state)
            X_train = apply_random_imputation(
                X=X_train,
                numerical_features=numerical_features,
                imputation_samples=fit_random_imputation_samples(
                    X_train, numerical_features
                ),
                rng=rng,
            )
            X_test = apply_random_imputation(
                X=X_test,
                numerical_features=numerical_features,
                imputation_samples=fit_random_imputation_samples(
                    X_test, numerical_features
                ),
                rng=rng,
            )
            X_syn = apply_random_imputation(
                X=X_syn,
                numerical_features=numerical_features,
                imputation_samples=fit_random_imputation_samples(
                    X_syn, numerical_features
                ),
                rng=rng,
            )
            if X_val is not None:
                X_val = apply_random_imputation(
                    X=X_val,
                    numerical_features=numerical_features,
                    imputation_samples=fit_random_imputation_samples(
                        X_val, numerical_features
                    ),
                    rng=rng,
                )
            return X_train, X_test, X_syn, X_val

        X_train, real_imputer, real_all_missing_columns = fit_transform_numeric_imputer(
            X=X_train,
            numerical_features=numerical_features,
            imputation_method=self.missing_imputation_method,
            random_state=self.random_state,
        )
        X_test = apply_fitted_numeric_imputer(
            X=X_test,
            numerical_features=numerical_features,
            imputer=real_imputer,
            imputation_method=self.missing_imputation_method,
            all_missing_numeric_columns=real_all_missing_columns,
        )
        if X_val is not None:
            X_val = apply_fitted_numeric_imputer(
                X=X_val,
                numerical_features=numerical_features,
                imputer=real_imputer,
                imputation_method=self.missing_imputation_method,
                all_missing_numeric_columns=real_all_missing_columns,
            )

        # synthetic data must use an imputer fitted on synthetic data itself
        X_syn, _, _ = fit_transform_numeric_imputer(
            X=X_syn,
            numerical_features=numerical_features,
            imputation_method=self.missing_imputation_method,
            random_state=self.random_state,
        )
        return X_train, X_test, X_syn, X_val

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

        X_train, X_test, X_syn, X_val = self._apply_missingness_preprocessing(
            X_train=X_train,
            X_test=X_test,
            X_syn=X_syn,
            X_val=X_val,
        )

        X_train, X_test, X_syn = (
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            X_syn.reset_index(drop=True),
        )
        if X_val is not None:
            X_val = X_val.reset_index(drop=True)

        # ensure discrete features are integer encoded
        available_discrete_features = [
            col for col in self.discrete_features if col in X_train.columns
        ]
        if len(available_discrete_features) > 0:
            ordinal_encoder = OrdinalEncoder(dtype=int)
            (
                ordinal_encoder.fit(
                    pd.concat(
                        [
                            X_train[available_discrete_features],
                            X_test[available_discrete_features],
                            X_syn[available_discrete_features],
                        ]
                    )
                )
                if X_val is None
                else ordinal_encoder.fit(
                    pd.concat(
                        [
                            X_train[available_discrete_features],
                            X_test[available_discrete_features],
                            X_syn[available_discrete_features],
                            X_val[available_discrete_features],
                        ]
                    )
                )
            )
            X_train[available_discrete_features] = ordinal_encoder.transform(
                X_train[available_discrete_features]
            )
            X_test[available_discrete_features] = ordinal_encoder.transform(
                X_test[available_discrete_features]
            )
            X_syn[available_discrete_features] = ordinal_encoder.transform(
                X_syn[available_discrete_features]
            )
            if X_val is not None:
                X_val[available_discrete_features] = ordinal_encoder.transform(
                    X_val[available_discrete_features]
                )

        dict_ = {}
        for metric__ in self.metrics.keys():
            print(f"Evaluating metric: {metric__}")
            metric_ = metric__.split("-")[0].strip().lower()
            metric_cls = get_metric(metric_)

            params = dict(self.metrics[metric__])
            if getattr(metric_cls, "needs_discrete_features", False):
                params["discrete_features"] = available_discrete_features
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
                    "instead of dict - result discarded."
                )

        return dict_
