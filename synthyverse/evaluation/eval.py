from typing import Union
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from . import get_metric


class TabularMetricEvaluator:

    def __init__(
        self,
        metrics: Union[dict, list],
        discrete_features: list = [],
        target_column: str = "target",
        random_state: int = 0,
    ):

        if isinstance(metrics, list):
            self.metrics = {metric: {} for metric in metrics}
        else:
            self.metrics = metrics
        self.discrete_features = discrete_features.copy()
        self.target_column = target_column
        self.random_state = random_state

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_val: pd.DataFrame = None,
    ):
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
            # Use class properties to determine which additional information needs to be passed to the metric
            if hasattr(metric_cls, "needs_discrete_features") and getattr(
                metric_cls, "needs_discrete_features", False
            ):
                self.metrics[metric__]["discrete_features"] = self.discrete_features
            if hasattr(metric_cls, "needs_target_column") and getattr(
                metric_cls, "needs_target_column", False
            ):
                self.metrics[metric__]["target_column"] = self.target_column
            if hasattr(metric_cls, "needs_random_state") and getattr(
                metric_cls, "needs_random_state", False
            ):
                self.metrics[metric__]["random_state"] = self.random_state

            if hasattr(metric_cls, "needs_val_set") and getattr(
                metric_cls, "needs_val_set", False
            ):
                self.metrics[metric__]["X_val"] = X_val

            metric = metric_cls(**self.metrics[metric__])
            # Use class property to determine which data to pass
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
                raise Exception(
                    f"Metric {metric_} not (fully) implemented or missing data_requirement property"
                )

            # add result to dict (note that quantitative metrics have to output a dict, else they won't get added here)
            if type(metric_result) == dict:
                dict_.update(metric_result)

        return dict_
