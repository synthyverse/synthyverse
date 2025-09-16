from typing import Union
import pandas as pd

from ..preprocessing.tabular import TabularPreprocessor

from .fidelity import (
    ClassifierTest,
    AlphaPrecisionBetaRecallAuthenticity,
    Similarity,
)

from .utility import MLE

from .privacy import DCR, DOMIAS

METRICS = {
    "classifier_test": ClassifierTest,
    "mle": MLE,
    "dcr": DCR,
    "similarity": Similarity,
    "prauth": AlphaPrecisionBetaRecallAuthenticity,
    "domias": DOMIAS,
}


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
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, X_syn: pd.DataFrame
    ):
        X_train, X_test, X_syn = (
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            X_syn.reset_index(drop=True),
        )

        # ensure that we do not evaluate a larger real dataset than synthetic
        X_train, X_test = X_train[: len(X_syn)], X_test[: len(X_syn)]
        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)

        # impute missingness
        X_all = pd.concat([X_train, X_test, X_syn])
        X_all = X_all.reset_index(drop=True)
        preprocessor = TabularPreprocessor(
            discrete_features=self.discrete_features, random_state=self.random_state
        )
        X_all, _ = preprocessor.impute_missings(
            X_all,
            method="drop",
            add_missing_indicator=False,
        )

        # onehot and scale the data
        X_all = preprocessor.scale(
            X_all,
            numerical_transformer="none",
            categorical_transformer="one-hot",
        )
        X_tr_scaled, X_te_scaled, X_syn_scaled = (
            X_all.iloc[: len(X_train)],
            X_all.iloc[len(X_train) : len(X_train) + len(X_test)],
            X_all.iloc[len(X_train) + len(X_test) :],
        )
        X_tr_scaled, X_te_scaled, X_syn_scaled = (
            X_tr_scaled.reset_index(drop=True),
            X_te_scaled.reset_index(drop=True),
            X_syn_scaled.reset_index(drop=True),
        )
        X_tr_scaled = preprocessor.scale(
            X_tr_scaled,
            numerical_transformer="standard",
            categorical_transformer="none",
        )
        X_te_scaled = preprocessor.scale(
            X_te_scaled,
            numerical_transformer="standard",
            categorical_transformer="none",
        )
        X_syn_scaled = preprocessor.scale(
            X_syn_scaled,
            numerical_transformer="standard",
            categorical_transformer="none",
        )

        dict_ = {}
        for metric__ in self.metrics.keys():
            metric_ = metric__.split("-")[0].strip().lower()
            metric_cls = METRICS[metric_]
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
            elif data_req == "test_preprocessed":
                metric_result = metric.evaluate(
                    X_te_scaled,
                    X_syn_scaled[-len(X_test) :],
                )
            elif data_req == "train_preprocessed":
                metric_result = metric.evaluate(
                    X_tr_scaled,
                    X_syn_scaled[: len(X_train)],
                )
            elif data_req == "train_and_test":
                metric_result = metric.evaluate(X_train, X_test, X_syn)
            elif data_req == "train_and_test_preprocessed":
                metric_result = metric.evaluate(
                    X_tr_scaled,
                    X_te_scaled,
                    X_syn_scaled,
                )
            else:
                raise Exception(
                    f"Metric {metric_} not (fully) implemented or missing data_requirement property"
                )

            # add result to dict (note that quantitative metrics have to output a dict, else they won't get added here)
            if type(metric_result) == dict:
                dict_.update(metric_result)

        return dict_
