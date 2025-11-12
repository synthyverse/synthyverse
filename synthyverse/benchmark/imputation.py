import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from ..imputers import get_imputer
from ..utils.reproducibility import set_seed
from ..utils.utils import free_up_memory
from ..evaluation.eval import TabularMetricEvaluator
from ..evaluation import get_metric
from .utils import format_results


class TabularImputationBenchmark:
    def __init__(
        self,
        imputer_name: str = "ice",
        imputer_params: dict = {},
        metrics: list = ["mae_mad"],
        n_random_splits: int = 1,
        n_imputations: int = 1,
        test_size: float = 0.2,
        missingness_proportion: float = 0.2,
        result_format: str = "frame",
    ):

        self.imp_metrics = []
        for metric in metrics:
            if hasattr(get_metric(metric)(), "is_imputation_metric") and getattr(
                get_metric(metric), "is_imputation_metric", False
            ):
                self.imp_metrics.append(metric)
        self.syn_metrics = [x for x in metrics if x not in self.imp_metrics]
        self.__dict__.update(locals())

    def run(self, X: pd.DataFrame, discrete_features: list, target_column: str):
        # train test split
        stratify = None
        if target_column in discrete_features:
            stratify = X[target_column]

        results = {}
        for split_i in range(self.n_random_splits):
            X_train, X_test = train_test_split(
                X, test_size=self.test_size, random_state=split_i, stratify=stratify
            )
            results[f"split_{split_i}"] = {}

            for imp_i in range(self.n_imputations):
                results[f"split_{split_i}"][f"imputation_{imp_i}"] = {}
                set_seed(imp_i)
                # make 20% missingness (but remove observations with only missings)
                mask = (
                    np.random.rand(*X_train.shape) < self.missingness_proportion
                ).astype(bool)
                remove = mask.all(axis=1)
                mask = mask[~remove]
                X_imputation = X_train.copy()
                X_imputation[mask] = np.nan

                # perform imputation
                imputer = get_imputer(self.imputer_name)
                imputer = imputer(random_state=imp_i, **self.imputer_params)
                imputer.fit(X_imputation, discrete_features)
                imputed = imputer.transform(X_imputation)

                # evaluate full dataset
                evaluator = TabularMetricEvaluator(
                    metrics=self.syn_metrics,
                    discrete_features=discrete_features,
                    target_column=target_column,
                    random_state=imp_i,
                )
                metric_results = evaluator.evaluate(
                    X_train=X_train, X_test=X_test, X_syn=imputed
                )
                results[f"split_{split_i}"][f"imputation_{imp_i}"].update(
                    metric_results
                )

                # evaluate imputation only
                evaluator = TabularMetricEvaluator(
                    metrics=self.imp_metrics,
                    discrete_features=discrete_features,
                    target_column=target_column,
                    random_state=imp_i,
                )
                metric_results = evaluator.evaluate(
                    X_train=X_train[mask], X_test=X_test, X_syn=imputed[mask]
                )
                results[f"split_{split_i}"][f"imputation_{imp_i}"].update(
                    metric_results
                )

                # release memory for next iteration
                free_up_memory()

        if self.result_format == "frame":
            results = format_results(results)

        return results
