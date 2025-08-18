from sklearn.model_selection import train_test_split
from ..evaluation.eval import MetricEvaluator
from ..utils.utils import get_generator, free_up_memory
from ..utils.reproducibility import set_seed
import pandas as pd
from time import time


class TabularBenchmark:
    def __init__(
        self,
        generator_name: str = "arf",
        generator_params: dict = {},
        n_random_splits: int = 1,
        n_inits: int = 1,
        n_generated_datasets: int = 1,
        metrics: list = ["classifier_test", "mle", "dcr"],
        test_size: float = 0.3,
    ):

        self.generator_name = generator_name
        self.generator_params = generator_params
        self.n_random_splits = n_random_splits
        self.n_inits = n_inits
        self.n_generated_datasets = n_generated_datasets
        self.metrics = metrics
        self.test_size = test_size

    def run(self, X: pd.DataFrame, target_column: str, discrete_columns: list):

        results = {}
        generator_ = get_generator(self.generator_name)
        for split_i in range(self.n_random_splits):
            results[f"split_{split_i}"] = {}

            # split data according to current seed
            stratify = None
            if target_column in discrete_columns:
                stratify = X[target_column]
            X_train, X_test = train_test_split(
                X, stratify=stratify, test_size=self.test_size, random_state=split_i
            )
            X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(
                drop=True
            )

            for init_i in range(self.n_inits):
                results[f"split_{split_i}"][f"init_{init_i}"] = {}
                set_seed(init_i)
                generator = generator_(random_state=init_i, **self.generator_params)
                start_time = time()
                generator.fit(X_train, discrete_columns)
                results[f"split_{split_i}"][f"init_{init_i}"]["training_time"] = (
                    time() - start_time
                )

                # potentially generate multiple datasets
                for generated_dataset_i in range(self.n_generated_datasets):
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ] = {}
                    start_time = time()
                    X_syn = generator.generate(len(X))
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ] = {}
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ]["inference_time"] = (time() - start_time)
                    start_time = time()
                    evaluator = MetricEvaluator(
                        metrics=self.metrics,
                        discrete_features=discrete_columns,
                        target_column=target_column,
                        random_state=init_i,
                    )
                    metric_results = evaluator.evaluate(X_train, X_test, X_syn)
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ]["evaluation_time"] = (time() - start_time)
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ].update(metric_results)

                    # free up memory for next iteration
                    free_up_memory()

        return results
