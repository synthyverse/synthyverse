from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import os
import shutil
from typing import Union

from .utils import format_results
from ..evaluation.eval import TabularMetricEvaluator
from ..generators import get_generator
from ..utils.utils import free_up_memory
from ..utils.reproducibility import set_seed


class TabularSynthesisBenchmark:
    def __init__(
        self,
        generator_name: str = "arf",
        generator_params: dict = {},
        n_random_splits: int = 1,
        n_inits: int = 1,
        n_generated_datasets: int = 1,
        metrics: list = ["classifier_test", "mle", "dcr"],
        test_size: float = 0.2,
        val_size: float = 0.1,
        missing_imputation_method: str = "drop",
        retain_missingness: bool = False,
        encode_mixed_numerical_features: bool = False,
        quantile_transform_numericals: bool = False,
        constraints: Union[str, list] = [],
        max_syn_size: int = int(1e9),
        workspace: str = "workspace",
    ):

        self.generator_name = generator_name
        self.max_syn_size = max_syn_size
        self.generator_params = generator_params
        self.generator_params.pop(
            "target_column", None
        )  # target column already provided if required
        self.generator_params.pop(
            "workspace", None
        )  # workspace already provided if needed
        self.generator_params.pop("random_state", None)  # use loop-based random_states

        self.generator_params.update(
            {
                "missing_imputation_method": missing_imputation_method,
                "retain_missingness": retain_missingness,
                "encode_mixed_numerical_features": encode_mixed_numerical_features,
                "quantile_transform_numericals": quantile_transform_numericals,
                "constraints": constraints,
            }
        )

        self.n_random_splits = n_random_splits
        self.n_inits = n_inits
        self.n_generated_datasets = n_generated_datasets
        self.metrics = metrics
        self.test_size = test_size
        self.val_size = val_size
        self.workspace = workspace

    def run(
        self,
        X: pd.DataFrame,
        target_column: str,
        discrete_columns: list,
        result_format: str = "frame",  # "frame" or "dict"
    ):
        os.makedirs(self.workspace, exist_ok=True)

        results = {}
        generator_ = get_generator(self.generator_name)
        # add workspace if needed
        if hasattr(generator_, "needs_workspace") and getattr(
            generator_, "needs_workspace", False
        ):
            self.generator_params["workspace"] = self.workspace
        # add target column if needed
        if hasattr(generator_, "needs_target_column") and getattr(
            generator_, "needs_target_column", False
        ):
            self.generator_params["target_column"] = target_column

        for split_i in range(self.n_random_splits):
            # remove any previously tuned hyperparameters; they need to be re-tuned for different training splits
            shutil.rmtree("synthyverse_hyperparams_tuned", ignore_errors=True)
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
            if self.val_size > 0:
                stratify = None
                if target_column in discrete_columns:
                    stratify = X_train[target_column]
                X_train, X_val = train_test_split(
                    X_train,
                    stratify=stratify,
                    test_size=self.val_size
                    / (
                        1 - self.test_size
                    ),  # val_size is a proportion of the training set
                    random_state=split_i,
                )
                X_train, X_val = X_train.reset_index(drop=True), X_val.reset_index(
                    drop=True
                )
            else:
                X_val = None

            for init_i in range(self.n_inits):
                results[f"split_{split_i}"][f"init_{init_i}"] = {}
                set_seed(init_i)
                # reset workspace each time we fit the generator
                generator = None
                self.clean_directory(self.workspace, remove_self=False)
                generator = generator_(random_state=init_i, **self.generator_params)
                start_time = time()
                generator.fit(
                    X=X_train, discrete_features=discrete_columns, X_val=X_val
                )
                results[f"split_{split_i}"][f"init_{init_i}"]["training_time"] = (
                    time() - start_time
                )

                # potentially generate multiple datasets
                for generated_dataset_i in range(self.n_generated_datasets):
                    set_seed(generated_dataset_i)
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ] = {}

                    # calculate sampling time for 1000 samples
                    start_time = time()
                    generator.generate(1000)
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ]["inference_time_1k_samples"] = (time() - start_time)

                    # sample synthetic dataset and perform evaluation
                    n_train = min(self.max_syn_size, len(X_train))
                    n_test = min(self.max_syn_size, len(X_test))
                    n_val = min(self.max_syn_size, len(X_val))
                    n = n_train + n_test
                    start_time = time()
                    X_syn = generator.generate(n)
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ]["inference_time"] = (time() - start_time)
                    evaluator = TabularMetricEvaluator(
                        metrics=self.metrics,
                        discrete_features=discrete_columns,
                        target_column=target_column,
                        random_state=generated_dataset_i,
                    )
                    start_time = time()
                    metric_results = evaluator.evaluate(
                        X_train.sample(
                            n_train, replace=False, random_state=generated_dataset_i
                        ),
                        X_test.sample(
                            n_test, replace=False, random_state=generated_dataset_i
                        ),
                        X_syn,
                        X_val.sample(
                            n_val, replace=False, random_state=generated_dataset_i
                        ),
                    )
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ]["evaluation_time"] = (time() - start_time)
                    results[f"split_{split_i}"][f"init_{init_i}"][
                        f"generated_dataset_{generated_dataset_i}"
                    ].update(metric_results)

                    # release memory for next iteration
                    free_up_memory()
        # remove the workspace which was used for intermediate storage
        generator = None
        self.clean_directory(self.workspace, remove_self=True)
        shutil.rmtree("synthyverse_hyperparams_tuned", ignore_errors=True)
        if result_format == "frame":
            results = format_results(results)
        return results

    def clean_directory(self, path: str, remove_self: bool = False) -> None:
        """
        Remove all files and subdirectories inside a directory.
        If remove_self=True, remove the directory itself as well.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")

        if remove_self:
            shutil.rmtree(path)
        else:
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.remove(entry_path)
                elif os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
