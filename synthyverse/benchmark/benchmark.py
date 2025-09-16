from sklearn.model_selection import train_test_split
from ..evaluation.eval import TabularMetricEvaluator
from ..utils.utils import get_generator, free_up_memory
from ..utils.reproducibility import set_seed
import pandas as pd
from time import time
import os
import shutil
from typing import Union


class TabularBenchmark:
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
        workspace: str = "workspace",
    ):

        self.generator_name = generator_name
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

            for init_i in range(self.n_inits):
                results[f"split_{split_i}"][f"init_{init_i}"] = {}
                set_seed(init_i)
                # reset workspace each time we fit the generator
                generator = None
                self.clean_directory(self.workspace, remove_self=False)
                generator = generator_(random_state=init_i, **self.generator_params)
                start_time = time()
                # pass validation data if needed
                if hasattr(generator_, "needs_validation_set") and getattr(
                    generator_, "needs_validation_set", False
                ):
                    generator.fit(X_train, X_val, discrete_columns)
                else:
                    generator.fit(X_train, discrete_columns)
                results[f"split_{split_i}"][f"init_{init_i}"]["training_time"] = (
                    time() - start_time
                )

                # potentially generate multiple datasets
                for generated_dataset_i in range(self.n_generated_datasets):
                    set_seed(generated_dataset_i)
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
                    evaluator = TabularMetricEvaluator(
                        metrics=self.metrics,
                        discrete_features=discrete_columns,
                        target_column=target_column,
                        random_state=generated_dataset_i,
                    )
                    metric_results = evaluator.evaluate(X_train, X_test, X_syn)
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
        if result_format == "frame":
            results = self.format_results(results)
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

    def format_results(self, data, value_at_level=3):
        """
        A generalistic function that formats nested dictionary data into a dataframe.
        Each row represents one metric with columns for metric name, value, and context.

        Args:
            data: Nested dictionary containing metric values
            value_at_level: The dictionary level where metric key-value pairs reside

        Returns:
            pd.DataFrame: Formatted dataframe with metric, value, and context columns
        """

        def extract_metrics_recursive(d, current_level=0, context_dict=None):
            """Recursively extract metrics at the target level"""
            if context_dict is None:
                context_dict = {}

            rows = []

            for key, value in d.items():
                if isinstance(value, dict):
                    if current_level == value_at_level - 1:
                        # We've reached the level where metrics are stored
                        # Each key-value pair becomes a row
                        for metric_key, metric_value in value.items():
                            row = context_dict.copy()
                            row[f"context{current_level + 1}"] = (
                                key  # Include current level key as context
                            )
                            row["metric"] = metric_key
                            row["value"] = metric_value
                            rows.append(row)
                    else:
                        # Continue traversing and collect context
                        new_context = context_dict.copy()
                        new_context[f"context{current_level + 1}"] = key
                        rows.extend(
                            extract_metrics_recursive(
                                value, current_level + 1, new_context
                            )
                        )
                else:
                    # Leaf node - if we're at the right level, this is a metric
                    if current_level == value_at_level - 1:
                        row = context_dict.copy()
                        row[f"context{current_level + 1}"] = (
                            key  # Include current level key as context
                        )
                        row["metric"] = key
                        row["value"] = value
                        rows.append(row)
                    else:
                        # This is context information at a higher level
                        context_dict[f"context{current_level + 1}"] = key

            return rows

        # Extract all rows
        all_rows = extract_metrics_recursive(data)

        if not all_rows:
            # Fallback: if no rows found, try to find the deepest level automatically
            def get_max_depth(d, level=0):
                if not isinstance(d, dict):
                    return level
                return max(get_max_depth(v, level + 1) for v in d.values())

            max_depth = get_max_depth(data)
            if max_depth > 0:
                # Try with the deepest level
                all_rows = extract_metrics_recursive(
                    data, current_level=0, context_dict=None
                )

        # Create dataframe
        df = pd.DataFrame(all_rows)

        # Ensure 'metric' and 'value' columns are first
        if "metric" in df.columns and "value" in df.columns:
            # Reorder columns to put metric and value first, then context columns
            context_cols = [col for col in df.columns if col not in ["metric", "value"]]
            df = df[["metric", "value"] + context_cols]

        return df
