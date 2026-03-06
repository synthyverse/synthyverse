from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import os
import shutil
import copy
from uuid import uuid4
from typing import Any, Dict, List, Tuple, Union

from .utils import format_results
from ..evaluation.eval import TabularMetricEvaluator
from ..generators import get_generator
from ..generators.base import TabularBaseGenerator
from ..utils.utils import free_up_memory
from ..utils.reproducibility import set_seed

DEFAULT_BENCHMARK_METRICS = ("classifier_test", "mle", "dcr")


class TabularSynthesisBenchmark:
    """Benchmark for evaluating tabular synthetic data generators.

    Args:
        generator (Union[str, TabularBaseGenerator]): Generator identifier. Can be a synthyverse generator name or a custom generator instance.
        generator_params (dict): Dictionary of generator-specific parameters. Default: None (empty dict).
        n_random_splits (int): Number of random train/test splits to evaluate. Default: 1.
        n_inits (int): Number of generator training initializations per split. Default: 1.
        test_size (float): Proportion of data to use for testing (0.0 to 1.0). Default: 0.2.
        val_size (float): Proportion of data to use for validation (0.0 to 1.0). Set to 0.0 to disable the validation split. Note that val_size+test_size must be < 1.0. Default: 0.1.
        missing_imputation_method (str): Method for handling missing values. "drop" removes missing rows, other options perform imputation: "random", "mean", "median", "most_frequent", "missforest". Default: "drop".
        retain_missingness (bool): Whether to retain missing values in generated datasets. Default: False.
        constraints (Union[str, list]): List of constraint strings which should hold in the generated data. Note that the constraints should already hold in the training datasets. Default: None (empty list).
        workspace (str): Directory for storing intermediate files. Default: "workspace".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.benchmark import TabularSynthesisBenchmark
        >>>
        >>> # Load your data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_columns = ["category_col"]
        >>> target_column = "target"
        >>>
        >>> # Create benchmark
        >>> benchmark = TabularSynthesisBenchmark(
        ...     generator="arf",
        ...     generator_params={"num_trees": 50},
        ...     n_random_splits=3,
        ...     n_inits=3
        ... )
        >>>
        >>> # Train and evaluate models
        >>> trained_models = benchmark.train(X, target_column, discrete_columns)
        >>> results = benchmark.eval(
        ...     X,
        ...     trained_models,
        ...     metrics=["classifier_test", "mle", "dcr"],
        ...     n_generated_datasets=1,
        ... )

        >>> # Or, train and evaluate models in one step:
        >>> results, trained_models = benchmark.train_and_eval(X, target_column, discrete_columns)
        >>> results
    """

    def __init__(
        self,
        generator: Union[str, TabularBaseGenerator] = "arf",
        generator_params: dict = None,
        n_random_splits: int = 1,
        n_inits: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.1,
        missing_imputation_method: str = "drop",
        retain_missingness: bool = False,
        constraints: Union[str, list] = None,
        workspace: str = "workspace",
        random_state: int = 0,
    ):
        if generator_params is None:
            generator_params = {}
        if constraints is None:
            constraints = []

        self._custom_generator_template = None
        if isinstance(generator, str):
            self.generator = generator
        elif isinstance(generator, TabularBaseGenerator):
            self._custom_generator_template = generator
            custom_name = getattr(generator, "name", generator.__class__.__name__)
            self.generator = str(custom_name).replace(" ", "_")
        else:
            raise TypeError(
                "generator must be either a generator name (str) or an instance of TabularBaseGenerator."
            )

        self.generator_params = dict(generator_params)
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
                "constraints": constraints,
            }
        )

        self.n_random_splits = n_random_splits
        self.n_inits = n_inits
        self.test_size = test_size
        self.val_size = val_size
        self.missing_imputation_method = missing_imputation_method
        self.retain_missingness = retain_missingness
        self.constraints = constraints
        self._trained_target_column = None
        self._trained_discrete_columns = None
        self.workspace = workspace
        self.random_state = random_state
        self._benchmark_instance_id = uuid4().hex

    def _get_generator_setup(self, target_column: str):
        if self._custom_generator_template is not None:
            generator_ = self._custom_generator_template.__class__
            return generator_, {}

        generator_ = get_generator(self.generator)
        generator_params = dict(self.generator_params)

        # add workspace if needed
        if getattr(generator_, "needs_workspace", False):
            generator_params["workspace"] = self.workspace
        # add target column if needed
        if getattr(generator_, "needs_target_column", False):
            generator_params["target_column"] = target_column

        # do not impute missing values if the generator natively handles missingness
        if (
            getattr(generator_, "handles_missingness", False)
            and generator_params["missing_imputation_method"] != "drop"
        ):
            generator_params["missing_imputation_method"] = "keep"

        return generator_, generator_params

    def _create_generator_instance(
        self,
        generator_: type,
        generator_params: dict,
        init_i: int,
        target_column: str,
        workspace: Union[str, None] = None,
    ):
        if self._custom_generator_template is None:
            return generator_(random_state=init_i, **generator_params)

        try:
            generator = copy.deepcopy(self._custom_generator_template)
        except Exception as exc:
            raise TypeError(
                "Custom generator instance must be deepcopy-able for repeated benchmark training."
            ) from exc

        # Harmonize benchmark-level preprocessing and seed settings on the copied instance.
        if hasattr(generator, "random_state"):
            generator.random_state = init_i
        if hasattr(generator, "missing_imputation_method"):
            generator.missing_imputation_method = self.missing_imputation_method
        if hasattr(generator, "retain_missingness"):
            generator.retain_missingness = self.retain_missingness
        if hasattr(generator, "constraints"):
            generator.constraints = copy.deepcopy(self.constraints)

        if hasattr(generator_, "needs_workspace") and getattr(
            generator_, "needs_workspace", False
        ):
            if hasattr(generator, "workspace"):
                generator.workspace = self.workspace if workspace is None else workspace
        if hasattr(generator_, "needs_target_column") and getattr(
            generator_, "needs_target_column", False
        ):
            if hasattr(generator, "target_column"):
                generator.target_column = target_column

        return generator

    def _split_data(
        self,
        X: pd.DataFrame,
        target_column: str,
        discrete_columns: list,
        split_i: int,
        test_size: Union[float, None] = None,
        val_size: Union[float, None] = None,
    ):
        if test_size is None:
            test_size = self.test_size
        if val_size is None:
            val_size = self.val_size

        # split data according to current seed
        stratify = None
        if target_column in discrete_columns:
            stratify = X[target_column]
        X_train, X_test = train_test_split(
            X, stratify=stratify, test_size=test_size, random_state=split_i
        )
        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
        if val_size > 0:
            stratify = None
            if target_column in discrete_columns:
                stratify = X_train[target_column]
            X_train, X_val = train_test_split(
                X_train,
                stratify=stratify,
                test_size=val_size
                / (1 - test_size),  # val_size is a proportion of the training set
                random_state=split_i,
            )
            X_train, X_val = X_train.reset_index(drop=True), X_val.reset_index(
                drop=True
            )
        else:
            X_val = None

        return X_train, X_test, X_val

    def _get_model_workspace_path(
        self, split_i: int, init_i: int, generator: Union[str, None] = None
    ):
        if generator is None:
            generator = self.generator
        return os.path.join(
            self.workspace,
            f"{generator}_split_{split_i}_init_{init_i}_workspace",
        )

    def _attach_training_context_to_model(
        self,
        model: TabularBaseGenerator,
        split_i: int,
        init_i: int,
        target_column: str,
        discrete_columns: list,
        training_time: float,
    ) -> None:
        model._benchmark_split_random_state = split_i
        model._benchmark_init_random_state = init_i
        model._benchmark_target_column = target_column
        model._benchmark_discrete_columns = list(discrete_columns)
        model._benchmark_test_size = self.test_size
        model._benchmark_val_size = self.val_size
        model._benchmark_base_random_state = self.random_state
        model._benchmark_training_time = training_time
        model._benchmark_instance_id = self._benchmark_instance_id

    @staticmethod
    def _extract_seed_from_key(prefix: str, key: Any) -> Union[int, None]:
        if not isinstance(key, str):
            return None
        expected_prefix = f"{prefix}_"
        if not key.startswith(expected_prefix):
            return None
        seed_str = key[len(expected_prefix) :]
        if seed_str.startswith("-"):
            digit_seed = seed_str[1:]
        else:
            digit_seed = seed_str
        if not digit_seed.isdigit():
            return None
        return int(seed_str)

    def _collect_model_entries(
        self,
        trained_models: Union[TabularBaseGenerator, Dict[str, Any]],
        split_seed: Union[int, None] = None,
        init_seed: Union[int, None] = None,
    ) -> List[Dict[str, Any]]:
        entries = []
        if isinstance(trained_models, TabularBaseGenerator):
            entries.append(
                {
                    "model": trained_models,
                    "split_seed": split_seed,
                    "init_seed": init_seed,
                }
            )
            return entries

        if isinstance(trained_models, dict):
            for key, value in trained_models.items():
                key_split_seed = self._extract_seed_from_key("split", key)
                key_init_seed = self._extract_seed_from_key("init", key)
                next_split_seed = (
                    split_seed if key_split_seed is None else key_split_seed
                )
                next_init_seed = init_seed if key_init_seed is None else key_init_seed
                entries.extend(
                    self._collect_model_entries(
                        trained_models=value,
                        split_seed=next_split_seed,
                        init_seed=next_init_seed,
                    )
                )
            return entries

        raise TypeError(
            "trained_models must be a trained model or a nested dict of trained models returned by this benchmark's train()."
        )

    def _validate_model_provenance(self, model: TabularBaseGenerator) -> None:
        required_attributes = [
            "_benchmark_split_random_state",
            "_benchmark_init_random_state",
            "_benchmark_target_column",
            "_benchmark_discrete_columns",
            "_benchmark_test_size",
            "_benchmark_val_size",
            "_benchmark_base_random_state",
            "_benchmark_instance_id",
        ]
        missing_attributes = [
            attribute_name
            for attribute_name in required_attributes
            if getattr(model, attribute_name, None) is None
        ]
        if missing_attributes:
            missing = ", ".join(missing_attributes)
            raise ValueError(
                f"Provided model is missing benchmark training metadata ({missing}). "
                "Pass models returned by this benchmark's train()."
            )

        model_instance_id = str(getattr(model, "_benchmark_instance_id"))
        if model_instance_id != self._benchmark_instance_id:
            raise ValueError(
                "Provided model was not trained by this benchmark instance. "
                "Pass models returned by this benchmark's train()."
            )

    def _normalize_trained_models(
        self, trained_models: Union[TabularBaseGenerator, Dict[str, Any]]
    ) -> Dict[int, Dict[int, TabularBaseGenerator]]:
        model_entries = self._collect_model_entries(trained_models=trained_models)
        if len(model_entries) == 0:
            raise ValueError("No trained models were provided to eval().")

        normalized_models: Dict[int, Dict[int, TabularBaseGenerator]] = {}
        for model_entry in model_entries:
            model = model_entry["model"]
            self._validate_model_provenance(model)
            split_from_dict = model_entry["split_seed"]
            init_from_dict = model_entry["init_seed"]
            split_from_model = getattr(model, "_benchmark_split_random_state")
            init_from_model = getattr(model, "_benchmark_init_random_state")

            if (
                split_from_model is not None
                and split_from_dict is not None
                and int(split_from_model) != int(split_from_dict)
            ):
                raise ValueError(
                    "Inconsistent split seeds detected between model metadata and dict keys."
                )
            if (
                init_from_model is not None
                and init_from_dict is not None
                and int(init_from_model) != int(init_from_dict)
            ):
                raise ValueError(
                    "Inconsistent init seeds detected between model metadata and dict keys."
                )

            split_i = split_from_model
            init_i = init_from_model

            split_i = int(split_i)
            init_i = int(init_i)
            if split_i not in normalized_models:
                normalized_models[split_i] = {}
            if init_i in normalized_models[split_i]:
                raise ValueError(
                    f"Duplicate model detected for split={split_i}, init={init_i}."
                )
            normalized_models[split_i][init_i] = model

        return normalized_models

    @staticmethod
    def _resolve_uniform_model_attribute(
        models: list, attribute_name: str, default_value: Any = None
    ) -> Any:
        resolved_value = default_value
        has_resolved_from_model = False
        for model in models:
            model_value = getattr(model, attribute_name, None)
            if model_value is None:
                continue
            if not has_resolved_from_model:
                resolved_value = model_value
                has_resolved_from_model = True
                continue
            if model_value != resolved_value:
                raise ValueError(
                    f"Inconsistent '{attribute_name}' found across provided models."
                )
        return resolved_value

    def train(
        self,
        X: pd.DataFrame,
        target_column: str,
        discrete_columns: list,
    ):
        """Train the configured generator and return trained model objects.

        Args:
            X: Full dataset as a pandas DataFrame.
            target_column: Name of the target column.
            discrete_columns: List of discrete/categorical column names.

        Returns:
            TabularBaseGenerator or dict: Trained model, or nested `split/init` dict of trained models.
        """
        os.makedirs(self.workspace, exist_ok=True)
        self._trained_target_column = target_column
        self._trained_discrete_columns = list(discrete_columns)
        trained_models = {}
        generator_, generator_params = self._get_generator_setup(target_column)
        needs_workspace = getattr(generator_, "needs_workspace", False)

        for split_i in range(
            self.random_state, self.random_state + self.n_random_splits
        ):
            # remove any previously tuned hyperparameters; they need to be re-tuned for different training splits
            shutil.rmtree("synthyverse_hyperparams_tuned", ignore_errors=True)
            split_key = f"split_{split_i}"
            trained_models[split_key] = {}
            X_train, _X_test, X_val = self._split_data(
                X, target_column, discrete_columns, split_i
            )

            for init_i in range(self.random_state, self.random_state + self.n_inits):
                set_seed(init_i)

                iteration_workspace = self.workspace
                iteration_generator_params = dict(generator_params)
                if needs_workspace:
                    iteration_workspace = self._get_model_workspace_path(
                        split_i, init_i
                    )
                    shutil.rmtree(iteration_workspace, ignore_errors=True)
                    os.makedirs(iteration_workspace, exist_ok=True)
                    iteration_generator_params["workspace"] = iteration_workspace
                else:
                    self.clean_directory(self.workspace, remove_self=False)

                generator = self._create_generator_instance(
                    generator_=generator_,
                    generator_params=iteration_generator_params,
                    init_i=init_i,
                    target_column=target_column,
                    workspace=iteration_workspace,
                )
                start_time = time()
                generator.fit(
                    X=X_train, discrete_features=discrete_columns, X_val=X_val
                )
                training_time = time() - start_time

                self._attach_training_context_to_model(
                    model=generator,
                    split_i=split_i,
                    init_i=init_i,
                    target_column=target_column,
                    discrete_columns=discrete_columns,
                    training_time=training_time,
                )
                trained_models[split_key][f"init_{init_i}"] = generator

                # release memory for next iteration
                free_up_memory()

        if not needs_workspace:
            self.clean_directory(self.workspace, remove_self=True)
        shutil.rmtree("synthyverse_hyperparams_tuned", ignore_errors=True)
        if len(trained_models) == 1:
            split_models = next(iter(trained_models.values()))
            if len(split_models) == 1:
                return next(iter(split_models.values()))
        return trained_models

    def eval(
        self,
        X: pd.DataFrame,
        trained_models: Union[TabularBaseGenerator, dict],
        metrics: Union[list, dict, None] = None,
        n_generated_datasets: int = 1,
        max_eval_size: int = int(1e9),
        result_format: str = "frame",  # "frame" or "dict"
    ):
        """Evaluate trained model objects.

        Args:
            X: Full dataset as a pandas DataFrame.
            trained_models: A single trained model or nested `split/init` dict returned by this benchmark's `train()`.
            metrics: List or dictionary of metrics to evaluate. Defaults to
                ["classifier_test", "mle", "dcr"] when None.
            n_generated_datasets: Number of synthetic datasets to generate per initialization.
            max_eval_size: Maximum size of sampled train/test/validation subsets used for evaluation.
            result_format: Format of results ("frame" for DataFrame, "dict" for nested dict).

        Returns:
            pd.DataFrame or dict: Benchmark results in the specified format.
        """
        if metrics is None:
            metrics = list(DEFAULT_BENCHMARK_METRICS)

        os.makedirs(self.workspace, exist_ok=True)

        normalized_models = self._normalize_trained_models(
            trained_models=trained_models
        )
        all_models = [
            model
            for split_models in normalized_models.values()
            for model in split_models.values()
        ]

        model_target_column = self._resolve_uniform_model_attribute(
            all_models, "_benchmark_target_column", default_value=None
        )
        model_discrete_columns = self._resolve_uniform_model_attribute(
            all_models, "_benchmark_discrete_columns", default_value=None
        )
        if model_discrete_columns is not None:
            model_discrete_columns = list(model_discrete_columns)

        resolved_target_column = model_target_column
        resolved_discrete_columns = model_discrete_columns
        if resolved_target_column is None or resolved_discrete_columns is None:
            raise ValueError(
                "Could not resolve target/discrete columns from model metadata. Pass models returned by this benchmark's train()."
            )
        resolved_discrete_columns = list(resolved_discrete_columns)

        resolved_test_size = self._resolve_uniform_model_attribute(
            all_models, "_benchmark_test_size", default_value=None
        )
        resolved_val_size = self._resolve_uniform_model_attribute(
            all_models, "_benchmark_val_size", default_value=None
        )
        resolved_random_state = self._resolve_uniform_model_attribute(
            all_models, "_benchmark_base_random_state", default_value=None
        )
        if (
            resolved_test_size is None
            or resolved_val_size is None
            or resolved_random_state is None
        ):
            raise ValueError(
                "Could not resolve test/validation split configuration from model metadata. "
                "Pass models returned by this benchmark's train()."
            )
        resolved_random_state = int(resolved_random_state)

        results = {}
        for split_i in sorted(normalized_models.keys()):
            split_key = f"split_{split_i}"
            results[split_key] = {}
            X_train, X_test, X_val = self._split_data(
                X,
                resolved_target_column,
                resolved_discrete_columns,
                split_i,
                test_size=resolved_test_size,
                val_size=resolved_val_size,
            )

            for init_i in sorted(normalized_models[split_i].keys()):
                init_key = f"init_{init_i}"
                results[split_key][init_key] = {}
                generator = normalized_models[split_i][init_i]

                # potentially generate multiple datasets
                for generated_dataset_i in range(
                    resolved_random_state,
                    resolved_random_state + n_generated_datasets,
                ):
                    set_seed(generated_dataset_i)
                    generated_dataset_key = f"generated_dataset_{generated_dataset_i}"
                    results[split_key][init_key][generated_dataset_key] = {}

                    # sample synthetic dataset and perform evaluation
                    n_train = min(max_eval_size, len(X_train))
                    n_test = min(max_eval_size, len(X_test))
                    n = n_train + n_test
                    start_time = time()
                    X_syn = generator.generate(n)
                    results[split_key][init_key][generated_dataset_key][
                        "inference_time"
                    ] = (time() - start_time)
                    evaluator = TabularMetricEvaluator(
                        metrics=metrics,
                        discrete_features=resolved_discrete_columns,
                        target_column=resolved_target_column,
                        missing_imputation_method=self.missing_imputation_method,
                        random_state=generated_dataset_i,
                    )
                    X_val_sample = None
                    if X_val is not None:
                        n_val = min(max_eval_size, len(X_val))
                        X_val_sample = X_val.sample(
                            n_val, replace=False, random_state=generated_dataset_i
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
                        X_val_sample,
                    )
                    results[split_key][init_key][generated_dataset_key][
                        "evaluation_time"
                    ] = (time() - start_time)
                    results[split_key][init_key][generated_dataset_key].update(
                        metric_results
                    )

                    # release memory for next iteration
                    free_up_memory()
        if result_format == "frame":
            results = format_results(results)
        return results

    def train_and_eval(
        self,
        X: pd.DataFrame,
        target_column: str,
        discrete_columns: list,
        metrics: Union[list, dict, None] = None,
        n_generated_datasets: int = 1,
        max_eval_size: int = int(1e9),
        result_format: str = "frame",  # "frame" or "dict"
    ) -> Tuple[Union[pd.DataFrame, dict], Union[TabularBaseGenerator, dict]]:
        """Train and evaluate the generator.

        This is a convenience wrapper for users who only need benchmark results
        and do not need to call `train()` and `eval()` separately.

        Args:
            X: Full dataset as a pandas DataFrame.
            target_column: Name of the target column.
            discrete_columns: List of discrete/categorical column names.
            metrics: List or dictionary of metrics to evaluate. Defaults to
                ["classifier_test", "mle", "dcr"] when None.
            n_generated_datasets: Number of synthetic datasets to generate per initialization.
            max_eval_size: Maximum size of sampled train/test/validation subsets used for evaluation.
            result_format: Format of results ("frame" for DataFrame, "dict" for nested dict).

        Returns:
            tuple: `(results, trained_models)`, where `results` is in the requested
            `result_format`, and `trained_models` is the output from `train()`.
        """
        trained_models = self.train(
            X=X,
            target_column=target_column,
            discrete_columns=discrete_columns,
        )
        results = self.eval(
            X=X,
            trained_models=trained_models,
            metrics=metrics,
            n_generated_datasets=n_generated_datasets,
            max_eval_size=max_eval_size,
            result_format=result_format,
        )
        return results, trained_models

    def clean_directory(self, path: str, remove_self: bool = False) -> None:

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
