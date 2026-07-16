import inspect
import json
import os
import pickle
import re
import signal
import sys
import shutil
import threading
import ctypes
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from synthyverse.generators import DataProcessor, TabularSchema, get_generator
from synthyverse.utils.reproducibility import set_seed
from synthyverse.utils.utils import free_up_memory


from synthyverse.evaluation.ml import HYPERPARAM_SAVE_DIR

RESULT_COLUMNS = ["metric name", "metric value", "train_seed", "set"]
SCHEMA_FILENAME = "schema.pkl"
BYTES_PER_MIB = 1024 * 1024
_PROCESS_MEMORY_READER = None


class TabularSynthesisBenchmark:
    """Benchmark a tabular synthetic data generator on a real dataset.

    ``TabularSynthesisBenchmark`` manages the complete workflow for comparing a
    generator against real tabular data. It creates reproducible train and test
    splits, preprocesses the data, trains or loads the
    configured generator, samples synthetic datasets, and evaluates metrics.

    The benchmark can be run in two ways:

    * call ``run()`` to train and evaluate in one step;
    * call ``train()`` and ``eval()`` separately to reuse saved models or run
      different metrics later.

    When ``model_save_dir`` is provided, the benchmark stores the fitted data
    processor and trained generator there. When ``model_save_dir`` is ``None``,
    models and processors are not saved. Results are returned as a long-format
    ``pandas.DataFrame`` and can also be written to CSV.

    Args:
        X (pd.DataFrame): Real tabular dataset to benchmark on.
        generator (str): Name of the generator to benchmark. The name is
            resolved with ``get_generator``.
        generator_params (dict): Keyword arguments used to initialize the
            generator.
        categorical_features (list): Names of categorical/discrete columns in
            the original dataset.
        target_column (str or None): Column used for supervised metrics and,
            when categorical, stratified splits. Use ``None`` when there is no
            target column.
        model_save_dir (str or Path or None): Directory used for saved models
            and processors. Models are saved under ``model_save_dir/models`` and
            preprocessing artifacts are saved under ``model_save_dir/processors``.
            When ``None``, models and processors are not saved. Default: None.
        random_state (int): First seed used for train splits and synthetic set
            sampling. Additional train seeds are consecutive integers starting
            from this value. Default: 42.
        constraints (list or str, optional): Optional data constraints passed to
            ``DataProcessor``. Default: None.
        missing_imputation_method (str): Missing-value strategy passed to
            ``DataProcessor``. Options include ``"drop"``, ``"keep"``,
            ``"mean"``, ``"median"``, ``"most_frequent"``, and
            ``"missforest"``. Default: ``"drop"``.
        monitor_memory (bool): Whether to record peak CPU and, when available,
            CUDA memory usage during training and sampling. Default: False.
        reuse_schema (bool): Whether to reuse a saved dataset schema from
            ``model_save_dir`` when available. Default: True.
        reuse_processors (bool): Whether to reuse saved preprocessing artifacts
            for each train seed when available. Default: True.
        max_eval_samples (int or None): Default maximum number of rows per real
            and synthetic dataset passed to evaluation metrics. Real train and
            test datasets are subsampled per synthetic set with
            the same varying seed used for sampling synthetic data. Synthetic
            train and test datasets are generated directly at the capped size.
            Default: None.
        dataset_save_dir (str or Path or None): Directory for saving sampled
            synthetic datasets as parquet files. When provided, datasets are
            saved under ``dataset_save_dir/train_seed/sampling_seed``.
            Default: None.
        cap_train_time (float or None): Maximum seconds allowed for each
            generator fit. Default: None.
        hyperparam_save_dir (str or Path): Directory used by evaluation metrics
            that cache tuned hyperparameters. Default: ``HYPERPARAM_SAVE_DIR``.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.benchmark.synthesis import TabularSynthesisBenchmark
        >>>
        >>> X = pd.read_csv("data/cohort.csv")
        >>> benchmark = TabularSynthesisBenchmark(
        ...     X=X,
        ...     generator="ctgan",
        ...     generator_params={"epochs": 300},
        ...     categorical_features=["sex", "mortality"],
        ...     target_column="mortality",
        ...     model_save_dir="dataset/ctgan",
        ...     random_state=42,
        ... )
        >>>
        >>> benchmark.train(n_train_seeds=3)
        >>> results = benchmark.eval(metrics=["marginals", "dcr"], n_train_seeds=3)

        Train now and evaluate later by creating a new benchmark with the same
        dataset, generator settings, and ``model_save_dir``:

        >>> benchmark = TabularSynthesisBenchmark(
        ...     X=X,
        ...     generator="ctgan",
        ...     generator_params={"epochs": 300},
        ...     categorical_features=["sex", "mortality"],
        ...     target_column="mortality",
        ...     model_save_dir="dataset/ctgan",
        ...     random_state=42,
        ... )
        >>> benchmark.train(n_train_seeds=3)
        >>>
        >>> reloaded_benchmark = TabularSynthesisBenchmark(
        ...     X=X,
        ...     generator="ctgan",
        ...     generator_params={"epochs": 300},
        ...     categorical_features=["sex", "mortality"],
        ...     target_column="mortality",
        ...     model_save_dir="dataset/ctgan",
        ...     random_state=42,
        ... )
        >>> results = reloaded_benchmark.eval(
        ...     metrics=["marginals", "dcr"],
        ...     n_train_seeds=3,
        ... )
    """

    def __init__(
        self,
        X: pd.DataFrame,
        generator: str,
        generator_params: Dict[str, Any],
        categorical_features: list[str],
        target_column: Optional[str],
        model_save_dir: Optional[Union[str, Path]] = None,
        random_state: int = 42,
        constraints: Optional[Union[list[str], str]] = None,
        missing_imputation_method: str = "drop",
        monitor_memory: bool = False,
        reuse_schema: bool = True,
        reuse_processors: bool = True,
        max_eval_samples: Optional[int] = None,
        dataset_save_dir: Optional[Union[str, Path]] = None,
        cap_train_time: Optional[float] = None,
        hyperparam_save_dir: Union[str, Path] = HYPERPARAM_SAVE_DIR,
    ):
        self.X = X.copy()
        self.generator = generator
        self.generator_lookup_name = re.split(r"[\W_]+", generator, maxsplit=1)[0]
        self.generator_params = dict(generator_params or {})
        self.model_save_dir = None if model_save_dir is None else Path(model_save_dir)
        self.random_state = random_state
        self.categorical_features = list(categorical_features)
        self.target_column = target_column
        self.constraints = self._normalize_constraints(constraints)
        self.missing_imputation_method = missing_imputation_method
        self.monitor_memory = monitor_memory
        self.reuse_schema = reuse_schema
        self.reuse_processors = reuse_processors
        self.max_eval_samples = self._validate_max_eval_samples(max_eval_samples)
        self.dataset_save_dir = (
            None if dataset_save_dir is None else Path(dataset_save_dir)
        )
        self.cap_train_time = self._validate_time_cap(cap_train_time, "cap_train_time")
        self.hyperparam_save_dir = Path(hyperparam_save_dir)
        self._schema: Optional[TabularSchema] = None

    def train(
        self,
        n_train_seeds: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        full_determinism: bool = False,
        results_save_path: Union[str, Path] = "results",
        write_results: bool = True,
        append_results: bool = False,
    ) -> pd.DataFrame:
        """Train generators for one or more reproducible data splits.

        For each train seed, this method creates the train/test split, fits or
        loads the corresponding ``DataProcessor``, trains the
        configured generator, and saves the trained generator under
        ``model_save_dir/models`` when ``model_save_dir`` is set.

        Use this method when you want to train models now and evaluate them
        later with ``eval()``. Use the same ``test_size`` during evaluation so
        that the saved artifacts match the recreated split.

        Args:
            n_train_seeds (int): Number of consecutive train split seeds to
                train. Default: 1.
            test_size (float): Fraction of data reserved for testing.
                Default: 0.2.
            val_size (float): Fraction of training data passed to generators
                that split validation data internally. Default: 0.2.
            full_determinism (bool): Whether to request stricter deterministic
                behavior from ``set_seed``. Default: False.
            results_save_path (str or Path): Directory or CSV path for training
                results. If a directory is provided, results are written to
                ``<results_save_path>/<generator>.csv``. Default: ``"results"``.
            write_results (bool): Whether to write training results to CSV.
                Default: True.
            append_results (bool): Whether to append to an existing result CSV.
                Default: False.

        Returns:
            pd.DataFrame: Training result rows with columns ``metric name``,
            ``metric value``, ``train_seed``, and ``set``.

        Example:
            >>> train_results = benchmark.train(n_train_seeds=3)
        """
        self._validate_split_config(test_size, val_size)
        results_path = self._resolve_results_path(results_save_path)
        result_rows = self._load_result_rows(results_path) if append_results else []
        new_rows = []

        for train_seed in self._train_seeds(self.random_state, n_train_seeds):
            split = self._make_split(
                train_seed=train_seed,
                test_size=test_size,
                full_determinism=full_determinism,
            )
            processor, processed = self._load_or_fit_processor(train_seed, split)

            generator_cls = get_generator(self.generator_lookup_name)
            try:
                generator, training_time, memory_monitor = self._fit_generator(
                    generator_cls=generator_cls,
                    train_seed=train_seed,
                    processor=processor,
                    X_train=processed["train_model"],
                    val_size=val_size,
                    full_determinism=full_determinism,
                )
            except BenchmarkTimeoutError as error:
                rows = training_timeout_rows(train_seed, error.elapsed)
                result_rows.extend(rows)
                new_rows.extend(rows)
                if write_results:
                    self._save_results(result_rows, results_path)
                free_up_memory()
                break
            if self.model_save_dir is not None:
                generator.save(self._model_dir(train_seed))

            rows = [
                {
                    "metric name": "training_time_seconds",
                    "metric value": training_time,
                    "train_seed": train_seed,
                    "set": pd.NA,
                }
            ]
            rows.extend(memory_metric_rows("training", memory_monitor, train_seed))
            result_rows.extend(rows)
            new_rows.extend(rows)
            if write_results:
                self._save_results(result_rows, results_path)

            free_up_memory()

        return pd.DataFrame(new_rows, columns=RESULT_COLUMNS)

    def eval(
        self,
        metrics: Union[dict, list],
        n_train_seeds: int = 1,
        n_sets: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        full_determinism: bool = False,
        results_save_path: Union[str, Path] = "results",
        append_results: bool = True,
        max_eval_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Evaluate saved generators against real train and test data.

        For each train seed, this method recreates the data split, loads the
        saved ``DataProcessor`` and generator, samples one or more synthetic
        train/test dataset pairs, postprocesses the samples back to the original
        data representation, and evaluates the requested metrics.

        Call ``train()`` before using this method unless compatible saved models
        already exist in ``model_save_dir``. The ``test_size`` value should
        match the value used for training.

        Args:
            metrics (dict or list): Metrics passed to
                ``TabularMetricEvaluator``.
            n_train_seeds (int): Number of consecutive train split seeds to
                evaluate. Matching saved models must exist for these seeds.
                Default: 1.
            n_sets (int): Number of synthetic datasets to sample per saved
                generator. Default: 1.
            test_size (float): Fraction of data reserved for testing.
                Default: 0.2.
            val_size (float): Fraction of metric training data reserved for
                validation internally when metrics need it. Default: 0.2.
            full_determinism (bool): Whether to request stricter deterministic
                behavior from ``set_seed``. Default: False.
            results_save_path (str or Path): Directory or CSV path for
                evaluation results. If a directory is provided, results are
                written to ``<results_save_path>/<generator>.csv``.
                Default: ``"results"``.
            append_results (bool): Whether to append evaluation rows to an
                existing result CSV. Default: True.
            max_eval_samples (int or None): Optional per-call override for the
                benchmark's default evaluation sample cap. Default: None.

        Returns:
            pd.DataFrame: Evaluation result rows with sampling time, optional
            memory usage, and metric values.

        Example:
            >>> results = benchmark.eval(
            ...     metrics=["marginals", "dcr"],
            ...     n_train_seeds=3,
            ...     n_sets=2,
            ... )
        """
        if self.model_save_dir is None:
            raise ValueError("model_save_dir must be set to load saved generators.")
        self._validate_split_config(test_size, val_size)
        self._validate_metrics(metrics)
        max_eval_samples = self._effective_max_eval_samples(max_eval_samples)
        results_path = self._resolve_results_path(results_save_path)
        result_rows = self._load_result_rows(results_path) if append_results else []
        new_rows = []

        generator_cls = get_generator(self.generator_lookup_name)
        for train_seed in self._train_seeds(self.random_state, n_train_seeds):
            split = self._make_split(
                train_seed=train_seed,
                test_size=test_size,
                full_determinism=full_determinism,
            )
            processor, processed = self._load_or_fit_processor(train_seed, split)
            generator = self._load_generator(generator_cls, train_seed)

            rows = self._evaluate_generator(
                generator=generator,
                processor=processor,
                processed=processed,
                metrics=metrics,
                n_sets=n_sets,
                train_seed=train_seed,
                full_determinism=full_determinism,
                max_eval_samples=max_eval_samples,
                val_size=val_size,
            )
            result_rows.extend(rows)
            new_rows.extend(rows)
            self._save_results(result_rows, results_path)
            free_up_memory()

        return pd.DataFrame(new_rows, columns=RESULT_COLUMNS)

    def eval_saved_datasets(
        self,
        metrics: Union[dict, list],
        n_train_seeds: int = 1,
        n_sets: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        full_determinism: bool = False,
        results_save_path: Union[str, Path] = "results",
        append_results: bool = True,
    ) -> pd.DataFrame:
        """Evaluate synthetic datasets saved by ``eval()`` or ``run()``.

        This method recreates the real train/test split and loads synthetic
        train/test parquet files from ``dataset_save_dir/train_seed/sampling_seed``.
        Use it to add metrics later without loading or sampling generators.
        """
        if self.dataset_save_dir is None:
            raise ValueError("dataset_save_dir must be set to evaluate saved datasets.")
        self._validate_split_config(test_size, val_size)
        self._validate_metrics(metrics)
        results_path = self._resolve_results_path(results_save_path)
        result_rows = self._load_result_rows(results_path) if append_results else []
        new_rows = []

        for train_seed in self._train_seeds(self.random_state, n_train_seeds):
            split = self._make_split(
                train_seed=train_seed,
                test_size=test_size,
                full_determinism=full_determinism,
            )
            _, processed = self._load_or_fit_processor(train_seed, split)

            for set_index in range(n_sets):
                sampling_seed = self.random_state + set_index
                X_syn, X_syn_test = self._load_synthetic_datasets(
                    train_seed,
                    sampling_seed,
                )
                X_train_eval = self._subsample_eval_dataset_to_size(
                    processed["train_eval"],
                    sampling_seed,
                    len(X_syn),
                )
                X_test_eval = self._subsample_eval_dataset_to_size(
                    processed["test_eval"],
                    sampling_seed,
                    len(X_syn_test),
                )
                rows = self._evaluate_synthetic_datasets(
                    X_train_eval=X_train_eval,
                    X_test_eval=X_test_eval,
                    X_syn=X_syn,
                    X_syn_test=X_syn_test,
                    metrics=metrics,
                    train_seed=train_seed,
                    set_index=set_index,
                    sampling_seed=sampling_seed,
                    val_size=val_size,
                )
                result_rows.extend(rows)
                new_rows.extend(rows)
                self._save_results(result_rows, results_path)
                free_up_memory()

            shutil.rmtree(self.hyperparam_save_dir, ignore_errors=True)

        return pd.DataFrame(new_rows, columns=RESULT_COLUMNS)

    def run(
        self,
        metrics: Union[dict, list],
        n_train_seeds: int = 1,
        n_sets: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        full_determinism: bool = False,
        results_save_path: Union[str, Path] = "results",
        max_eval_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Train generators and evaluate them in one benchmark run.

        This convenience method trains each generator and evaluates it directly,
        without loading the model back from disk.

        Args:
            metrics (dict or list): Metrics passed to
                ``TabularMetricEvaluator``.
            n_train_seeds (int): Number of consecutive train split seeds to
                train and evaluate. Default: 1.
            n_sets (int): Number of synthetic datasets to sample per saved
                generator. Default: 1.
            test_size (float): Fraction of data reserved for testing.
                Default: 0.2.
            val_size (float): Fraction of training data reserved internally by
                generators or metrics that need validation data. Default: 0.2.
            full_determinism (bool): Whether to request stricter deterministic
                behavior from ``set_seed``. Default: False.
            results_save_path (str or Path): Directory or CSV path for all
                benchmark results. If a directory is provided, results are
                written to ``<results_save_path>/<generator>.csv``.
                Default: ``"results"``.
            max_eval_samples (int or None): Optional per-call override for the
                benchmark's default evaluation sample cap. Default: None.
        Returns:
            pd.DataFrame: Combined training and evaluation result rows.

        Example:
            >>> metrics = ["marginals", "dcr"]
            >>> results = benchmark.run(metrics=metrics, n_train_seeds=3, n_sets=2)
        """
        self._validate_split_config(test_size, val_size)
        self._validate_metrics(metrics)
        max_eval_samples = self._effective_max_eval_samples(max_eval_samples)
        results_path = self._resolve_results_path(results_save_path)
        result_rows = []
        new_rows = []

        generator_cls = get_generator(self.generator_lookup_name)
        for train_seed in self._train_seeds(self.random_state, n_train_seeds):
            split = self._make_split(
                train_seed=train_seed,
                test_size=test_size,
                full_determinism=full_determinism,
            )
            processor, processed = self._load_or_fit_processor(
                train_seed,
                split,
            )
            try:
                generator, training_time, memory_monitor = self._fit_generator(
                    generator_cls=generator_cls,
                    train_seed=train_seed,
                    processor=processor,
                    X_train=processed["train_model"],
                    val_size=val_size,
                    full_determinism=full_determinism,
                )
            except BenchmarkTimeoutError as error:
                rows = training_timeout_rows(train_seed, error.elapsed)
                result_rows.extend(rows)
                new_rows.extend(rows)
                self._save_results(result_rows, results_path)
                free_up_memory()
                break
            if self.model_save_dir is not None:
                generator.save(self._model_dir(train_seed))

            rows = [
                {
                    "metric name": "training_time_seconds",
                    "metric value": training_time,
                    "train_seed": train_seed,
                    "set": pd.NA,
                }
            ]
            rows.extend(memory_metric_rows("training", memory_monitor, train_seed))
            result_rows.extend(rows)
            new_rows.extend(rows)
            self._save_results(result_rows, results_path)

            rows = self._evaluate_generator(
                generator=generator,
                processor=processor,
                processed=processed,
                metrics=metrics,
                n_sets=n_sets,
                train_seed=train_seed,
                full_determinism=full_determinism,
                max_eval_samples=max_eval_samples,
                val_size=val_size,
            )
            result_rows.extend(rows)
            new_rows.extend(rows)
            self._save_results(result_rows, results_path)
            free_up_memory()

        return pd.DataFrame(new_rows, columns=RESULT_COLUMNS)

    def _make_split(
        self,
        train_seed: int,
        test_size: float,
        full_determinism: bool,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        set_seed(train_seed, full_determinism)
        stratify = self._stratify_values(self.X)
        X_train, X_test = train_test_split(
            self.X,
            test_size=test_size,
            random_state=train_seed,
            stratify=stratify,
        )

        return {"train": X_train, "test": X_test}

    def _load_or_fit_processor(
        self,
        train_seed: int,
        split: Dict[str, Optional[pd.DataFrame]],
        persist: bool = True,
    ) -> tuple[DataProcessor, Dict[str, Optional[pd.DataFrame]]]:
        persist = persist and self.model_save_dir is not None
        schema = self._load_or_create_schema(persist=persist)
        processor_dir = self._processor_dir(train_seed) if persist else None
        processor_file = processor_dir / "processor.pkl" if persist else None

        if persist and self.reuse_processors and processor_file.exists():
            processor = DataProcessor.load(processor_dir)
            self._validate_cached_processor(processor, train_seed)
            processed = processor.preprocess(X=split["train"])
            print(f"Loaded DataProcessor from {processor_file}")
        else:
            processor = DataProcessor(
                constraints=self.constraints,
                missing_imputation_method=self.missing_imputation_method,
                random_state=train_seed,
            )
            processed = processor.preprocess(
                X=split["train"],
                discrete_features=self.categorical_features,
            )
            self._apply_schema(processor, schema)
            if persist:
                processor.save(processor_dir)
                print(f"Saved DataProcessor to {processor_file}")

        self._apply_schema(processor, schema)

        X_train_model = processed
        X_test_model = processor.preprocess(X=split["test"])

        return processor, {
            "train_model": X_train_model,
            "test_model": X_test_model,
            "train_eval": processor.postprocess(X_train_model),
            "test_eval": processor.postprocess(X_test_model),
        }

    def _fit_generator(
        self,
        generator_cls,
        train_seed: int,
        processor: DataProcessor,
        X_train: pd.DataFrame,
        val_size: float,
        full_determinism: bool,
    ):
        set_seed(train_seed, full_determinism)
        params = dict(self.generator_params)
        signature = inspect.signature(generator_cls.__init__)
        signature_params = signature.parameters

        if "target_column" in signature_params and "target_column" not in params:
            params["target_column"] = self.target_column
        if (
            self._accepts_kwarg(signature, "random_state")
            and "random_state" not in params
        ):
            params["random_state"] = train_seed
        generator_handles_train_time_cap = self._accepts_kwarg(
            signature,
            "cap_train_time",
        )
        if generator_handles_train_time_cap and "cap_train_time" not in params:
            params["cap_train_time"] = self.cap_train_time

        generator = generator_cls(**params)
        fit_kwargs = {"X": X_train, "discrete_features": processor.categorical_features}
        fit_signature = inspect.signature(generator.fit)
        if self._accepts_kwarg(fit_signature, "val_size"):
            fit_kwargs["val_size"] = val_size
        time_limit = None if generator_handles_train_time_cap else self.cap_train_time
        train_start_time = perf_counter()
        memory_monitor = None
        try:
            if self.monitor_memory:
                with TimeLimit(time_limit):
                    with PeakMemoryMonitor() as memory_monitor:
                        generator.fit(**fit_kwargs)
            else:
                with TimeLimit(time_limit):
                    generator.fit(**fit_kwargs)
        except BenchmarkTimeoutError as error:
            error.elapsed = perf_counter() - train_start_time
            raise
        training_time = perf_counter() - train_start_time
        return generator, training_time, memory_monitor

    def _load_generator(self, generator_cls, train_seed: int):
        if self.model_save_dir is None:
            raise ValueError("model_save_dir must be set to load saved generators.")
        model_dir = self._model_dir(train_seed)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No saved generator found for train_seed={train_seed} at {model_dir}. "
                "Run train() first, or check model_save_dir/generator/random_state."
            )
        return generator_cls.load(model_dir)

    def _evaluate_generator(
        self,
        generator,
        processor: DataProcessor,
        processed: Dict[str, Optional[pd.DataFrame]],
        metrics: Union[dict, list],
        n_sets: int,
        train_seed: int,
        full_determinism: bool,
        max_eval_samples: Optional[int],
        val_size: float,
    ) -> list[dict[str, Any]]:
        result_rows = []
        for set_index in range(n_sets):
            sampling_seed = self.random_state + set_index
            set_seed(sampling_seed, full_determinism)
            X_train_eval = self._subsample_eval_dataset(
                processed["train_eval"], sampling_seed, max_eval_samples
            )
            X_test_eval = self._subsample_eval_dataset(
                processed["test_eval"], sampling_seed, max_eval_samples
            )
            n_train_syn = len(X_train_eval)
            n_test_syn = len(X_test_eval)

            memory_monitor = None
            sampling_start_time = perf_counter()
            if self.monitor_memory:
                with PeakMemoryMonitor() as memory_monitor:
                    X_syn_all = generator.generate(n_train_syn + n_test_syn)
                    X_syn_all = processor.postprocess(X_syn_all)
            else:
                X_syn_all = generator.generate(n_train_syn + n_test_syn)
                X_syn_all = processor.postprocess(X_syn_all)

            X_syn = X_syn_all.iloc[:n_train_syn].reset_index(drop=True)
            X_syn_test = X_syn_all.iloc[n_train_syn:].reset_index(drop=True)
            sampling_time = perf_counter() - sampling_start_time
            self._save_synthetic_datasets(
                train_seed,
                sampling_seed,
                X_syn,
                X_syn_test,
            )

            result_rows.append(
                {
                    "metric name": "sampling_time_seconds",
                    "metric value": sampling_time,
                    "train_seed": train_seed,
                    "set": set_index,
                }
            )
            result_rows.extend(
                memory_metric_rows("sampling", memory_monitor, train_seed, set_index)
            )
            result_rows.extend(
                self._evaluate_synthetic_datasets(
                    X_train_eval=X_train_eval,
                    X_test_eval=X_test_eval,
                    X_syn=X_syn,
                    X_syn_test=X_syn_test,
                    metrics=metrics,
                    train_seed=train_seed,
                    set_index=set_index,
                    sampling_seed=sampling_seed,
                    val_size=val_size,
                )
            )

            free_up_memory()
        # remove tuned hyperparams before next training seed
        shutil.rmtree(self.hyperparam_save_dir, ignore_errors=True)

        return result_rows

    def _evaluate_synthetic_datasets(
        self,
        X_train_eval: pd.DataFrame,
        X_test_eval: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_syn_test: pd.DataFrame,
        metrics: Union[dict, list],
        train_seed: int,
        set_index: int,
        sampling_seed: int,
        val_size: float,
    ) -> list[dict[str, Any]]:
        from synthyverse.evaluation.eval import TabularMetricEvaluator

        eval_categorical_features = [
            col for col in self.categorical_features if col in X_train_eval.columns
        ]
        evaluator = TabularMetricEvaluator(
            metrics=metrics,
            discrete_features=eval_categorical_features,
            target_column=self.target_column,
            random_state=sampling_seed,
            val_size=val_size,
            hyperparam_save_dir=self.hyperparam_save_dir,
        )
        evaluation_start_time = perf_counter()
        metric_results = evaluator.evaluate(
            X_train_eval,
            X_test_eval,
            X_syn,
            X_syn_test,
        )
        evaluation_time = perf_counter() - evaluation_start_time

        result_rows = [
            {
                "metric name": "evaluation_time_seconds",
                "metric value": evaluation_time,
                "train_seed": train_seed,
                "set": set_index,
            }
        ]
        for metric_name, metric_value in flatten_metrics(metric_results):
            result_rows.append(
                {
                    "metric name": metric_name,
                    "metric value": metric_value,
                    "train_seed": train_seed,
                    "set": set_index,
                }
            )

        print(metric_results)
        return result_rows

    def _load_or_create_schema(self, persist: bool = True) -> TabularSchema:
        if self._schema is not None:
            return self._schema

        if persist:
            schema_path = self._schema_path()
            if self.reuse_schema and schema_path.exists():
                with schema_path.open("rb") as f:
                    self._schema = pickle.load(f)
                return self._schema

        numerical_features = [
            col for col in self.X.columns if col not in self.categorical_features
        ]
        self._schema = TabularSchema.from_dataframe(self.X, numerical_features)
        if persist:
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            with schema_path.open("wb") as f:
                pickle.dump(self._schema, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self._schema

    def _apply_schema(self, processor: DataProcessor, schema: TabularSchema) -> None:
        processor.schema = schema
        processor.ori_col_order = schema.column_order
        processor.ori_dtypes = schema.dtypes
        processor.ori_precision = schema.precision

    def _validate_cached_processor(
        self,
        processor: DataProcessor,
        train_seed: int,
    ) -> None:
        if processor.missing_imputation_method != self.missing_imputation_method:
            raise ValueError(
                "Cached DataProcessor has a different missing_imputation_method. "
                "Use a different model_save_dir or remove the cached processor."
            )
        if list(processor.constraints) != self.constraints:
            raise ValueError(
                "Cached DataProcessor has different constraints. "
                "Use a different model_save_dir or remove the cached processor."
            )
        if processor.random_state != train_seed:
            raise ValueError(
                "Cached DataProcessor random_state does not match the train split seed."
            )

    @staticmethod
    def _validate_metrics(metrics: Union[dict, list]) -> None:
        if metrics is None:
            raise ValueError("metrics must be provided as a list or dictionary.")
        if not isinstance(metrics, (dict, list)):
            raise TypeError("metrics must be a list or dictionary.")

    def _load_result_rows(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict("records")

    def _save_results(self, result_rows: list[dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result_rows, columns=RESULT_COLUMNS).to_csv(path, index=False)

    def _resolve_results_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.suffix:
            return path
        return path / f"{self.generator}.csv"

    def _schema_path(self) -> Path:
        return self.model_save_dir / "processors" / SCHEMA_FILENAME

    def _processor_dir(self, train_seed: int) -> Path:
        return self.model_save_dir / "processors" / str(train_seed)

    def _model_dir(self, train_seed: int) -> Path:
        return (
            self.model_save_dir / "models" / self.generator / f"train_seed_{train_seed}"
        )

    def _dataset_dir(self, train_seed: int, sampling_seed: int) -> Path:
        return self.dataset_save_dir / str(train_seed) / str(sampling_seed)

    def _stratify_values(self, X: pd.DataFrame):
        if self.target_column is None:
            return None
        if self.target_column not in self.categorical_features:
            return None
        if self.target_column not in X.columns:
            return None
        return X[self.target_column]

    def _subsample_eval_dataset(
        self,
        X: Optional[pd.DataFrame],
        random_state: int,
        max_eval_samples: Optional[int],
    ) -> Optional[pd.DataFrame]:
        if X is None:
            return None
        if max_eval_samples is None or len(X) <= max_eval_samples:
            return X
        return X.sample(n=max_eval_samples, random_state=random_state)

    def _subsample_eval_dataset_to_size(
        self,
        X: pd.DataFrame,
        random_state: int,
        size: int,
    ) -> pd.DataFrame:
        if len(X) < size:
            raise ValueError("Saved synthetic dataset has more rows than real data.")
        if len(X) == size:
            return X
        return X.sample(n=size, random_state=random_state)

    def _save_synthetic_datasets(
        self,
        train_seed: int,
        sampling_seed: int,
        X_syn: pd.DataFrame,
        X_syn_test: pd.DataFrame,
    ) -> None:
        if self.dataset_save_dir is None:
            return
        dataset_dir = self._dataset_dir(train_seed, sampling_seed)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        X_syn.to_parquet(dataset_dir / "synthetic_train.parquet", index=False)
        X_syn_test.to_parquet(dataset_dir / "synthetic_test.parquet", index=False)

    def _load_synthetic_datasets(
        self,
        train_seed: int,
        sampling_seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataset_dir = self._dataset_dir(train_seed, sampling_seed)
        train_path = dataset_dir / "synthetic_train.parquet"
        test_path = dataset_dir / "synthetic_test.parquet"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"No saved synthetic datasets found for train_seed={train_seed}, "
                f"sampling_seed={sampling_seed} at {dataset_dir}."
            )
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    @staticmethod
    def _train_seeds(random_state: int, n_train_seeds: int) -> range:
        return range(random_state, random_state + n_train_seeds)

    @staticmethod
    def _validate_split_config(test_size: float, val_size: float) -> None:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 <= val_size < 1:
            raise ValueError("val_size must be non-negative and less than 1.")

    @staticmethod
    def _validate_max_eval_samples(max_eval_samples: Optional[int]) -> Optional[int]:
        if max_eval_samples is None:
            return None
        if not isinstance(max_eval_samples, int) or isinstance(max_eval_samples, bool):
            raise TypeError("max_eval_samples must be an integer or None.")
        if max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be greater than 0.")
        return max_eval_samples

    def _effective_max_eval_samples(
        self,
        max_eval_samples: Optional[int],
    ) -> Optional[int]:
        if max_eval_samples is None:
            return self.max_eval_samples
        return self._validate_max_eval_samples(max_eval_samples)

    @staticmethod
    def _validate_time_cap(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number or None.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0 or None.")
        return float(value)

    @staticmethod
    def _normalize_constraints(
        constraints: Optional[Union[list[str], str]],
    ) -> list[str]:
        if constraints is None:
            return []
        if isinstance(constraints, str):
            return [constraints]
        return list(constraints)

    @staticmethod
    def _accepts_kwarg(signature: inspect.Signature, name: str) -> bool:
        if name in signature.parameters:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )


def make_csv_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(
            {str(key): make_csv_safe(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple, set)):
        return json.dumps([make_csv_safe(item) for item in value])
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient="records")
    if isinstance(value, pd.Series):
        return make_csv_safe(value.to_list())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return make_csv_safe(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return make_csv_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return value


def flatten_metrics(metrics: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            metric_name = f"{prefix}.{name}" if prefix else str(name)
            yield from flatten_metrics(value, metric_name)
    else:
        yield prefix, make_csv_safe(metrics)


class BenchmarkTimeoutError(TimeoutError):
    def __init__(self):
        super().__init__("benchmark step exceeded its time cap")
        self.elapsed: Optional[float] = None


class TimeLimit:
    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self._timer = None
        self._old_handler = None
        self._use_signal = False
        self._thread_id = None

    def __enter__(self):
        if self.seconds is None:
            return self
        if self.seconds <= 0:
            raise BenchmarkTimeoutError()
        if (
            hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        ):
            self._use_signal = True
            self._old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        else:
            self._thread_id = threading.get_ident()
            self._timer = threading.Timer(self.seconds, self._raise_in_thread)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old_handler)
        elif self._timer is not None:
            self._timer.cancel()

    def _raise_timeout(self, signum, frame):
        raise BenchmarkTimeoutError()

    def _raise_in_thread(self):
        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = (
            ctypes.c_ulong,
            ctypes.py_object,
        )
        ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_int
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(self._thread_id),
            ctypes.py_object(BenchmarkTimeoutError),
        )
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._thread_id),
                ctypes.py_object(None),
            )


def training_timeout_rows(
    train_seed: int,
    elapsed: Optional[float],
) -> list[dict[str, Any]]:
    return [
        {
            "metric name": "training_time_seconds",
            "metric value": elapsed,
            "train_seed": train_seed,
            "set": pd.NA,
        },
        {
            "metric name": "training_timed_out",
            "metric value": True,
            "train_seed": train_seed,
            "set": pd.NA,
        },
    ]


class PeakMemoryMonitor:
    """Sample process memory while a benchmark step is running."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_memory_mb: Optional[float] = None
        self.peak_cuda_memory_mb: Optional[float] = None
        self._peak_rss_bytes: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "PeakMemoryMonitor":
        self._sample_once()
        _reset_cuda_peak_memory()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._sample_once()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample_once()
        self.peak_memory_mb = _bytes_to_mib(self._peak_rss_bytes)
        self.peak_cuda_memory_mb = _bytes_to_mib(_cuda_peak_memory_bytes())

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        rss_bytes = _current_process_memory_bytes()
        if rss_bytes is None:
            return
        if self._peak_rss_bytes is None or rss_bytes > self._peak_rss_bytes:
            self._peak_rss_bytes = rss_bytes


def _bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / BYTES_PER_MIB


def _current_process_memory_bytes() -> Optional[int]:
    global _PROCESS_MEMORY_READER
    if _PROCESS_MEMORY_READER is None:
        _PROCESS_MEMORY_READER = _select_process_memory_reader()
    return _PROCESS_MEMORY_READER()


def _select_process_memory_reader():
    psutil_reader = _make_psutil_memory_reader()
    if psutil_reader is not None:
        return psutil_reader
    if os.name == "nt":
        return _windows_process_memory_bytes
    if sys.platform.startswith("linux"):
        return _linux_process_memory_bytes
    return _resource_process_memory_bytes


def _make_psutil_memory_reader():
    try:
        import psutil
    except ImportError:
        return None

    try:
        process = psutil.Process(os.getpid())
    except Exception:
        return None

    def read_memory_bytes() -> Optional[int]:
        try:
            return int(process.memory_info().rss)
        except Exception:
            return None

    return read_memory_bytes


def _windows_process_memory_bytes() -> Optional[int]:
    try:
        import ctypes
    except ImportError:
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_int
    success = get_memory_info(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        return None
    return int(counters.WorkingSetSize)


def _linux_process_memory_bytes() -> Optional[int]:
    try:
        with Path("/proc/self/statm").open() as statm_file:
            resident_pages = int(statm_file.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _resource_process_memory_bytes() -> Optional[int]:
    try:
        import resource
    except ImportError:
        return None

    try:
        max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None

    if sys.platform == "darwin":
        return max_rss
    return max_rss * 1024


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _cuda_peak_memory_bytes() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def memory_metric_rows(
    prefix: str,
    monitor: Optional[PeakMemoryMonitor],
    train_seed: int,
    set_index: Any = pd.NA,
) -> list[dict[str, Any]]:
    rows = []
    if monitor is None:
        return rows
    if monitor.peak_memory_mb is not None:
        rows.append(
            {
                "metric name": f"{prefix}_peak_memory_mb",
                "metric value": monitor.peak_memory_mb,
                "train_seed": train_seed,
                "set": set_index,
            }
        )
    if monitor.peak_cuda_memory_mb is not None:
        rows.append(
            {
                "metric name": f"{prefix}_peak_cuda_memory_mb",
                "metric value": monitor.peak_cuda_memory_mb,
                "train_seed": train_seed,
                "set": set_index,
            }
        )
    return rows
