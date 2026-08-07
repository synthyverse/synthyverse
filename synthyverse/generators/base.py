import pickle
import importlib
import inspect
import re
from copy import deepcopy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ..utils.reproducibility import set_seed


PROCESSOR_FILENAME = "processor.pkl"
WRAPPER_FILENAME = "synthyverse_generator.pkl"
WRAPPED_GENERATOR_DIR = "generator"
WRAPPED_PROCESSOR_DIR = "processor"


class BaseGenerator(ABC):
    """Base interface for synthyverse tabular generators.

    Subclasses implement :meth:`_fit` and :meth:`_generate`; this class owns
    the shared public API, docstrings, and fluent ``fit`` return value.
    """

    supports_purely_numerical = True
    supports_purely_categorical = True

    def __init__(self, random_state: int = 0, full_determinism: bool = False):
        """Initialize shared generator reproducibility settings.

        Args:
            random_state: Training seed used by :meth:`fit`. Default: 0.
            full_determinism: Whether to request deterministic backend behavior
                when setting the training seed. Default: False.
        """
        self.random_state = random_state
        self.full_determinism = full_determinism
        self.ordinal_encoder = None
        self._base_discrete_features = []
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

    def _validate_feature_mix(self, X: pd.DataFrame, discrete_features: list) -> None:
        missing_features = [col for col in discrete_features if col not in X.columns]
        if missing_features:
            missing = ", ".join(missing_features)
            raise ValueError(f"discrete_features are not present in X: {missing}")

        non_numerical_features = [
            col
            for col in X.columns
            if col not in discrete_features
            and not pd.api.types.is_numeric_dtype(X[col])
        ]
        if non_numerical_features:
            non_numerical = ", ".join(non_numerical_features)
            raise ValueError(
                f"Non-discrete features must be numerical: {non_numerical}"
            )

        n_discrete_features = sum(col in discrete_features for col in X.columns)
        if n_discrete_features == 0 and not self.supports_purely_numerical:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one categorical feature."
            )
        if (
            n_discrete_features == len(X.columns)
            and not self.supports_purely_categorical
        ):
            raise ValueError(
                f"{self.__class__.__name__} requires at least one numerical feature."
            )

    def fit(
        self,
        X: pd.DataFrame,
        discrete_features: list = [],
    ):
        """Fit the generator to tabular data.

        Args:
            X: Training data in the generator's input space.
            discrete_features: Names of categorical/discrete columns in ``X``.
                Default: [].

        Returns:
            The fitted generator.
        """
        discrete_features = list(discrete_features)
        self._validate_feature_mix(X, discrete_features)
        self.schema = TabularSchema.from_dataframe(X)
        X = self._fit_encode_categoricals(X, discrete_features)
        set_seed(self.random_state, self.full_determinism)
        self._generate_model_space = True
        try:
            self._fit(X, discrete_features)
        finally:
            self._generate_model_space = False
        return self

    def _fit_encode_categoricals(
        self, X: pd.DataFrame, discrete_features: list
    ) -> pd.DataFrame:
        self._base_discrete_features = [
            col for col in X.columns if col in discrete_features
        ]
        self.ordinal_encoder = None
        if not self._base_discrete_features:
            return X.copy()

        missing_value = np.iinfo("int64").max - 2
        x = X.copy()
        categorical = x[self._base_discrete_features].astype(object)
        categorical = categorical.mask(categorical.isna(), np.nan)
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.iinfo("int64").max - 3,
            encoded_missing_value=missing_value,
            dtype=np.int64,
        )
        encoded = self.ordinal_encoder.fit_transform(categorical)
        for i, categories in enumerate(self.ordinal_encoder.categories_):
            missing = np.flatnonzero(pd.isna(categories))
            if len(missing):
                encoded[:, i] = np.where(
                    encoded[:, i] == missing_value, missing[0], encoded[:, i]
                )
        x[self._base_discrete_features] = encoded.astype(np.int64)
        return x

    def _decode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X.copy()
        if self.ordinal_encoder is not None and len(x) > 0:
            x[self._base_discrete_features] = self.ordinal_encoder.inverse_transform(
                x[self._base_discrete_features].round().astype(np.int64)
            )
        return x

    def _categorical_cardinality(self, col: str) -> int:
        idx = self._base_discrete_features.index(col)
        return len(self.ordinal_encoder.categories_[idx])

    def _categorical_cardinalities(self, cols: list = None) -> list:
        if cols is None:
            cols = self._base_discrete_features
        return [self._categorical_cardinality(col) for col in cols]

    @abstractmethod
    def _fit(
        self,
        X: pd.DataFrame,
        discrete_features: list,
    ):
        """Generator-specific fitting implementation."""

    def generate(self, n: int, random_state: int = 0):
        """Generate synthetic tabular data.

        Args:
            n: Number of synthetic rows to generate.
            random_state: Sampling seed. Default: 0.

        Returns:
            Synthetic data restored to the schema passed to :meth:`fit`.
        """
        if not hasattr(self, "schema"):
            raise ValueError("Cannot generate before the generator is fitted.")
        fit_random_state = self.random_state
        set_seed(random_state, self.full_determinism)
        self.random_state = random_state
        try:
            syn = self._generate(n)
        finally:
            self.random_state = fit_random_state
        if getattr(self, "_generate_model_space", False):
            return syn
        return self.schema.restore(self._decode_categoricals(syn))

    @abstractmethod
    def _generate(self, n: int):
        """Generator-specific sampling implementation."""

    def _state(self) -> dict:
        return self.__dict__.copy()

    def _base_state(self) -> dict:
        return {
            "random_state": self.random_state,
            "full_determinism": self.full_determinism,
            "ordinal_encoder": self.ordinal_encoder,
            "_base_discrete_features": self._base_discrete_features,
        }

    def _save_extra(self, path: Path) -> None:
        pass

    @classmethod
    def _restore_state(cls, state: dict):
        from .persistence import restore_generator

        return restore_generator(cls, state)

    def _load_extra(self, path: Path) -> None:
        pass

    def save(self, path):
        """Persist the generator state with the default pickle layout."""
        from .persistence import save_generator_state

        state = self._state()
        state.update(self._base_state())
        if hasattr(self, "schema"):
            state["schema"] = self.schema
        path = save_generator_state(path, state)
        self._save_extra(path)
        return path

    @classmethod
    def load(cls, path):
        """Load a generator persisted with the default pickle layout."""
        from .persistence import load_generator_state

        path = Path(path)
        generator = cls._restore_state(load_generator_state(path))
        generator._load_extra(path)
        return generator


class TabularSchema:
    """Column order, dtype, and precision contract for tabular data.

    Captures the schema of a real pandas DataFrame so generated or transformed
    data can be restored to the same column order, dtypes, and numeric
    precision. This is usually managed by :class:`DataProcessor`, but it can be
    useful directly when you need to restore model-space data yourself.

    Args:
        column_order (list): Column names in the desired output order.
        dtypes (dict): Mapping from column names to pandas dtypes.
        precision (dict): Mapping from numeric column names to decimal places.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabularSchema
        >>>
        >>> X = pd.DataFrame({"age": [31, 42], "score": [0.25, 0.75]})
        >>> schema = TabularSchema.from_dataframe(X)
        >>> restored = schema.restore(X[["score", "age"]])
        >>> list(restored.columns)
        ['age', 'score']
    """

    def __init__(self, column_order, dtypes, precision):
        self.column_order = list(column_order)
        self.dtypes = dtypes.copy()
        self.precision = dict(precision)

    @classmethod
    def from_dataframe(cls, X: pd.DataFrame, numerical_features: list = None):
        """Create a schema contract from a DataFrame.

        Args:
            X (pd.DataFrame): Data whose columns, dtypes, and numeric precision
                should be captured.
            numerical_features (list): Optional names of numeric columns to
                inspect for decimal precision. When omitted, numeric columns
                are inferred from pandas dtypes.

        Returns:
            TabularSchema: Schema object fitted to ``X``.
        """
        if numerical_features is None:
            numerical_features = [
                col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
            ]
        precision = {
            col: column_precision(X[col]) for col in numerical_features if col in X
        }
        return cls(X.columns, X.dtypes, precision)

    def validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that a DataFrame contains every column in the schema.

        Args:
            X (pd.DataFrame): Data to validate.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing_columns = [col for col in self.column_order if col not in X.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Input data is missing expected columns: {missing}")

    def round_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Round numeric columns to the precision captured from real data.

        Args:
            X (pd.DataFrame): Data to round.

        Returns:
            pd.DataFrame: Copy of ``X`` with known numeric columns rounded.
        """
        x = X.copy()
        for col, precision in self.precision.items():
            if col in x.columns and pd.api.types.is_numeric_dtype(x[col]):
                x[col] = x[col].round(precision)
        return x

    def restore(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restore column order, numeric precision, and pandas dtypes.

        Args:
            X (pd.DataFrame): Data containing the columns captured by the
                schema.

        Returns:
            pd.DataFrame: Restored data with the original schema.
        """
        x = self.round_numeric(X)
        self.validate_columns(x)
        x = x[self.column_order]
        return x.astype(self.dtypes)


class TabularImputer:
    """Reusable missing-value transformer for tabular generator inputs.

    Handles missing numerical values before data is passed to a low-level
    generator. It supports dropping rows, keeping missing values, simple
    imputation, and a MissForest-style iterative imputer. Most users get this
    through :class:`DataProcessor`; use it directly when you only need missing
    value handling without constraints or schema restoration.

    Args:
        method (str): Missing-value strategy. Options are ``"drop"``,
            ``"keep"``, ``"mean"``, ``"median"``, ``"most_frequent"``, and
            ``"missforest"``. Default: ``"drop"``.
        random_state (int): Random seed used by stochastic imputers. Default:
            0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabularImputer
        >>>
        >>> X = pd.DataFrame({"age": [31, None, 42], "group": ["a", "b", "a"]})
        >>> imputer = TabularImputer(method="median", random_state=42)
        >>> X_imputed, _ = imputer.fit_transform(X, numerical_features=["age"])
        >>> X_later = imputer.transform(X)
    """

    def __init__(self, method: str = "drop", random_state: int = 0):
        self.method = method
        self.random_state = random_state
        self.imputer = None
        self.imputer_base_cols = None
        self.categorical_features = []
        self.ordinal_encoder = None
        self.numerical_features = []

    def _has_missing_numerical(self, X: pd.DataFrame) -> bool:
        if X is None or len(self.numerical_features) == 0:
            return False
        return bool(X[self.numerical_features].isna().any().any())

    def _validate_method(self) -> None:
        valid_methods = {
            "drop",
            "keep",
            "mean",
            "median",
            "most_frequent",
            "missforest",
        }
        if self.method not in valid_methods:
            raise ValueError(f"Invalid missing imputation method: {self.method}")

    def fit_transform(
        self,
        X: pd.DataFrame,
        numerical_features: list,
        X_val: pd.DataFrame = None,
    ):
        """Fit the imputer on training data and transform train/validation data.

        Args:
            X (pd.DataFrame): Training data.
            numerical_features (list): Numeric columns to impute for simple
                strategies and to inspect for ``"drop"``.
            X_val (pd.DataFrame): Optional validation data in the same schema
                as ``X``. Default: None.

        Returns:
            tuple: ``(X_processed, X_val_processed)``. The second item is
            ``None`` when no validation data is provided.
        """
        self.numerical_features = list(numerical_features)
        x = X.copy()
        x_val = X_val.copy() if X_val is not None else None
        self._validate_method()

        if self.method == "keep":
            return x, x_val

        if not self._has_missing_numerical(x) and not self._has_missing_numerical(
            x_val
        ):
            return x, x_val

        if self.method == "drop":
            x = x.dropna(subset=self.numerical_features)
            if x_val is not None:
                x_val = x_val.dropna(subset=self.numerical_features)
        elif self.method in ["mean", "median", "most_frequent"]:
            self.imputer = SimpleImputer(strategy=self.method)
            if len(self.numerical_features) > 0:
                x[self.numerical_features] = self.imputer.fit_transform(
                    x[self.numerical_features]
                )
                if x_val is not None:
                    x_val[self.numerical_features] = self.imputer.transform(
                        x_val[self.numerical_features]
                    )
        elif self.method == "missforest":
            estimator = RandomForestRegressor(
                n_estimators=20,
                max_depth=10,
                random_state=self.random_state,
            )
            self.imputer = IterativeImputer(
                estimator=estimator,
                random_state=self.random_state,
                tol=1e-3,
                max_iter=10,
            )
            self.imputer_base_cols = x.columns.tolist()
            self.categorical_features = [
                col
                for col in self.imputer_base_cols
                if col not in self.numerical_features
            ]
            if self.categorical_features:
                self.ordinal_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                )
                x[self.categorical_features] = self.ordinal_encoder.fit_transform(
                    x[self.categorical_features]
                )
            x[self.imputer_base_cols] = self.imputer.fit_transform(
                x[self.imputer_base_cols]
            )
            if self.categorical_features:
                x[self.categorical_features] = self.ordinal_encoder.inverse_transform(
                    x[self.categorical_features]
                )
            if x_val is not None:
                if self.categorical_features:
                    x_val[self.categorical_features] = self.ordinal_encoder.transform(
                        x_val[self.categorical_features]
                    )
                x_val[self.imputer_base_cols] = self.imputer.transform(
                    x_val[self.imputer_base_cols]
                )
                if self.categorical_features:
                    x_val[self.categorical_features] = (
                        self.ordinal_encoder.inverse_transform(
                            x_val[self.categorical_features]
                        )
                    )

        return x, x_val

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with the fitted missing-value strategy.

        Args:
            X (pd.DataFrame): Data in the same schema used during fitting.

        Returns:
            pd.DataFrame: Data with missing values handled according to
            ``method``.
        """
        x = X.copy()
        self._validate_method()

        if self.method == "drop":
            if self._has_missing_numerical(x):
                return x.dropna(subset=self.numerical_features)
            return x
        if self.method == "keep":
            return x
        if not self._has_missing_numerical(x):
            return x
        if self.method in ["mean", "median", "most_frequent"]:
            if len(self.numerical_features) > 0:
                if self.imputer is None:
                    raise RuntimeError(
                        "Cannot impute missing numerical values because fitting "
                        "skipped imputation when no missing numerical values were present."
                    )
                x[self.numerical_features] = self.imputer.transform(
                    x[self.numerical_features]
                )
            return x
        if self.method == "missforest":
            if self.imputer is None:
                raise RuntimeError(
                    "Cannot impute missing numerical values because fitting "
                    "skipped imputation when no missing numerical values were present."
                )
            if self.categorical_features:
                x[self.categorical_features] = self.ordinal_encoder.transform(
                    x[self.categorical_features]
                )
            x[self.imputer_base_cols] = self.imputer.transform(
                x[self.imputer_base_cols]
            )
            if self.categorical_features:
                x[self.categorical_features] = self.ordinal_encoder.inverse_transform(
                    x[self.categorical_features]
                )
            return x


class DataProcessor:
    """Reusable tabular pre/postprocessor for synthyverse generators.

    The first call to :meth:`preprocess` fits the processor state. Later calls
    reuse the fitted imputers, constraints, precision, dtypes, and column order,
    allowing multiple generators to share one processor for the same dataset.

    Use ``preprocess`` before fitting a low-level generator, then use
    ``postprocess`` on generated model-space data to restore the original
    schema. For single-generator workflows, :class:`SynthyverseGenerator`
    provides the same behavior as a wrapper.

    Args:
        constraints (list, str): Optional equality or inequality constraints
            to enforce in model space and restore after generation. Examples:
            ``"total=part_a+part_b"``, ``"age>=18"``, or
            ``"income>expenses"``. Default: None.
        missing_imputation_method (str): Missing-value strategy. Options are
            ``"drop"``, ``"keep"``, ``"mean"``, ``"median"``,
            ``"most_frequent"``, and ``"missforest"``. Default: ``"drop"``.
        random_state (int): Random seed used by stochastic preprocessing steps.
            Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import DataProcessor, UnivariateGenerator
        >>>
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> processor = DataProcessor(
        ...     constraints=["total=part_a+part_b"],
        ...     missing_imputation_method="median",
        ...     random_state=42,
        ... )
        >>> X_processed = processor.preprocess(X, discrete_features)
        >>>
        >>> generator = UnivariateGenerator(random_state=42)
        >>> generator.fit(X_processed, discrete_features)
        >>> X_syn = processor.postprocess(generator.generate(1000))
    """

    def __init__(
        self,
        constraints: Union[list, str, None] = None,
        missing_imputation_method: str = "drop",
        random_state: int = 0,
    ):
        if constraints is None:
            constraints = []
        if isinstance(constraints, str):
            constraints = [constraints]

        self.constraints = constraints
        self.missing_imputation_method = missing_imputation_method
        self.random_state = random_state
        self.imputer = TabularImputer(missing_imputation_method, random_state)
        self.constant_values = {}
        self.fitted = False

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "constant_values"):
            self.constant_values = {}
        if self.fitted and not hasattr(self, "schema"):
            self.schema = TabularSchema(
                self.ori_col_order,
                self.ori_dtypes,
                self.ori_precision,
            )
        if not isinstance(getattr(self, "imputer", None), TabularImputer):
            legacy_imputer = getattr(self, "imputer", None)
            self.imputer = TabularImputer(
                self.missing_imputation_method,
                self.random_state,
            )
            self.imputer.numerical_features = list(
                getattr(self, "numerical_features", [])
            )
            self.imputer.imputer = legacy_imputer
            self.imputer.imputer_base_cols = getattr(self, "imputer_base_cols", None)
            if self.imputer.imputer_base_cols is not None:
                self.imputer.categorical_features = [
                    col
                    for col in self.imputer.imputer_base_cols
                    if col not in self.imputer.numerical_features
                ]
            else:
                self.imputer.categorical_features = list(
                    getattr(self, "categorical_features", [])
                )
            self.imputer.ordinal_encoder = getattr(
                self,
                "ordinal_encoder",
                getattr(self, "ord_encoder", None),
            )
        if self.fitted and not hasattr(self, "input_schema_fingerprint"):
            self.input_schema_fingerprint = self._schema_fingerprint_from_schema(
                self.schema
            )
        if self.fitted and not hasattr(self, "input_categorical_features"):
            self.input_categorical_features = list(
                getattr(self, "categorical_features", [])
            )

    def preprocess(
        self,
        X: pd.DataFrame,
        discrete_features: list = None,
    ):
        """Fit-if-needed and transform input data for model training.

        On the first call, this method records the original schema, fits the
        missing-value handler, and prepares constraint handling. Later calls
        reuse that fitted state and only transform data with the same original
        schema.

        Args:
            X (pd.DataFrame): Training or later input data in the original
                schema.
            discrete_features (list): Names of categorical/discrete columns.
                Required on the first call and optional after the processor is
                fitted. Default: None.

        Returns:
            pd.DataFrame: Processed ``X``.
        """
        if self.fitted:
            return self._transform(X)

        if discrete_features is None:
            raise ValueError(
                "discrete_features must be provided when fitting a DataProcessor."
            )
        return self._fit_transform(X, discrete_features)

    def postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform generated model-space data back to the original schema.

        Restores the processed model-space schema, applies inverse constraints,
        restores dropped constraint columns, rounds numeric columns to the
        original precision, restores the original column order, and casts
        columns back to their original pandas dtypes.

        Args:
            X (pd.DataFrame): Generated data in the processor's model-space
                schema.

        Returns:
            pd.DataFrame: Generated data in the original input schema.
        """
        if not self.fitted:
            raise ValueError("Cannot postprocess before the DataProcessor is fitted.")

        syn_X = self.model_schema.restore(X)
        syn_X = self.constraint_enforcer.inverse_transform(syn_X)
        for col, value in self.constant_values.items():
            syn_X[col] = value
        return self.schema.restore(syn_X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist this processor to disk.

        ``path`` may be either the target file path or a directory. When a
        directory is provided, the processor is written to ``processor.pkl``.
        """
        path = self._resolve_persistence_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataProcessor":
        """Load a persisted processor from disk.

        Args:
            path (str, Path): Path to a saved ``processor.pkl`` file or a
                directory containing one.

        Returns:
            DataProcessor: Restored processor.
        """
        path = cls._resolve_persistence_path(path)
        with path.open("rb") as f:
            processor = pickle.load(f)
        if not isinstance(processor, cls):
            raise TypeError(
                f"Expected a {cls.__name__} object, found {type(processor).__name__}."
            )
        return processor

    @staticmethod
    def _resolve_persistence_path(path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.suffix:
            return path
        return path / PROCESSOR_FILENAME

    def _fit_transform(
        self,
        X: pd.DataFrame,
        discrete_features: list,
    ):
        discrete_features = list(discrete_features)
        self._validate_fit_input(X)
        self._validate_feature_mix(X, discrete_features)
        self.input_schema_fingerprint = self._schema_fingerprint_from_dataframe(X)
        self.input_categorical_features = list(discrete_features)

        self.categorical_features = [x for x in X.columns if x in discrete_features]
        self.numerical_features = [
            x for x in X.columns if x not in self.categorical_features
        ]
        self.schema = TabularSchema.from_dataframe(X, self.numerical_features)
        self.ori_col_order = self.schema.column_order
        self.ori_dtypes = self.schema.dtypes
        self.ori_precision = self.schema.precision

        x = X.copy()

        self.ord_encoder = None

        self.imputer = TabularImputer(
            self.missing_imputation_method,
            self.random_state,
        )
        x, _ = self.imputer.fit_transform(
            x,
            self.numerical_features,
        )
        self.constraint_enforcer = ConstraintEnforcer(self.constraints)
        self.constraint_enforcer.validate(x)
        x = self.constraint_enforcer.transform(x)

        constraint_columns = self.constraint_enforcer.referenced_columns(x.columns)
        self.constant_values = {
            col: x[col].iloc[0]
            for col in x.columns
            if col not in constraint_columns and x[col].nunique(dropna=False) == 1
        }
        if self.constant_values:
            constant_cols = list(self.constant_values)
            x = x.drop(columns=constant_cols)
            self.categorical_features = [
                col
                for col in self.categorical_features
                if col not in self.constant_values
            ]
            self.numerical_features = [
                col
                for col in self.numerical_features
                if col not in self.constant_values
            ]
        self.categorical_features = [
            col for col in self.categorical_features if col in x.columns
        ]
        self.numerical_features = [
            col for col in self.numerical_features if col in x.columns
        ]
        self.model_schema = TabularSchema.from_dataframe(x)

        self.fitted = True

        return x

    def _transform(self, X: pd.DataFrame):
        return self._transform_one(X)

    def _transform_one(self, X: pd.DataFrame) -> pd.DataFrame:
        if X is None:
            return None
        self.schema.validate_columns(X)

        x = X.copy()
        x = x[self.ori_col_order]

        x = self.imputer.transform(x)
        if self.constant_values:
            x = x.drop(columns=list(self.constant_values))
        x = self.constraint_enforcer.transform(x)
        return x

    def _validate_fit_input(self, X: pd.DataFrame) -> None:
        if any(" " in x for x in X.columns):
            raise ValueError(
                "Feature names cannot contain spaces. Please rename the features and try again."
            )

        for col in X.columns:
            if X[col].isna().all():
                raise ValueError(
                    f"Column {col} has only missing values in the training set, which we currently do not support."
                )

    def _validate_feature_mix(self, X: pd.DataFrame, discrete_features: list) -> None:
        missing_features = [col for col in discrete_features if col not in X.columns]
        if missing_features:
            missing = ", ".join(missing_features)
            raise ValueError(f"discrete_features are not present in X: {missing}")

        non_numerical_features = [
            col
            for col in X.columns
            if col not in discrete_features
            and not pd.api.types.is_numeric_dtype(X[col])
        ]
        if non_numerical_features:
            non_numerical = ", ".join(non_numerical_features)
            raise ValueError(
                f"Non-discrete features must be numerical: {non_numerical}"
            )

    @staticmethod
    def _schema_fingerprint_from_dataframe(X: pd.DataFrame) -> list[tuple[object, str]]:
        return [(col, str(X[col].dtype)) for col in X.columns]

    @staticmethod
    def _schema_fingerprint_from_schema(schema) -> list[tuple[object, str]]:
        return [(col, str(schema.dtypes[col])) for col in schema.column_order]


class SynthyverseGenerator:
    """Synthyverse high-level generator wrapper for tabular data.

    Combines a low-level ``BaseGenerator`` with the shared ``DataProcessor`` to
    provide missing-value handling, constraint handling, dtype restoration,
    column-order restoration, and numeric precision restoration around any
    Synthyverse generator.

    The wrapped low-level generator and processor remain available through the
    ``generator`` and ``processor`` attributes for users who want explicit
    control over each step.

    Args:
        generator (str, type, BaseGenerator): Generator name, ``BaseGenerator``
            subclass, or fitted/unfitted generator instance to wrap.
        generator_params (dict): Keyword arguments used when ``generator`` is a
            name or class. Default: None.
        processor (DataProcessor): Optional preconfigured data processor. If
            provided, ``constraints`` and ``missing_imputation_method`` are
            ignored. Default: None.
        constraints (list, str): Optional equality or inequality constraints
            applied during preprocessing and reversed during postprocessing.
            Default: None.
        missing_imputation_method (str): Missing-value strategy used by the
            created ``DataProcessor``. Options: "drop", "keep", "mean",
            "median", "most_frequent", "missforest". Default: "drop".
        random_state (int): Random seed used by the created ``DataProcessor``
            and by generator classes that accept ``random_state`` when no value
            is supplied in ``generator_params`` or ``generator_kwargs``.
            Default: 0.
        full_determinism (bool): Whether the wrapped generator should request
            stricter deterministic framework behavior during fitting. Default:
            False.
        **generator_kwargs: Additional keyword arguments passed to the wrapped
            generator constructor. These override keys in ``generator_params``.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import SynthyverseGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create high-level wrapper around a low-level generator
        >>> generator = SynthyverseGenerator(
        ...     "univariate",
        ...     missing_imputation_method="median",
        ...     random_state=42,
        ... )
        >>>
        >>> # Fit and generate data in the original schema
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    def __init__(
        self,
        generator,
        generator_params: dict = None,
        processor: DataProcessor = None,
        constraints: Union[list, str, None] = None,
        missing_imputation_method: str = "drop",
        random_state: int = 0,
        full_determinism: bool = False,
        **generator_kwargs,
    ):
        """Create a high-level Synthyverse generator wrapper.

        Args:
            generator: Generator name, generator class, or fitted/unfitted
                generator instance.
            generator_params: Keyword arguments used when ``generator`` is a
                name or class.
            processor: Optional preconfigured ``DataProcessor``.
            constraints: Constraints passed to ``DataProcessor`` when no
                processor is supplied.
            missing_imputation_method: Missing-value strategy for the created
                ``DataProcessor``.
            random_state: Random seed used by the created ``DataProcessor`` and
                by generator classes that accept ``random_state`` when it is not
                already specified in ``generator_params`` or ``generator_kwargs``.
            full_determinism: Whether the wrapped generator should request
                stricter deterministic framework behavior during fitting.
            **generator_kwargs: Additional generator constructor arguments.
        """
        self.random_state = random_state
        self.full_determinism = full_determinism
        self.processor = processor or DataProcessor(
            constraints=constraints,
            missing_imputation_method=missing_imputation_method,
            random_state=random_state,
        )
        self.generator = self._make_generator(
            generator,
            generator_params=generator_params,
            generator_kwargs=generator_kwargs,
            random_state=random_state,
            full_determinism=full_determinism,
        )
        self.fitted = False

    def train(
        self,
        X: pd.DataFrame,
        discrete_features: list = None,
    ):
        """Preprocess data and fit the wrapped low-level generator.

        Args:
            X (pd.DataFrame): Training data in the original tabular schema.
            discrete_features (list): Names of categorical/discrete columns in
                ``X``. Required when fitting a new processor. Default: None.

        Returns:
            SynthyverseGenerator: The fitted high-level generator.
        """
        if discrete_features is not None:
            self.generator._validate_feature_mix(X, discrete_features)

        processor_state = deepcopy(self.processor.__dict__)
        try:
            processed = self.processor.preprocess(
                X=X,
                discrete_features=discrete_features,
            )
            X_processed = processed

            if discrete_features is None:
                discrete_features = self.processor.categorical_features
            else:
                discrete_features = [
                    col for col in discrete_features if col in X_processed.columns
                ]

            self.generator.fit(X=X_processed, discrete_features=discrete_features)
        except Exception:
            self.processor.__dict__.clear()
            self.processor.__dict__.update(processor_state)
            raise
        self.fitted = True
        return self

    def fit(
        self,
        X: pd.DataFrame,
        discrete_features: list = None,
    ):
        """Fit the high-level generator to tabular data.

        Alias for :meth:`train` for consistency with low-level generators.

        Args:
            X (pd.DataFrame): Training data in the original tabular schema.
            discrete_features (list): Names of categorical/discrete columns in
                ``X``. Required when fitting a new processor. Default: None.

        Returns:
            SynthyverseGenerator: The fitted high-level generator.
        """
        return self.train(X=X, discrete_features=discrete_features)

    def sample(self, n: int, random_state: int = 0) -> pd.DataFrame:
        """Generate synthetic rows and restore the original tabular schema.

        Args:
            n (int): Number of synthetic rows to generate.
            random_state (int): Sampling seed passed to the wrapped generator.
                Default: 0.

        Returns:
            pd.DataFrame: Synthetic data with the original columns, dtypes, and
            numeric precision restored.
        """
        if not self.processor.fitted:
            raise ValueError("Cannot sample before the generator is trained.")
        syn = self.generator.generate(n, random_state=random_state)
        return self.processor.postprocess(syn)

    def generate(self, n: int, random_state: int = 0) -> pd.DataFrame:
        """Generate synthetic tabular data.

        Alias for :meth:`sample` for consistency with low-level generators.

        Args:
            n (int): Number of synthetic rows to generate.
            random_state (int): Sampling seed passed to the wrapped generator.
                Default: 0.

        Returns:
            pd.DataFrame: Synthetic data with the original columns, dtypes, and
            numeric precision restored.
        """
        return self.sample(n, random_state=random_state)

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the wrapper, processor, and wrapped generator to a directory.

        Args:
            path (str, Path): Directory where the wrapper state should be saved.

        Returns:
            Path: Directory containing the saved wrapper state.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.processor.save(path / WRAPPED_PROCESSOR_DIR)
        self.generator.save(path / WRAPPED_GENERATOR_DIR)

        state = {
            "generator_module": self.generator.__class__.__module__,
            "generator_class": self.generator.__class__.__name__,
            "generator_name": getattr(self.generator, "name", None),
            "random_state": self.random_state,
            "full_determinism": self.full_determinism,
            "fitted": self.fitted,
        }
        with (path / WRAPPER_FILENAME).open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SynthyverseGenerator":
        """Load a high-level generator wrapper saved with :meth:`save`.

        Args:
            path (str, Path): Directory containing the saved wrapper state.

        Returns:
            SynthyverseGenerator: Restored high-level generator wrapper.
        """
        path = Path(path)
        with (path / WRAPPER_FILENAME).open("rb") as f:
            state = pickle.load(f)

        generator_cls = cls._load_generator_class(state)
        generator = generator_cls.load(path / WRAPPED_GENERATOR_DIR)
        processor = DataProcessor.load(path / WRAPPED_PROCESSOR_DIR)

        wrapper = cls.__new__(cls)
        wrapper.generator = generator
        wrapper.processor = processor
        wrapper.random_state = state.get("random_state", 0)
        wrapper.full_determinism = state.get("full_determinism", False)
        wrapper.generator.full_determinism = wrapper.full_determinism
        wrapper.fitted = state.get("fitted", True)
        return wrapper

    @staticmethod
    def _make_generator(
        generator,
        generator_params: dict = None,
        generator_kwargs: dict = None,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        if isinstance(generator, BaseGenerator):
            generator.full_determinism = full_determinism
            return generator

        generator_cls = SynthyverseGenerator._resolve_generator_class(generator)
        params = {}
        if generator_params is not None:
            params.update(generator_params)
        if generator_kwargs is not None:
            params.update(generator_kwargs)
        if "random_state" not in params:
            signature = inspect.signature(generator_cls.__init__)
            if "random_state" in signature.parameters:
                params["random_state"] = random_state
        if "full_determinism" not in params:
            params["full_determinism"] = full_determinism
        return generator_cls(**params)

    @staticmethod
    def _resolve_generator_class(generator):
        if isinstance(generator, str):
            from . import get_generator

            return get_generator(generator)

        if inspect.isclass(generator) and issubclass(generator, BaseGenerator):
            return generator

        raise TypeError(
            "generator must be a generator name, BaseGenerator subclass, "
            "or BaseGenerator instance."
        )

    @staticmethod
    def _load_generator_class(state: dict):
        module_name = state["generator_module"]
        class_name = state["generator_class"]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


def column_precision(series: pd.Series) -> int:
    """
    Return the maximum number of decimal places in a pandas Series,
    ignoring trailing zeros. NaNs are ignored.
    """

    def count_decimal_places(x) -> int:
        if pd.isna(x):
            return 0

        s = str(x)

        if "e" in s.lower():
            s = format(float(x), "f")

        if "." not in s:
            return 0

        frac = s.split(".", 1)[1].rstrip("0")
        return len(frac)

    return max(series.map(count_decimal_places), default=0)


class ConstraintEnforcer:
    """Apply simple column constraints before and after generation.

    Converts equality and inequality constraints into a model-space
    representation that is easier for generators to learn. Equalities remove
    one constrained column before training and reconstruct it after generation.
    Inequalities store the constrained side as a nonnegative difference and add
    the expression back during inverse transformation.

    This class is used internally by :class:`DataProcessor`, but you can use it
    directly when you want explicit control over constraint transformations.

    Args:
        constraints (list): Constraint strings with one binary operator.
            Equalities use ``=`` or ``==`` and inequalities use ``<``, ``<=``,
            ``>``, or ``>=``. One side must be a single feature that can be
            removed or converted to a margin. Examples:
            ``"total=part_a+part_b"``, ``"age>=18"``, and
            ``"income>expenses"``.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import ConstraintEnforcer
        >>>
        >>> X = pd.DataFrame({"part_a": [2], "part_b": [3], "total": [5]})
        >>> enforcer = ConstraintEnforcer(["total=part_a+part_b"])
        >>> X_model = enforcer.transform(X)
        >>> X_restored = enforcer.inverse_transform(X_model)
    """

    _OPERATORS = (">=", "<=", "==", "!=", "=", ">", "<")

    def __init__(self, constraints: list):
        self.constraints = list(constraints or [])
        self.equalities = []
        self.inequalities = []
        for c in self.constraints:
            parsed = self._parse(c)
            if parsed["op"] in ("=", "=="):
                self.equalities.append(parsed)
            else:
                self.inequalities.append(parsed)

    def _parse(self, constraint):
        if not isinstance(constraint, str):
            raise TypeError("Constraints must be strings.")
        raw = constraint
        c = constraint.replace(" ", "")
        matches = [op for op in self._OPERATORS if op in c]
        if not matches:
            raise ValueError(f"Constraint {raw} does not contain a supported operator.")
        op = matches[0]
        if op == "!=":
            raise ValueError(f"Constraint {raw} uses !=, which cannot be enforced.")
        parts = c.split(op)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Constraint {raw} must contain exactly one binary operator."
            )
        if any(extra_op in part for part in parts for extra_op in self._OPERATORS):
            raise ValueError(
                f"Constraint {raw} must contain exactly one binary operator."
            )
        return {"raw": raw, "left": parts[0], "op": op, "right": parts[1]}

    def referenced_columns(self, columns) -> set:
        refs = set()
        for c in self.equalities + self.inequalities:
            refs.update(self._expression_columns(c["left"], columns))
            refs.update(self._expression_columns(c["right"], columns))
        return refs

    def validate(self, X: pd.DataFrame) -> None:
        for c in self.equalities + self.inequalities:
            valid = self._holds(X, c)
            if not isinstance(valid, pd.Series):
                valid = pd.Series(bool(valid), index=X.index)
            if not bool(valid.all()):
                n = int((~valid).sum())
                raise ValueError(
                    f"Constraint {c['raw']} is violated by {n} training rows."
                )

    def _holds(self, X: pd.DataFrame, c: dict) -> pd.Series:
        left = self._eval(X, c["left"])
        right = self._eval(X, c["right"])
        if c["op"] in ("=", "=="):
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
                right
            ):
                return (left - right).abs() <= 1e-9
            return left == right
        if c["op"] == ">=":
            return left >= right
        if c["op"] == "<=":
            return left <= right
        if c["op"] == ">":
            return left > right
        return left < right

    def _eval(self, X: pd.DataFrame, expression: str):
        if expression in X.columns:
            return X[expression]
        return X.eval(expression)

    def _expression_columns(self, expression: str, columns) -> set:
        return {
            col
            for col in columns
            if re.search(rf"(?<![\w`]){re.escape(str(col))}(?![\w`])", expression)
        }

    def _target(self, x: pd.DataFrame, c: dict) -> str:
        target = c.get("target")
        if target is not None:
            if target not in x.columns:
                raise ValueError(f"Constraint {c['raw']} is missing column {target}.")
            return target
        if c["left"] in x.columns:
            target = c["left"]
        elif c["right"] in x.columns:
            target = c["right"]
        else:
            raise ValueError(
                f"Constraint {c['raw']} is not valid. One side must be a single feature."
            )
        c["target"] = target
        return target

    def _check_dependencies(self, x: pd.DataFrame, c: dict, dropped: set) -> None:
        available_or_dropped = set(x.columns) | dropped
        refs = self._expression_columns(c["left"], available_or_dropped)
        refs.update(self._expression_columns(c["right"], available_or_dropped))
        blocked = refs & dropped
        if blocked:
            cols = ", ".join(sorted(blocked))
            raise ValueError(
                f"Constraint {c['raw']} references columns dropped by earlier "
                f"equality constraints: {cols}."
            )

    def _minimum_margin(self, series: pd.Series, op: str):
        if op not in (">", "<"):
            return 0
        if pd.api.types.is_integer_dtype(series):
            return 1
        return 1e-12

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform constrained data into model space.

        Args:
            X (pd.DataFrame): Data containing the columns referenced by the
                constraints.

        Returns:
            pd.DataFrame: Copy of ``X`` with equality columns removed and
            inequality columns converted to differences.
        """

        x = X.copy()

        dropped = set()
        for c in self.equalities:
            self._check_dependencies(x, c, dropped)
            target = self._target(x, c)
            x = x.drop(columns=[target])
            dropped.add(target)
        for c in self.inequalities:
            self._check_dependencies(x, c, dropped)
            target = self._target(x, c)
            left = self._eval(x, c["left"])
            right = self._eval(x, c["right"])
            if c["op"] in (">", ">="):
                x[target] = left - right
            else:
                x[target] = right - left

        return x

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restore constrained columns after generation.

        Args:
            X (pd.DataFrame): Generated data in the transformed model-space
                schema.

        Returns:
            pd.DataFrame: Copy of ``X`` with constrained columns reconstructed.
        """
        x = X.copy()
        for c in reversed(self.equalities):
            target = c.get("target")
            if target == c["left"]:
                x[target] = self._eval(x, c["right"])
            elif target == c["right"]:
                x[target] = self._eval(x, c["left"])
            else:
                raise ValueError(
                    f"Constraint {c['raw']} cannot be reversed before transform is called."
                )
        for c in self.inequalities:
            target = c.get("target")
            if target not in x.columns:
                raise ValueError(
                    f"Constraint {c['raw']} cannot be reversed before transform is called."
                )
            margin = x[target].clip(lower=self._minimum_margin(x[target], c["op"]))
            if c["op"] in (">", ">="):
                if target == c["left"]:
                    x[target] = margin + self._eval(x, c["right"])
                else:
                    x[target] = self._eval(x, c["left"]) - margin
            elif target == c["left"]:
                x[target] = self._eval(x, c["right"]) - margin
            else:
                x[target] = margin + self._eval(x, c["left"])

        return x
