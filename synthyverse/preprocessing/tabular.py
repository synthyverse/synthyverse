import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)


class TabularPreprocessor:
    """Preprocessing class for tabular data.

    Mainly geared towards preprocessing for tabular synthetic data generation.
    This class is already implemented within the synthyverse API, so you should not need to use it directly.
    However, many of the class methods are general and can be used for other purposes (e.g., ML tasks and data cleaning), if you so desire.


    Args:
        discrete_features (list): List of column names that are discrete/categorical. Default: [].
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.preprocessing import TabularPreprocessor
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create preprocessor
        >>> preprocessor = TabularPreprocessor(
        ...     discrete_features=discrete_features,
        ...     random_state=42
        ... )
        >>>
        >>> # Execute preprocessing pipeline
        >>> X_prep = preprocessor.pipeline(
        ...     X,
        ...     missing_imputation_method="mean",
        ...     quantile_transform_numericals=True
        ... )
        >>>
        >>> # After generation, inverse transform
        >>> X_syn_original = preprocessor.inverse_pipeline(X_syn_prep)
    """

    def __init__(self, discrete_features: list = [], random_state: int = 0):
        self.discrete_features = discrete_features.copy()
        self.random_state = random_state
        self.pipeline_is_fitted = False

    def pipeline(
        self,
        X: pd.DataFrame,
        missing_imputation_method: str = "drop",
        retain_missingness: bool = False,
        encode_mixed_numerical_features: bool = False,
        quantile_transform_numericals: bool = False,
        constraints: list = [],
    ):
        """Execute a preprocessing pipeline for tabular synthetic data generation.

        Operations include:
        -> Imputing or dropping missing values (most generative models cannot natively handle missing values)
        -> Encoding mixed numerical features (most generative models cannot natively handle discontinuous numerical distributions)
        -> Quantile transforming numerical features (most generative models work better with normally distributed numerical features)
        -> Applying constraints (through pre- and postprocessing we can enforce intercolumn constraints)

        This method also retains information on the original dataset, e.g., numerical precision
        and data types. This way the inverse pipeline can be executed to ensure same format
        synthetic data post-generation.

        Args:
            X: Input data as a pandas DataFrame.
            missing_imputation_method: Method for handling missing values.
                Options: "drop", "random", "mean", "median", "mode".
            retain_missingness: If True, add indicator columns for missing values.
            encode_mixed_numerical_features: If True, encode mixed numerical-categorical
                features (e.g., zero-inflated features).
            quantile_transform_numericals: If True, apply quantile transformation to
                numerical features.
            constraints: List of constraint strings to enforce.

        Returns:
            pd.DataFrame: Preprocessed data ready for generative model training.
        """
        self.encode_mixed_numerical_features = encode_mixed_numerical_features
        self.retain_missingness = retain_missingness
        self.numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]

        self.ori_dtypes = X.dtypes
        self.ori_columns = X.columns
        self.ori_precision = {}
        for col in self.numerical_features:
            self.ori_precision[col] = calculate_column_precision(X[col])

        X_prep = X.copy()
        X_prep, missing_indicator_X = self.impute_missings(
            X_prep,
            method=missing_imputation_method,
            add_missing_indicator=retain_missingness,
        )

        # missingness indicator columns are discrete features
        if missing_indicator_X is not None:
            self.discrete_features.extend(missing_indicator_X.columns.tolist())
            X_prep = pd.concat([X_prep, missing_indicator_X], axis=1)

        # prep constraints
        X_prep, self.constraints = self.prep_constraints(X_prep, constraints.copy())

        # encode mixed numerical features if desired
        if encode_mixed_numerical_features:
            X_prep, mixed_discretes, self.mixed_features = (
                self.encode_mixed_numerical_categorical(
                    X_prep,
                    min_spike_prop=0.3,
                    rounding=5,
                    min_cont_unique=20,
                    max_discrete_values=3,
                    discrete_suffix="_MIXEDDISCRETE",
                )
            )
            if mixed_discretes is not None:
                self.discrete_features.extend(mixed_discretes.columns.tolist())
                X_prep = pd.concat([X_prep, mixed_discretes], axis=1)

        # ensure all data is numerical and quantile transform numericals if desired
        X_prep = self.scale(
            X_prep,
            numerical_transformer=(
                "quantile" if quantile_transform_numericals else "passthrough"
            ),
            categorical_transformer="ordinal",
        )
        self.pipeline_is_fitted = True
        return X_prep

    def inverse_pipeline(self, X: pd.DataFrame):
        """Inverse the preprocessing pipeline.

        This ensures the same format synthetic data post-generation. For example in
        terms of columns, data types, numerical precision, constraints, etc.

        Args:
            X: Preprocessed data to inverse transform.

        Returns:
            pd.DataFrame: Data with preprocessing reversed, matching original format.

        Raises:
            Exception: If pipeline has not been fitted yet.
        """
        if not self.pipeline_is_fitted:
            raise Exception(
                "Pipeline must be fitted before inverse_pipeline can be called."
            )

        X_reverse = X.copy()

        # inverse any scaling transformations
        X_reverse = self.inverse_scale(X_reverse)

        # decode mixed numerical features if desired
        if self.encode_mixed_numerical_features:
            X_reverse = self.decode_mixed_numerical_categorical(X_reverse)

        # align precision of numerical columns
        for col in [x for x in self.numerical_features if x in X_reverse.columns]:
            X_reverse[col] = X_reverse[col].astype(float).round(self.ori_precision[col])

        # apply constraints
        X_reverse = self.apply_constraints(X_reverse)

        # reinstate missingness if retained
        if self.retain_missingness:
            X_reverse = self.reinstate_missings(X_reverse)

        # align columns with original columns
        X_reverse = X_reverse[self.ori_columns]

        # align dtypes with original datatypes
        X_reverse = X_reverse.astype(self.ori_dtypes)

        return X_reverse

    def impute_missings(
        self,
        X: pd.DataFrame,
        method: str = "random",  # drop, random, mean, median, mode
        add_missing_indicator: bool = True,
        missing_indicator_suffix: str = "_MISSING",
    ):
        """Impute missing values in the dataframe.

        Args:
            X: Input dataframe with missing values.
            method: Method for imputation. Options:
                - "drop": drop missing rows
                - "random": randomly impute missing values
                - "mean": impute missing values with the mean
                - "median": impute missing values with the median
                - "mode": impute missing values with the mode
            add_missing_indicator: If True, add indicator columns for missing values.
            missing_indicator_suffix: Suffix to add to indicator columns for missing values.
                Ensure these do not already exist in the data.

        Returns:
            tuple: Tuple of (imputed_X, missing_indicator_X) where missing_indicator_X
                is None if add_missing_indicator is False.

        Raises:
            Exception: If method is "drop" and add_missing_indicator is True, or if
                indicator suffix conflicts with existing columns.
        """
        self.missing_indicator_suffix = missing_indicator_suffix

        if method == "drop" and add_missing_indicator:
            raise Exception(
                "Cannot drop missing rows AND add a missing value indicator."
            )

        numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]

        # if there are no missing values, return already
        if not X[numerical_features].isna().any().any():
            return X, None

        # add indicator columns for missing values
        if add_missing_indicator:
            missing_indicator_X = []
            for col in numerical_features:
                if col.endswith(missing_indicator_suffix):
                    raise Exception(
                        f"Column {col} exists in the data, but this conflicts with the missing indicator suffix. Please rename the column."
                    )
                if not X[col].isna().any():
                    continue
                missings = X[col].isna().to_frame()
                missings.columns = [col + missing_indicator_suffix]
                missing_indicator_X.append(missings)
            missing_indicator_X = pd.concat(missing_indicator_X, axis=1)
        else:
            missing_indicator_X = None

        # do the actual handling of missing values
        imputed_X = X.copy()
        if method == "drop":
            imputed_X = imputed_X.dropna(subset=numerical_features)
        else:
            for col in numerical_features:
                if method == "random":
                    imputed_X.loc[imputed_X[col].isna(), col] = (
                        imputed_X[col]
                        .sample(
                            imputed_X[col].isna().sum(), random_state=self.random_state
                        )
                        .values
                    )
                elif method == "mean":
                    imputed_X[col] = imputed_X[col].fillna(imputed_X[col].mean())
                elif method == "median":
                    imputed_X[col] = imputed_X[col].fillna(imputed_X[col].median())
                elif method == "mode":
                    imputed_X[col] = imputed_X[col].fillna(
                        imputed_X[col].mode().iloc[0]
                    )

        return imputed_X, missing_indicator_X

    def reinstate_missings(self, X: pd.DataFrame):
        """Reinstate missing values (which were previously imputed) in the dataframe.

        Args:
            X: Dataframe with missing indicator columns.

        Returns:
            pd.DataFrame: Dataframe with missing values reinstated based on indicators.
        """
        missing_cols = [
            col for col in X.columns if col.endswith(self.missing_indicator_suffix)
        ]
        missing_cols = X[missing_cols].copy()

        reinstated_X = X.copy()
        reinstated_X = reinstated_X.drop(columns=missing_cols.columns.tolist())

        for ori, miss in zip(
            missing_cols.columns.str.split("_").str[0],
            missing_cols.columns,
        ):
            reinstated_X.loc[missing_cols[miss] == 1, ori] = np.nan
        return reinstated_X

    def scale(
        self,
        X: pd.DataFrame,
        numerical_transformer: str = "standard",  # standard, minmax, quantile, passthrough
        categorical_transformer: str = "one-hot",  # one-hot, label, passthrough
        numerical_transformer_hparams: dict = {},
        categorical_transformer_hparams: dict = {},
    ):
        """Scale and/or numerically encode a dataset.

        Supports a variety of transformers for both numerical and categorical features.

        Args:
            X: Input dataframe to transform.
            numerical_transformer: Transformer for numerical features. Options:
                - "standard": standard scaling
                - "minmax": minmax scaling
                - "quantile": quantile transformation
                - "passthrough": passthrough (no transformation)
                - "none": passthrough (no transformation)
            categorical_transformer: Transformer for categorical features. Options:
                - "one-hot": one-hot encoding
                - "ordinal": ordinal encoding
                - "label": ordinal encoding (alias for ordinal)
                - "passthrough": passthrough (no transformation)
                - "none": passthrough (no transformation)
            numerical_transformer_hparams: Hyperparameters for the numerical transformer.
            categorical_transformer_hparams: Hyperparameters for the categorical transformer.

        Returns:
            pd.DataFrame: Transformed dataframe with scaled/encoded features.
        """
        encoders = {
            "one-hot": OneHotEncoder(sparse_output=False),
            "ordinal": OrdinalEncoder(),
            "label": OrdinalEncoder(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "quantile": QuantileTransformer(
                n_quantiles=1000,
                output_distribution="normal",
                subsample=10_000,
                random_state=self.random_state,
            ),
            "none": FunctionTransformer(feature_names_out="one-to-one"),
            "passthrough": FunctionTransformer(feature_names_out="one-to-one"),
        }

        self.prescaling_col_order = X.columns.tolist()

        numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]
        categorical_features = [
            col for col in X.columns if col in self.discrete_features
        ]  # ensure only use columns which are in the data
        numerical_transformer = encoders[numerical_transformer]
        numerical_transformer.set_params(**numerical_transformer_hparams)

        X_num = numerical_transformer.fit_transform(X[numerical_features])
        X_num = pd.DataFrame(
            X_num, columns=numerical_transformer.get_feature_names_out()
        )

        categorical_transformer = encoders[categorical_transformer]
        categorical_transformer.set_params(**categorical_transformer_hparams)

        X_cat = categorical_transformer.fit_transform(X[categorical_features])

        X_cat = pd.DataFrame(
            X_cat, columns=categorical_transformer.get_feature_names_out()
        )

        X_num, X_cat = X_num.reset_index(drop=True), X_cat.reset_index(drop=True)
        X_transformed = pd.concat([X_num, X_cat], axis=1)

        self.transformers = {
            "numerical": numerical_transformer,
            "categorical": categorical_transformer,
        }

        return X_transformed

    def inverse_scale(self, X: pd.DataFrame):
        """Inverse the scaling transformations applied to the dataframe.

        Args:
            X: Scaled/encoded dataframe to inverse transform.

        Returns:
            pd.DataFrame: Dataframe with scaling/encoding reversed.
        """

        X_num = self.transformers["numerical"].inverse_transform(
            X[self.transformers["numerical"].get_feature_names_out()]
        )
        X_num = pd.DataFrame(
            X_num, columns=self.transformers["numerical"].feature_names_in_
        )
        X_cat = self.transformers["categorical"].inverse_transform(
            X[self.transformers["categorical"].get_feature_names_out()]
        )
        X_cat = pd.DataFrame(
            X_cat, columns=self.transformers["categorical"].get_feature_names_out()
        )
        X_transformed = pd.concat([X_num, X_cat], axis=1)

        # align columns with original columns
        X_transformed = X_transformed[self.prescaling_col_order]
        return X_transformed

    def encode_mixed_numerical_categorical(
        self,
        X: pd.DataFrame,
        min_spike_prop: float = 0.3,
        rounding: int = 6,
        min_cont_unique: int = 20,
        max_discrete_values: int = 3,
        discrete_suffix: str = "_MIXEDDISCRETE",
    ):
        """Encode features which are a mix of continuous values + discrete spikes.

        Common examples are zero-inflated features. Discrete values are one-hot encoded,
        and their values replaced by random samples from the original column.

        Args:
            X: Input dataframe.
            min_spike_prop: Minimum proportion of same values which constitute a
                discrete spike.
            rounding: Number of decimal places to round before counting unique values.
            min_cont_unique: Minimum number of distinct (rounded) values for a column
                to be considered mixed (and not purely discrete).
            max_discrete_values: Maximum number of discrete values to return per column.
            discrete_suffix: Suffix to add to the column names of the discrete values.

        Returns:
            tuple: Tuple of (X_new, discretes, mixed_features) where:
                - X_new: Dataframe with mixed features encoded
                - discretes: Dataframe with discrete indicator columns (or None)
                - mixed_features: Dictionary mapping column names to discrete spike values

        Raises:
            Exception: If discrete_suffix conflicts with existing columns.
        """
        self.discrete_suffix = discrete_suffix
        # dictionary of mixed feature-value pairs
        numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]
        mixed_features = self._detect_mixed_features(
            X[numerical_features],
            min_spike_prop,
            rounding,
            min_cont_unique,
            max_discrete_values,
        )

        print(f"Columns {[*mixed_features]} detected as mixed numerical-categorical.")

        X_new = X.copy()
        discretes = []
        for key, values in mixed_features.items():
            if key.endswith(discrete_suffix):
                raise Exception(
                    f"Column {key} exists in the data, but this conflicts with the discrete suffix. Please rename the column."
                )
            for i, val in enumerate(values):
                mask = X[key] == val
                discretes.append(
                    pd.DataFrame(mask, columns=[key + f"{discrete_suffix}_{i}"])
                )
                X_new.loc[mask, key] = (
                    X_new[key].sample(mask.sum(), random_state=self.random_state).values
                )

        discretes = pd.concat(discretes, axis=1) if len(discretes) > 0 else None

        return X_new, discretes, mixed_features

    def decode_mixed_numerical_categorical(
        self,
        X: pd.DataFrame,
    ):
        """Return mixed discrete-numerical columns to their original state.

        Args:
            X: Dataframe with encoded mixed features.

        Returns:
            pd.DataFrame: Dataframe with mixed features decoded to original format.
        """
        for key, values in self.mixed_features.items():
            for i, val in enumerate(values):
                colname = key + f"{self.discrete_suffix}_{i}"
                mask = X[colname] == 1
                X.loc[mask, key] = val
                X = X.drop(columns=[colname])
        return X

    def _detect_mixed_features(
        self,
        df: pd.DataFrame,
        min_spike_prop: float = 0.3,
        rounding: int = 6,
        min_cont_unique: int = 20,
        max_discrete_values: int = 3,
    ):

        result = {}

        for col in df.columns:
            s = df[col].copy()

            # Round to reduce float noise before counting unique values
            sr = s.round(rounding)

            vc = sr.value_counts(dropna=False)
            n = int(vc.sum())
            props = vc / n

            # Candidate spikes: values with large mass
            spikes = props[props >= min_spike_prop].index.tolist()

            if not spikes:
                continue

            # Check that there's still a meaningful continuous "tail" outside spikes
            mask_non_spike = ~sr.isin(spikes)
            cont_unique = sr[mask_non_spike].nunique()

            if cont_unique >= min_cont_unique:
                # Sort spikes by value and cap the length
                spikes_sorted = sorted(spikes)[:max_discrete_values]
                # Cast to builtins (float) for clean JSON/serialization
                result[col] = [float(v) for v in spikes_sorted]

        return result

    def prep_constraints(self, X: pd.DataFrame, constraints: list):
        """Preprocess a dataframe so constraints can be enforced post synthetic data generation.

        For equality constraints, we can remove the "left" side of the constraint, as it
        can be computed post-hoc from the "right" side. For inequality constraints, we can
        replace the "left" side with the diff to the right side.

        Args:
            X: Input dataframe.
            constraints: List of constraint strings (e.g., ["col1=col2+col3", "col1<col2"]).

        Returns:
            tuple: Tuple of (X_new, constraints_dict) where:
                - X_new: Preprocessed dataframe with constraints prepared
                - constraints_dict: Dictionary storing constraint information
        """
        X_new = X.copy()
        constraints_dict = {}
        for constraint in constraints:
            if "=" in constraint and not "<" in constraint and not ">" in constraint:
                # equality constraint

                constraints_dict["equality"] = constraint
                # remove left feature, as it can be exactly computed post-hoc from right side of constraint
                left, right = tuple(constraint.split("="))
                X_new = X_new.drop(columns=[left])
            else:
                constraint = constraint.replace("=", "")
                constraints_dict["inequality"] = constraint
                left, right = tuple(
                    constraint.split("<")
                    if "<" in constraint
                    else constraint.split(">")
                )
                diff = X[left] - X.eval(right)
                X_new[left] = diff

        return X_new, constraints_dict

    def apply_constraints(self, X: pd.DataFrame):
        """Apply constraints to the generated dataframe.

        For equality constraints, we can compute the "left" side from the "right" side
        exactly. For inequality constraints, we can add the "right" side to the diff and
        thus enforce the constraint. Inequality constraints will only strictly hold if the
        generator outputs values within the range of the training data.

        Args:
            X: Generated dataframe to apply constraints to.

        Returns:
            pd.DataFrame: Dataframe with constraints applied.
        """
        X_new = X.copy()

        for constraint in self.constraints.keys():
            if constraint == "equality":
                constraint = self.constraints[constraint]
                left, right = tuple(constraint.split("="))
                X_new[left] = X_new.eval(right)

            elif constraint == "inequality":
                constraint = self.constraints[constraint]

                left, right = tuple(
                    constraint.split("<")
                    if "<" in constraint
                    else constraint.split(">")
                )
                X_new[left] = X_new[left] + X_new.eval(right)

        return X_new


def calculate_column_precision(col_values: pd.Series) -> int:
    """Calculate the maximum precision within a numerical column.

    Args:
        col_values: Pandas Series containing numerical values.

    Returns:
        int: Maximum precision (number of decimal places) needed.
    """

    # Convert to string and split by decimal point
    str_values = col_values.dropna().astype(str)

    # Vectorized operation to find decimal parts
    decimal_parts = str_values.str.split(".").str[-1]

    # Handle cases where there's no decimal point (integer values)
    has_decimal = str_values.str.contains(".")

    # Calculate precision for each value
    def get_precision(decimal_part, has_dec):
        if not has_dec or pd.isna(decimal_part):
            return 0
        # Find last non-zero digit from the right
        for i in range(len(decimal_part) - 1, -1, -1):
            if decimal_part[i] != "0":
                return i + 1
        return 0

    # Apply the precision calculation vectorized
    precisions = [
        get_precision(part, has_dec)
        for part, has_dec in zip(decimal_parts, has_decimal)
    ]

    return max(precisions) if precisions else 0
