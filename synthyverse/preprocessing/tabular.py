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
    """
    Preprocessing class for tabular data. Mainly geared towards preprocessing for tabularsynthetic data generation.
    However, many of the class methods are general and can be used for other purposes (e.g. ML tasks and data cleaning).
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
        """
        Executes a preprocessing pipeline for tabular synthetic data generation. Operations include:
        - imputing missing values - most generative models cannot natively handle missing values
        - encoding mixed numerical features - most generative models cannot natively handle discontinuous numerical distributions
        - quantile transforming numerical features - most generative models work better with normally distributed numerical features
        - applying constraints - through pre- and postprocessing we can enforce intercolumn constraints.
        This method also retains information on the original dataset, e.g., numerical precision and data types.
        This way the inverse pipeline can be executed to ensure same format synthetic data post-generation.
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
        """
        Inverse the preprocessing pipeline. This ensures the same format synthetic data post-generation.
        For example in terms of columns, data types, numerical precision, constraints, etc.
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
        """
        Impute missing values in the dataframe.
        Parameters:
            X: pd.DataFrame
            method: str
                - "drop": drop missing rows
                - "random": randomly impute missing values
                - "mean": impute missing values with the mean
                - "median": impute missing values with the median
                - "mode": impute missing values with the mode
            missing_indicator_suffix: str
                - suffix to add to indicator columns for missing values (ensure these do not already exist in the data)
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
        """
        Reinstate missing values (which were previously imputed) in the dataframe.
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
        """ "
        Method for scaling and/or numerical encoding of a dataset. Supports a variety of transformers.
        Parameters:
            X: pd.DataFrame
            numerical_transformer: str
                - "standard": standard scaling
                - "minmax": minmax scaling
                - "quantile": quantile transformation
                - "passthrough": passthrough
                - "none": passthrough
            categorical_transformer: str
                - "one-hot": one-hot encoding
                - "ordinal": ordinal encoding
                - "label": ordinal encoding
                - "passthrough": passthrough
                - "none": passthrough
            numerical_transformer_hparams: dict
                - hyperparameters for the numerical transformer
            categorical_transformer_hparams: dict
                - hyperparameters for the categorical transformer
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
        """
        Inverse the scaling transformations applied to the dataframe.
        Parameters:
            X: pd.DataFrame
            transformers: dict
                - the transformers used to scale the dataframe
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
        """
        Encode features which are a mix of continuous values + discrete spikes. Common examples are zero-inflated features.
        Discrete values are one-hot encoded, and their values replaced by random samples from the original column.
        Parameters:
            X: pd.DataFrame
            min_spike_prop: float
                - minimum proportion of same values which constitute a discrete spike
            rounding: int
                - number of decimal places to round before counting unique values
            min_cont_unique: int
                - minimum number of distinct (rounded) values for a column to be considered mixed (and not purely discrete)
            max_discrete_values: int
                - maximum number of discrete values to return per column
            discrete_suffix: str
                - suffix to add to the column names of the discrete values

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
        """
        Return mixed discrete-numerical columns to their original state.
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
        """
        Detect numeric features that are a mix of continuous values + discrete spikes.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        min_spike_prop : float, default 0.05
            "How discrete" a spike must be: a value is considered a discrete spike in a column
            if it accounts for at least this fraction of the non-missing rows in that column.
            Raise this if you want *fewer* columns to qualify (stricter), lower it to be looser.
        rounding : int, default 6
            Number of decimal places to round before counting unique values (helps merge
            near-identical floats like 0.30000000004).
        min_cont_unique : int, default 10
            Require at least this many distinct (rounded) values *outside* the detected spikes
            for the column to be considered “mixed” rather than purely discrete.
        max_discrete_values : int, default 3
            Upper bound on how many spike values to return per column (safety against
            pathological cases).

        Returns
        -------
        dict[str, list[float]]
            Mapping of column name -> sorted list of detected discrete spike values.
            Only columns that meet the “mixed” criterion are included.

        Notes
        -----
        - Typical zero-inflated columns will be captured by setting min_spike_prop
        somewhere around 0.05–0.20 depending on your dataset size.
        - If a column is *fully* discrete (e.g., only a handful of unique values total),
        it will be *excluded* unless there are at least `min_cont_unique` unique
        non-spike values remaining.
        """

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
        """
        Preprocess a dataframe s.t. constraints can be enforced post synthetic data generation.
        For equality constraints, we can remove the "left" side of the constraint, as it can be computed post-hoc from the "right" side.
        For inequality constraints, we can replace the "left" side with the diff to the right side.
        Parameters:
            X: pd.DataFrame
            constraints: list
                - list of constraints
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
        """
        Apply constraints to the generated dataframe.
        For equality constraints, we can compute the "left" side from the "right" side exactly.
        For inequality constraints, we can add the "right" side to the diff and thus enforce the constraint.
        Inequality constraints will only strictly hold if the generator outputs values within the range of the training data.
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
    """
    Calculate the maximum precision within a numerical column.

    Args:
        col_values: Pandas Series containing numerical values

    Returns:
        int: Maximum precision (number of decimal places) needed
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
