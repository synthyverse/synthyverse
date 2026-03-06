import copy
import pickle
from pathlib import Path
from typing import Dict, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from ..utils.missingness import (
    apply_fitted_numeric_imputer,
    apply_random_imputation,
    fit_random_imputation_samples,
    fit_transform_numeric_imputer,
    validate_missing_imputation_method,
)


TTabularBaseGenerator = TypeVar("TTabularBaseGenerator", bound="TabularBaseGenerator")


class TabularBaseGenerator:

    def __init__(
        self,
        constraints: Union[list, str, None] = None,
        missing_imputation_method: str = "drop",
        retain_missingness: bool = False,
        random_state: int = 0,
    ):
        self.random_state = random_state
        if constraints is None:
            constraints = []
        if isinstance(constraints, str):
            self.constraints = [constraints]
        else:
            self.constraints = constraints
        self.missing_imputation_method = missing_imputation_method
        self.retain_missingness = retain_missingness

    def fit(self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None):
        self.base_discrete_features = discrete_features.copy()
        X_prep = self._fit_preprocessing(X.copy())
        X_val_prep = (
            self._transform_preprocessing(X_val.copy()) if X_val is not None else None
        )

        # only include discrete columns that actually exist post preprocessing
        model_discrete_features = [
            col for col in self.discrete_features_ if col in X_prep.columns
        ]

        self._fit_model(
            X=X_prep, discrete_features=model_discrete_features, X_val=X_val_prep
        )

        return self

    def generate(self, n: int):
        syn_X = self._generate_data(n)

        # inverse the preprocessing pipeline
        syn_X = self._inverse_preprocessing(syn_X)

        return syn_X

    def save_model(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Work on a copy so the in-memory trained model remains untouched.
        model_to_save = copy.deepcopy(self)
        model_to_save._cleanup_for_save()

        with path.open("wb") as file:
            pickle.dump(model_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)

        return path

    @classmethod
    def load_model(
        cls: type[TTabularBaseGenerator], path: Union[str, Path]
    ) -> TTabularBaseGenerator:
        path = Path(path)
        with path.open("rb") as file:
            model = pickle.load(file)

        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded model type {type(model).__name__} does not match expected type {cls.__name__}."
            )
        return model

    def _cleanup_for_save(self) -> None:
        for attr in self._transient_attributes_for_save():
            if hasattr(self, attr):
                delattr(self, attr)
        self._cleanup_additional_state_for_save()

    def _transient_attributes_for_save(self) -> Tuple[str, ...]:
        # Random/imputation sampling pools are only needed during preprocessing transforms.
        return ("_rng", "imputation_samples_")

    def _cleanup_additional_state_for_save(self) -> None:
        # Subclasses can override to strip large, generation-irrelevant objects.
        return None

    def _fit_model(
        self,
        X: pd.DataFrame,
        discrete_features: list = None,
        X_val: pd.DataFrame = None,
    ):
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _generate_data(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _generate_data")

    def _fit_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        self.pipeline_is_fitted = False
        self._rng = np.random.default_rng(self.random_state)
        self.ori_columns = X.columns
        self.ori_dtypes = X.dtypes
        self.discrete_features_ = self.base_discrete_features.copy()
        self.initial_numerical_features_ = [
            col for col in X.columns if col not in self.discrete_features_
        ]
        self.ori_precision = {
            col: calculate_column_precision(X[col])
            for col in self.initial_numerical_features_
        }

        X_prep = X.copy()
        X_prep = self._fit_transform_missingness(X_prep)
        X_prep = self._fit_transform_constraints(X_prep)
        X_prep = self._fit_transform_scale(X_prep)

        self.pipeline_is_fitted = True
        return X_prep

    def _transform_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "pipeline_is_fitted", False):
            raise Exception("Preprocessing must be fitted on training data first.")

        X_prep = X.copy()
        X_prep = self._transform_missingness(X_prep)
        X_prep = self._transform_constraints(X_prep)
        X_prep = self._transform_scale(X_prep)
        return X_prep

    def _inverse_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "pipeline_is_fitted", False):
            raise Exception(
                "Preprocessing must be fitted before inverse preprocessing."
            )

        X_reverse = X.copy()
        X_reverse = self._inverse_scale(X_reverse)

        for col in [
            c for c in self.initial_numerical_features_ if c in X_reverse.columns
        ]:
            X_reverse[col] = X_reverse[col].astype(float).round(self.ori_precision[col])

        X_reverse = self._apply_constraints(X_reverse)
        X_reverse = self._reinstate_missings(X_reverse)
        X_reverse = X_reverse[self.ori_columns]
        X_reverse = X_reverse.astype(self.ori_dtypes)
        return X_reverse

    def _fit_transform_missingness(self, X: pd.DataFrame) -> pd.DataFrame:
        self.missing_indicator_suffix_ = "_MISSING"
        self.has_missingness_ = self.retain_missingness
        self.imputation_method_ = self.missing_imputation_method
        self.numerical_features_ = [
            col for col in X.columns if col not in self.base_discrete_features
        ]
        self.missing_numeric_columns_ = []
        self.missing_indicator_columns_ = []
        self.missing_indicator_ = None
        self.numeric_imputer_ = None
        self.all_missing_numeric_columns_ = []
        self.imputation_samples_: Dict[str, np.ndarray] = {}

        if self.imputation_method_ == "drop" and self.has_missingness_:
            raise Exception(
                "Cannot drop missing rows and retain missingness indicators."
            )

        try:
            validate_missing_imputation_method(self.imputation_method_)
        except ValueError as exc:
            raise Exception(str(exc)) from exc

        if self.has_missingness_ and len(self.numerical_features_) > 0:
            self.missing_indicator_ = MissingIndicator(
                features="missing-only",
                error_on_new=False,
            )
            indicator_values = self.missing_indicator_.fit_transform(
                X[self.numerical_features_]
            )
            self.missing_numeric_columns_ = [
                self.numerical_features_[i] for i in self.missing_indicator_.features_
            ]
            self.missing_indicator_columns_ = [
                f"{col}{self.missing_indicator_suffix_}"
                for col in self.missing_numeric_columns_
            ]
            if len(self.missing_indicator_columns_) > 0:
                indicator_df = pd.DataFrame(
                    indicator_values.astype(int),
                    columns=self.missing_indicator_columns_,
                    index=X.index,
                )
            else:
                indicator_df = pd.DataFrame(index=X.index)
        else:
            indicator_df = pd.DataFrame(index=X.index)

        X_new = X.copy()
        if len(self.numerical_features_) == 0:
            if self.has_missingness_:
                X_new = pd.concat([X_new, indicator_df.reindex(X_new.index)], axis=1)
                self.discrete_features_.extend(self.missing_indicator_columns_)
            return X_new

        if self.imputation_method_ == "drop":
            X_new = X_new.dropna(subset=self.numerical_features_)
        elif self.imputation_method_ == "keep":
            pass
        elif self.imputation_method_ == "random":
            self.imputation_samples_ = fit_random_imputation_samples(
                X_new, self.numerical_features_
            )
            X_new = apply_random_imputation(
                X_new,
                self.numerical_features_,
                self.imputation_samples_,
                self._rng,
            )
        elif self.imputation_method_ in {
            "mean",
            "median",
            "most_frequent",
            "missforest",
        }:
            X_new, self.numeric_imputer_, self.all_missing_numeric_columns_ = (
                fit_transform_numeric_imputer(
                    X=X_new,
                    numerical_features=self.numerical_features_,
                    imputation_method=self.imputation_method_,
                    random_state=self.random_state,
                )
            )

        if self.has_missingness_:
            indicator_df = indicator_df.reindex(X_new.index, fill_value=0)
            X_new = pd.concat([X_new, indicator_df], axis=1)
            self.discrete_features_.extend(self.missing_indicator_columns_)

        return X_new

    def _transform_missingness(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        if len(self.numerical_features_) == 0:
            return X_new

        if self.imputation_method_ == "drop":
            X_new = X_new.dropna(subset=self.numerical_features_)
        elif self.imputation_method_ == "keep":
            pass
        elif self.imputation_method_ == "random":
            X_new = apply_random_imputation(
                X_new,
                self.numerical_features_,
                self.imputation_samples_,
                self._rng,
            )
        elif self.imputation_method_ in {
            "mean",
            "median",
            "most_frequent",
            "missforest",
        }:
            X_new = apply_fitted_numeric_imputer(
                X=X_new,
                numerical_features=self.numerical_features_,
                imputer=self.numeric_imputer_,
                imputation_method=self.imputation_method_,
                all_missing_numeric_columns=self.all_missing_numeric_columns_,
            )

        if self.has_missingness_:
            if self.missing_indicator_ is not None:
                indicator_values = self.missing_indicator_.transform(
                    X[self.numerical_features_]
                )
                indicator_df = pd.DataFrame(
                    indicator_values.astype(int),
                    columns=self.missing_indicator_columns_,
                    index=X.index,
                )
                indicator_df = indicator_df.reindex(X_new.index, fill_value=0)
                X_new = pd.concat([X_new, indicator_df], axis=1)

        return X_new

    def _reinstate_missings(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.has_missingness_ or len(self.missing_indicator_columns_) == 0:
            return X

        X_new = X.copy()
        for indicator_col in self.missing_indicator_columns_:
            base_col = indicator_col.replace(self.missing_indicator_suffix_, "")
            if indicator_col in X_new.columns and base_col in X_new.columns:
                X_new.loc[X_new[indicator_col] == 1, base_col] = np.nan
                X_new = X_new.drop(columns=[indicator_col])
        return X_new

    def _fit_transform_constraints(self, X: pd.DataFrame) -> pd.DataFrame:
        self.constraints_ = {"equalities": [], "inequalities": []}
        X_new = X.copy()

        for raw_constraint in self.constraints:
            constraint = raw_constraint.replace(" ", "")
            if "=" in constraint and "<" not in constraint and ">" not in constraint:
                left, right = tuple(constraint.split("="))
                self.constraints_["equalities"].append((left, right))
                if left in X_new.columns:
                    X_new = X_new.drop(columns=[left])
            else:
                if "<" in constraint:
                    left, right = tuple(constraint.replace("=", "").split("<"))
                    operator = "<"
                elif ">" in constraint:
                    left, right = tuple(constraint.replace("=", "").split(">"))
                    operator = ">"
                else:
                    continue
                self.constraints_["inequalities"].append((left, right, operator))
                if left in X_new.columns:
                    X_new[left] = X_new[left] - X_new.eval(right)

        return X_new

    def _transform_constraints(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for left, _right in self.constraints_["equalities"]:
            if left in X_new.columns:
                X_new = X_new.drop(columns=[left])
        for left, right, _operator in self.constraints_["inequalities"]:
            if left in X_new.columns:
                X_new[left] = X_new[left] - X_new.eval(right)
        return X_new

    def _apply_constraints(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for left, right in self.constraints_["equalities"]:
            X_new[left] = X_new.eval(right)
        for left, right, _operator in self.constraints_["inequalities"]:
            if left in X_new.columns:
                X_new[left] = X_new[left] + X_new.eval(right)
        return X_new

    def _fit_transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        self.prescaling_col_order_ = X.columns.tolist()
        self.scale_has_numerical_ = False
        self.scale_has_categorical_ = False

        numerical_features = [
            col for col in X.columns if col not in self.discrete_features_
        ]
        categorical_features = [
            col for col in X.columns if col in self.discrete_features_
        ]
        self.scaler_input_numerical_ = numerical_features
        self.scaler_input_categorical_ = categorical_features

        self.transformers_ = {
            "numerical": FunctionTransformer(feature_names_out="one-to-one"),
            "categorical": OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        }

        parts = []
        if len(numerical_features) > 0:
            self.scale_has_numerical_ = True
            X_num = self.transformers_["numerical"].fit_transform(X[numerical_features])
            X_num = pd.DataFrame(
                X_num,
                columns=self.transformers_["numerical"].get_feature_names_out(),
                index=X.index,
            )
            parts.append(X_num)
        if len(categorical_features) > 0:
            self.scale_has_categorical_ = True
            X_cat = self.transformers_["categorical"].fit_transform(
                X[categorical_features]
            )
            X_cat = pd.DataFrame(
                X_cat,
                columns=self.transformers_["categorical"].get_feature_names_out(),
                index=X.index,
            )
            parts.append(X_cat)

        X_transformed = (
            pd.concat(parts, axis=1) if len(parts) > 0 else pd.DataFrame(index=X.index)
        )
        self.preprocessed_columns_ = X_transformed.columns.tolist()
        return X_transformed

    def _transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.scale_has_numerical_ and len(self.scaler_input_numerical_) > 0:
            X_num = self.transformers_["numerical"].transform(
                X[self.scaler_input_numerical_]
            )
            X_num = pd.DataFrame(
                X_num,
                columns=self.transformers_["numerical"].get_feature_names_out(),
                index=X.index,
            )
            parts.append(X_num)

        if self.scale_has_categorical_ and len(self.scaler_input_categorical_) > 0:
            X_cat = self.transformers_["categorical"].transform(
                X[self.scaler_input_categorical_]
            )
            X_cat = pd.DataFrame(
                X_cat,
                columns=self.transformers_["categorical"].get_feature_names_out(),
                index=X.index,
            )
            parts.append(X_cat)

        X_transformed = (
            pd.concat(parts, axis=1) if len(parts) > 0 else pd.DataFrame(index=X.index)
        )
        return X_transformed.reindex(columns=self.preprocessed_columns_, fill_value=0)

    def _inverse_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if (
            self.scale_has_numerical_
            and self.transformers_.get("numerical") is not None
        ):
            num_cols = self.transformers_["numerical"].get_feature_names_out()
            if len(num_cols) > 0 and all(col in X.columns for col in num_cols):
                X_num = self.transformers_["numerical"].inverse_transform(X[num_cols])
                X_num = pd.DataFrame(
                    X_num,
                    columns=self.transformers_["numerical"].feature_names_in_,
                    index=X.index,
                )
                parts.append(X_num)

        if (
            self.scale_has_categorical_
            and self.transformers_.get("categorical") is not None
        ):
            cat_cols = self.transformers_["categorical"].get_feature_names_out()
            if len(cat_cols) > 0 and all(col in X.columns for col in cat_cols):
                X_cat_input = X[cat_cols].round().astype(int)
                X_cat = self.transformers_["categorical"].inverse_transform(X_cat_input)
                X_cat = pd.DataFrame(
                    X_cat,
                    columns=self.transformers_["categorical"].get_feature_names_out(),
                    index=X.index,
                )
                parts.append(X_cat)

        X_transformed = (
            pd.concat(parts, axis=1) if len(parts) > 0 else pd.DataFrame(index=X.index)
        )
        return X_transformed[self.prescaling_col_order_]


def calculate_column_precision(col_values: pd.Series) -> int:
    str_values = col_values.dropna().astype(str)
    decimal_parts = str_values.str.split(".").str[-1]
    has_decimal = str_values.str.contains(".", regex=False)

    def get_precision(decimal_part, has_dec):
        if not has_dec or pd.isna(decimal_part):
            return 0
        for i in range(len(decimal_part) - 1, -1, -1):
            if decimal_part[i] != "0":
                return i + 1
        return 0

    precisions = [
        get_precision(part, has_dec)
        for part, has_dec in zip(decimal_parts, has_decimal)
    ]
    return max(precisions) if precisions else 0
