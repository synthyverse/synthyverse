from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


class GowerLikePreprocessor:
    """Transform tabular data into the internal mixed-type metric space.

    Numerical features are min-max scaled to [0, 1] using a reference dataset.
    Discrete features are one-hot encoded and divided by two, so an L1 distance
    of two one-hot columns contributes one categorical mismatch.
    """

    def __init__(
        self,
        discrete_features: Optional[list] = None,
        categorical_handle_unknown: str = "error",
    ):
        self.discrete_features = list(discrete_features or [])
        self.categorical_handle_unknown = categorical_handle_unknown

    def fit(
        self,
        reference_data: pd.DataFrame,
        categorical_fit_data: Optional[Iterable[pd.DataFrame]] = None,
    ):
        self.columns_ = reference_data.columns.tolist()
        self._validate_feature_columns(reference_data, "reference_data")

        self.numerical_features_ = [
            col for col in self.columns_ if col not in self.discrete_features
        ]
        if not self.discrete_features and not self.numerical_features_:
            raise ValueError("Metric preprocessing requires at least one feature.")

        self.categorical_encoder_ = None
        if self.discrete_features:
            fit_frames = list(categorical_fit_data or [reference_data])
            for idx, frame in enumerate(fit_frames):
                self._validate_feature_columns(frame, f"categorical_fit_data[{idx}]")
            self.categorical_encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.categorical_handle_unknown,
            )
            self.categorical_encoder_.fit(
                pd.concat(
                    [frame[self.discrete_features] for frame in fit_frames],
                    axis=0,
                )
            )

        self.numerical_scaler_ = None
        if self.numerical_features_:
            self.numerical_scaler_ = MinMaxScaler()
            self.numerical_scaler_.fit(reference_data[self.numerical_features_])

        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        self._validate_is_fitted()
        self._validate_feature_columns(data, "data")

        parts = []
        if self.categorical_encoder_ is not None:
            categorical = self.categorical_encoder_.transform(
                data[self.discrete_features]
            )
            parts.append(categorical / 2)
        if self.numerical_scaler_ is not None:
            parts.append(
                self.numerical_scaler_.transform(data[self.numerical_features_])
            )
        return np.concatenate(parts, axis=1)

    def _validate_feature_columns(self, data: pd.DataFrame, data_name: str):
        missing = [col for col in self.discrete_features if col not in data.columns]
        if missing:
            raise ValueError(f"{data_name} is missing discrete columns: {missing}.")

        if hasattr(self, "columns_"):
            missing = [col for col in self.columns_ if col not in data.columns]
            if missing:
                raise ValueError(f"{data_name} is missing columns: {missing}.")

    def _validate_is_fitted(self):
        if not hasattr(self, "columns_"):
            raise ValueError("GowerLikePreprocessor must be fitted before transform.")


def bin_numerical_columns(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    columns: list,
    n_bins: int,
    skip_constant: bool = False,
    fit_data: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_binned = real.copy()
    syn_binned = syn.copy()
    fit_data = real if fit_data is None else fit_data

    if not columns or n_bins < 2:
        return real_binned, syn_binned

    if skip_constant:
        columns = [
            col for col in columns if fit_data[col].min() != fit_data[col].max()
        ]
        if not columns:
            return real_binned, syn_binned

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    discretizer.fit(fit_data[columns])
    real_binned[columns] = discretizer.transform(real[columns])
    syn_binned[columns] = discretizer.transform(syn[columns])
    return real_binned, syn_binned


def empirical_discrete_distribution(
    x: np.ndarray,
    y: np.ndarray,
    dropna: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    s1 = pd.Series(np.asarray(x))
    s2 = pd.Series(np.asarray(y))
    p = s1.value_counts(normalize=True, dropna=dropna)
    q = s2.value_counts(normalize=True, dropna=dropna)
    support = p.index.union(q.index)
    p = p.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    q = q.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    return p, q


def gower_like_transform(
    data: Mapping[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    discrete_features: Optional[list] = None,
    categorical_fit_data: Optional[Iterable[pd.DataFrame]] = None,
    categorical_handle_unknown: str = "error",
) -> dict[str, np.ndarray]:
    """Fit the internal metric preprocessor and transform named datasets."""
    transformer = GowerLikePreprocessor(
        discrete_features=discrete_features,
        categorical_handle_unknown=categorical_handle_unknown,
    ).fit(reference_data, categorical_fit_data=categorical_fit_data)
    return {name: transformer.transform(df) for name, df in data.items()}


def fast_gower_transform(
    data: Mapping[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    discrete_features: Optional[list] = None,
    categorical_fit_data: Optional[Iterable[pd.DataFrame]] = None,
) -> dict[str, pd.DataFrame]:
    """Min-max scale numericals and integer-code categoricals for FastGowerNN."""
    discrete_features = list(discrete_features or [])
    columns = reference_data.columns.tolist()
    numerical_features = [col for col in columns if col not in discrete_features]

    scaler = None
    if numerical_features:
        scaler = MinMaxScaler().fit(reference_data[numerical_features])

    encoder = None
    if discrete_features:
        fit_frames = list(categorical_fit_data or [reference_data])
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-2,
            encoded_missing_value=-1,
        ).fit(
            pd.concat([df[discrete_features] for df in fit_frames], axis=0)
        )

    transformed = {}
    for name, df in data.items():
        out = pd.DataFrame(index=df.index)
        if numerical_features:
            out[numerical_features] = scaler.transform(df[numerical_features])
        if discrete_features:
            categorical = encoder.transform(df[discrete_features]).astype(np.int32)
            categorical[df[discrete_features].isna().to_numpy()] = -1
            out[discrete_features] = categorical
        transformed[name] = out[columns]
    return transformed
