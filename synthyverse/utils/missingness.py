from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


ALLOWED_MISSING_IMPUTATION_METHODS = {
    "drop",
    "keep",
    "random",
    "mean",
    "median",
    "most_frequent",
    "missforest",
}


def validate_missing_imputation_method(imputation_method: str) -> None:
    if imputation_method not in ALLOWED_MISSING_IMPUTATION_METHODS:
        raise ValueError(f"Unknown missing imputation method: {imputation_method}")


def drop_rows_with_missing_numericals(
    X: pd.DataFrame, numerical_features: List[str]
) -> pd.DataFrame:
    if len(numerical_features) == 0:
        return X
    return X.dropna(subset=numerical_features)


def fit_random_imputation_samples(
    X: pd.DataFrame, numerical_features: List[str]
) -> Dict[str, np.ndarray]:
    imputation_samples: Dict[str, np.ndarray] = {}
    for col in numerical_features:
        observed = X[col].dropna().values
        if observed.size == 0:
            observed = np.array([0.0])
        imputation_samples[col] = observed
    return imputation_samples


def apply_random_imputation(
    X: pd.DataFrame,
    numerical_features: List[str],
    imputation_samples: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    X_new = X.copy()
    for col in numerical_features:
        miss_mask = X_new[col].isna()
        if miss_mask.any():
            X_new.loc[miss_mask, col] = rng.choice(
                imputation_samples[col],
                size=miss_mask.sum(),
                replace=True,
            )
    return X_new


def fit_numeric_imputer(
    X: pd.DataFrame,
    numerical_features: List[str],
    imputation_method: str,
    random_state: int,
) -> Tuple[Union[SimpleImputer, IterativeImputer], List[str]]:
    if imputation_method in {"mean", "median", "most_frequent"}:
        imputer: Union[SimpleImputer, IterativeImputer] = SimpleImputer(
            strategy=imputation_method,
            keep_empty_features=True,
        )
    elif imputation_method == "missforest":
        estimator = RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            random_state=random_state,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            random_state=random_state,
            tol=1e-3,
            max_iter=10,
        )
    else:
        raise ValueError(
            f"fit_numeric_imputer only supports statistical/iterative methods, got: {imputation_method}"
        )

    imputer.fit(X[numerical_features])
    all_missing_numeric_columns = (
        [col for col in numerical_features if X[col].isna().all()]
        if imputation_method in {"mean", "median"}
        else []
    )
    return imputer, all_missing_numeric_columns


def apply_fitted_numeric_imputer(
    X: pd.DataFrame,
    numerical_features: List[str],
    imputer: Union[SimpleImputer, IterativeImputer],
    imputation_method: str,
    all_missing_numeric_columns: List[str],
) -> pd.DataFrame:
    X_new = X.copy()
    miss_mask = X_new[numerical_features].isna()
    imputed_values = imputer.transform(X_new[numerical_features])
    imputed_df = pd.DataFrame(
        imputed_values,
        columns=numerical_features,
        index=X_new.index,
    )

    if imputation_method in {"mean", "median"}:
        for col in all_missing_numeric_columns:
            if col in imputed_df.columns:
                imputed_df.loc[miss_mask[col], col] = np.nan

    X_new.loc[:, numerical_features] = imputed_df[numerical_features]
    return X_new


def fit_transform_numeric_imputer(
    X: pd.DataFrame,
    numerical_features: List[str],
    imputation_method: str,
    random_state: int,
) -> Tuple[pd.DataFrame, Union[SimpleImputer, IterativeImputer], List[str]]:
    imputer, all_missing_numeric_columns = fit_numeric_imputer(
        X=X,
        numerical_features=numerical_features,
        imputation_method=imputation_method,
        random_state=random_state,
    )
    X_imputed = apply_fitted_numeric_imputer(
        X=X,
        numerical_features=numerical_features,
        imputer=imputer,
        imputation_method=imputation_method,
        all_missing_numeric_columns=all_missing_numeric_columns,
    )
    return X_imputed, imputer, all_missing_numeric_columns
