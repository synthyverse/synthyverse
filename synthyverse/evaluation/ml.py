import json
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SCORE_FNS = {
    "auc": roc_auc_score,
    "f1": f1_score,
    "accuracy": accuracy_score,
    "r2": r2_score,
    "rmse": root_mean_squared_error,
}

SCORE_ALIASES = {
    "roc_auc": "auc",
    "roc_auc_score": "auc",
    "f1_score": "f1",
    "accuracy_score": "accuracy",
    "r2_score": "r2",
    "root_mean_squared_error": "rmse",
}

XGBOOST_GPU_MIN_ELEMENTS = 1_000_000


def split_validation(X, y, val_size: float, random_state: int, stratify: bool = True):
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1 when validation is needed.")
    if stratify:
        counts = pd.Series(y).value_counts()
        if counts.min() < 2:
            keep = pd.Series(y).isin(counts[counts > 1].index).to_numpy()
            X = X.iloc[keep]
            y = y.iloc[keep]
            counts = pd.Series(y).value_counts()
        n_classes = len(counts)
        n_val = int(np.ceil(len(y) * val_size))
        if n_val < n_classes or len(y) - n_val < n_classes:
            raise ValueError("val_size is too small for stratified validation.")
    return train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def order_features(X: pd.DataFrame = None, discrete_features: list = None):
    discrete_features = discrete_features or []
    if X is None:
        return X
    numerical_features = [col for col in X.columns if col not in discrete_features]
    return X[numerical_features + discrete_features]


def build_ml_preprocessor(
    model_name: str, categorical_features: list, numerical_features: list
):
    cat_encoder, num_scaler = get_preprocessors(model_name)
    return ColumnTransformer(
        transformers=[
            ("cat", cat_encoder, categorical_features),
            ("num", num_scaler, numerical_features),
        ],
        remainder="drop",
    )


def cv_ml_task(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_features: list,
    task: str = "binary",
    model_name: str = "xgboost",
    model_params: dict = None,
    random_state: int = 0,
    score_fn="auc",
    nfold: int = 3,
    n_jobs: int = -1,
):
    X = X.copy().reset_index(drop=True)
    y = y.copy().reset_index(drop=True)
    numerical_features = [col for col in X.columns if col not in discrete_features]
    categorical_features = [col for col in X.columns if col in discrete_features]
    X = order_features(X, categorical_features)
    if task != "regression":
        y = pd.Series(LabelEncoder().fit_transform(y))
        min_class_count = int(y.value_counts().min())
        if min_class_count < 2:
            raise ValueError(
                "Stratified cross-validation requires at least two rows in each class."
            )
        nfold = min(nfold, min_class_count)

    model_name = resolve_model_name(model_name)
    model_params = {} if model_params is None else model_params.copy()
    key = score_key(resolve_score_fn(score_fn)[0])

    if model_name == "xgboost":
        return {
            key: _xgboost_cv_score(
                X,
                y,
                categorical_features,
                task,
                key,
                model_params,
                random_state,
                nfold,
            )
        }

    model = Pipeline(
        [
            (
                "preprocess",
                build_ml_preprocessor(
                    model_name, categorical_features, numerical_features
                ),
            ),
            ("model", build_ml_model(model_name, task, model_params, random_state)),
        ]
    )
    cv = (
        StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_state)
        if task != "regression"
        else KFold(n_splits=nfold, shuffle=True, random_state=random_state)
    )
    score = cross_val_score(
        model, X, y, cv=cv, scoring=_cv_scoring(key, task), n_jobs=n_jobs
    ).mean()
    return {key: float(-score if key == "rmse" else score)}


def ml_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    discrete_features: list,
    task: str = "binary",
    model_name: str = "xgboost",
    model_params: dict = None,
    random_state: int = 0,
    score_fns: list = None,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
):
    X_train = X_train.copy().reset_index(drop=True)
    X_test = X_test.copy().reset_index(drop=True)
    X_val = X_val.copy().reset_index(drop=True) if X_val is not None else None
    y_train = y_train.copy().reset_index(drop=True)
    y_test = y_test.copy().reset_index(drop=True)
    y_val = y_val.copy().reset_index(drop=True) if y_val is not None else None

    numerical_features = [
        col for col in X_train.columns if col not in discrete_features
    ]
    categorical_features = [col for col in X_train.columns if col in discrete_features]

    X_train = order_features(X_train, categorical_features)
    X_test = order_features(X_test, categorical_features)
    X_val = order_features(X_val, categorical_features)
    model_name = resolve_model_name(model_name)

    has_validation_data = (X_val is not None) and (y_val is not None)

    model_params = {} if model_params is None else model_params.copy()

    if task != "regression" and y_train.nunique() == 1:
        return _constant_classification_result(y_train, y_test, score_fns)

    if (
        _uses_xgboost_early_stopping(model_name, model_params)
        and not has_validation_data
    ):
        raise ValueError(
            "X_val and y_val must be provided when using XGBoost early stopping."
        )

    if model_name == "xgboost":
        model_params.setdefault(
            "feature_types",
            ["c"] * len(categorical_features) + ["q"] * len(numerical_features),
        )

    target_info = _fit_target(y_train, y_test, y_val, task)
    if task != "regression" and has_validation_data:
        X_val = X_val.loc[target_info["val_mask"]]
        y_val = y_val.loc[target_info["val_mask"]]

    X_train, X_test, X_val = _transform_features(
        model_name, categorical_features, numerical_features, X_train, X_test, X_val
    )
    y_train, y_test, y_val = _transform_target(
        y_train, y_test, y_val, task, target_info
    )

    if model_name == "xgboost":
        _maybe_enable_xgboost_gpu(model_params, *X_train.shape)
    model = build_ml_model(model_name, task, model_params, random_state=random_state)
    model.fit(
        X_train,
        y_train,
        **_fit_kwargs(model_name, model_params, X_val, y_val),
    )

    preds, pred_labels = _predict(model, X_test, y_test, task, target_info)
    if score_fns is None:
        return preds
    return score_ml_predictions(
        y_test,
        preds,
        pred_labels,
        score_fns,
        target_info["score_task"] if task != "regression" else task,
    )


def score_ml_predictions(
    y_true,
    preds,
    pred_labels,
    score_fns: list,
    task: str = "binary",
):
    scores = {}
    for score_fn in score_fns:
        name, fn = resolve_score_fn(score_fn)
        key = score_key(name)

        if key == "auc":
            y_true_series = pd.Series(y_true)
            if y_true_series.nunique() < 2:
                warnings.warn(
                    "AUC is undefined when labels are constant; returning NaN.",
                    UserWarning,
                )
                scores[key] = float(np.nan)
                continue
            if task == "multiclass":
                labels = np.arange(preds.shape[1])
                missing = labels[~np.isin(labels, y_true_series)]
                if len(missing):
                    warnings.warn(
                        "Multiclass AUC is undefined when y_true is missing "
                        f"classes {missing.tolist()}; returning NaN.",
                        UserWarning,
                    )
                    scores[key] = float(np.nan)
                    continue
            value = (
                fn(
                    y_true,
                    preds,
                    average="macro",
                    multi_class="ovr",
                    labels=np.arange(preds.shape[1]),
                )
                if task == "multiclass"
                else fn(y_true, preds)
            )
        elif key == "f1":
            value = (
                fn(y_true, pred_labels, average="weighted")
                if task == "multiclass"
                else fn(y_true, pred_labels)
            )
        elif key == "accuracy":
            value = fn(y_true, pred_labels)
        else:
            value = fn(y_true, preds)

        scores[key] = float(value) if np.isscalar(value) else value
    return scores


def resolve_score_fn(score_fn):
    if isinstance(score_fn, str):
        key = score_key(score_fn.lower())
        if key not in SCORE_FNS:
            raise ValueError(f"Score function {score_fn} not supported")
        return score_fn, SCORE_FNS[key]

    if isinstance(score_fn, tuple):
        if len(score_fn) != 2:
            raise ValueError("Score function tuples must be (name, callable)")
        name, fn = score_fn
        return str(name), fn

    return getattr(score_fn, "__name__", score_fn.__class__.__name__), score_fn


def score_key(score_name: str):
    return SCORE_ALIASES.get(score_name.lower(), score_name.lower())


def _constant_classification_result(y_train, y_test, score_fns):
    encoder = LabelEncoder().fit(pd.concat([y_train, y_test], ignore_index=True))
    y_test = encoder.transform(y_test)
    constant_label = encoder.transform([y_train.iloc[0]])[0]
    pred_labels = np.full(len(y_test), constant_label)
    score_task = "multiclass" if len(encoder.classes_) > 2 else "binary"
    if score_task == "multiclass":
        preds = np.zeros((len(y_test), len(encoder.classes_)))
        preds[:, constant_label] = 1.0
    else:
        preds = pred_labels.astype(float)
    return (
        score_ml_predictions(y_test, preds, pred_labels, score_fns, score_task)
        if score_fns is not None
        else preds
    )


def _uses_xgboost_early_stopping(model_name, model_params):
    return (
        model_name == "xgboost"
        and model_params.get("early_stopping_rounds") is not None
    )


def _fit_target(y_train, y_test, y_val, task):
    if task == "regression":
        return {"encoder": StandardScaler(), "score_task": task}

    model_encoder = LabelEncoder().fit(y_train)
    score_encoder = LabelEncoder().fit(pd.concat([y_train, y_test], ignore_index=True))
    return {
        "model_encoder": model_encoder,
        "score_encoder": score_encoder,
        "class_map": score_encoder.transform(model_encoder.classes_),
        "score_task": "multiclass" if len(score_encoder.classes_) > 2 else "binary",
        "val_mask": None if y_val is None else y_val.isin(model_encoder.classes_),
    }


def _transform_features(
    model_name, categorical_features, numerical_features, X_train, X_test, X_val
):
    transformer = build_ml_preprocessor(
        model_name, categorical_features, numerical_features
    )
    return (
        transformer.fit_transform(X_train),
        transformer.transform(X_test),
        None if X_val is None else transformer.transform(X_val),
    )


def _transform_target(y_train, y_test, y_val, task, target):
    if task == "regression":
        encoder = target["encoder"]
        return (
            encoder.fit_transform(y_train.to_frame()).ravel(),
            encoder.transform(y_test.to_frame()).ravel(),
            None if y_val is None else encoder.transform(y_val.to_frame()).ravel(),
        )

    return (
        target["model_encoder"].transform(y_train),
        target["score_encoder"].transform(y_test),
        None if y_val is None else target["model_encoder"].transform(y_val),
    )


def _fit_kwargs(model_name, model_params, X_val, y_val):
    if _uses_xgboost_early_stopping(model_name, model_params):
        return {"eval_set": [(X_val, y_val)]}
    return {}


def _predict(model, X_test, y_test, task, target):
    if task == "regression":
        preds = model.predict(X_test)
        return preds, preds

    raw_preds = model.predict_proba(X_test)
    model_classes = np.asarray(
        getattr(model, "classes_", np.arange(raw_preds.shape[1])), dtype=int
    )
    class_map = target["class_map"]
    pred_labels = class_map[np.asarray(model.predict(X_test), dtype=int)]

    if task == "multiclass" or target["score_task"] == "multiclass":
        preds = np.zeros((len(y_test), len(target["score_encoder"].classes_)))
        preds[:, class_map[model_classes]] = raw_preds
    else:
        preds = raw_preds[:, np.where(class_map[model_classes] == 1)[0][0]]
    return preds, pred_labels


def _xgboost_cv_score(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list,
    task: str,
    score_key_: str,
    model_params: dict,
    random_state: int,
    nfold: int,
):
    if categorical_features:
        X = X.copy()
        X[categorical_features] = OrdinalEncoder().fit_transform(
            X[categorical_features]
        )

    params = model_params.copy()
    num_boost_round = params.pop("num_boost_round", params.pop("n_estimators", 100))
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
    params.update(
        {
            "objective": (
                "binary:logistic"
                if task == "binary"
                else ("multi:softprob" if task == "multiclass" else "reg:squarederror")
            ),
            "seed": random_state,
            "tree_method": "hist",
        }
    )
    _maybe_enable_xgboost_gpu(params, len(X), X.shape[1])
    if task == "multiclass":
        params.setdefault("num_class", y.nunique())

    metric, maximize = _xgboost_cv_metric(score_key_)
    results = xgb.cv(
        params=params,
        dtrain=xgb.DMatrix(
            X,
            label=y,
            feature_types=[
                "c" if x in categorical_features else "q" for x in X.columns
            ],
        ),
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=task != "regression",
        early_stopping_rounds=early_stopping_rounds,
        seed=random_state,
        shuffle=True,
        metrics=metric,
        maximize=maximize,
        as_pandas=True,
    )
    return float(results[f"test-{metric}-mean"].iloc[-1])


def _xgboost_cv_metric(score_key_: str):
    if score_key_ == "auc":
        return "auc", True
    if score_key_ == "rmse":
        return "rmse", False
    raise ValueError(f"Score function {score_key_} is not supported for XGBoost CV")


def _maybe_enable_xgboost_gpu(params, n_rows, n_features):
    if "device" not in params and n_rows * n_features > XGBOOST_GPU_MIN_ELEMENTS:
        if _xgboost_can_use_gpu():
            params["device"] = "cuda"


@lru_cache(maxsize=1)
def _xgboost_can_use_gpu():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            booster = xgb.train(
                {
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "device": "cuda",
                    "verbosity": 0,
                },
                xgb.DMatrix(X, label=y),
                num_boost_round=1,
            )
        except Exception:
            return False

    messages = " ".join(str(w.message).lower() for w in caught)
    if (
        "no visible gpu" in messages
        or "not compiled with cuda" in messages
        or "setting device to cpu" in messages
    ):
        return False

    config = json.loads(booster.save_config())
    device = config["learner"]["generic_param"]["device"]
    return device.startswith("cuda")


def _cv_scoring(score_key_: str, task: str):
    scoring = {
        "auc": "roc_auc_ovr" if task == "multiclass" else "roc_auc",
        "f1": "f1_weighted" if task == "multiclass" else "f1",
        "accuracy": "accuracy",
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
    }
    if score_key_ not in scoring:
        raise ValueError(f"Score function {score_key_} not supported for CV")
    return scoring[score_key_]


def resolve_model_name(model_name: str):
    model_name = model_name.lower()
    for r in [" ", "-", "_", ".", ","]:
        model_name = model_name.replace(r, "")

    if model_name == "rf" or model_name.startswith("randomforest"):
        return "randomforest"
    if model_name == "dt" or model_name.startswith("decisiontree"):
        return "decisiontree"
    if model_name in [
        "lr",
        "linearregression",
        "logisticregression",
        "ridge",
        "lasso",
        "elasticnet",
    ]:
        return "linearregression"
    if model_name in ["svm", "svc", "svr"] or model_name.startswith("supportvector"):
        return "svm"
    if model_name.startswith("xgb"):
        return "xgboost"
    raise ValueError(f"Model {model_name} not supported")


def build_ml_model(
    model_name: str,
    task: str = "binary",
    model_params: dict = None,
    random_state: int = 0,
):
    model_params = {} if model_params is None else model_params.copy()

    if model_name == "xgboost":
        model = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        defaults = {
            "random_state": random_state,
            "objective": (
                "binary:logistic"
                if task == "binary"
                else ("multi:softprob" if task == "multiclass" else "reg:squarederror")
            ),
            "tree_method": "hist",
        }
    elif model_name == "randomforest":
        model = (
            RandomForestRegressor if task == "regression" else RandomForestClassifier
        )
        defaults = {"random_state": random_state}
    elif model_name == "decisiontree":
        model = (
            DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
        )
        defaults = {"random_state": random_state}
    elif model_name == "linearregression":
        model = ElasticNet if task == "regression" else LogisticRegression
        defaults = {"random_state": random_state}
    elif model_name == "svm":
        model = SVR if task == "regression" else SVC
        defaults = {} if task == "regression" else {"random_state": random_state}
        if task != "regression":
            defaults["probability"] = True
    else:
        raise ValueError(f"Model {model_name} not supported")

    defaults.update(model_params)
    return model(**defaults)


def get_preprocessors(model_name: str):
    if model_name in ["linearregression", "svm"]:
        return (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            StandardScaler(),
        )

    return (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "passthrough",
    )
