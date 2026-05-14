import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
import optuna
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    r2_score,
    root_mean_squared_error,
)

from sklearn.ensemble import RandomForestRegressor

HYPERPARAM_SAVE_DIR = "synthyverse_hyperparams_tuned"


def tune_ml_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    discrete_features: list,
    task: str = "binary",
    model_name: str = "xgboost",
    tuning_trials: int = 32,
    random_state: int = 0,
    score_fns: list = None,
):

    model_name = resolve_model_name(model_name)
    tuning_score_fns = (
        get_default_tuning_score_fns(task) if score_fns is None else score_fns
    )

    def objective(trial: optuna.Trial):
        params = get_hyperparams(model_name, task, trial)
        scores = ml_task(
            X_train,
            X_val,
            y_train,
            y_val,
            discrete_features,
            task,
            model_name,
            params,
            random_state=random_state,
            score_fns=tuning_score_fns,
        )
        return next(iter(scores.values()))

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=random_state),
        direction="maximize",
    )
    study.optimize(
        objective,
        n_trials=tuning_trials,
        show_progress_bar=True,
    )
    return study.best_params.copy()


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
):
    numerical_features = [x for x in X_train.columns if x not in discrete_features]
    categorical_features = [x for x in X_train.columns if x in discrete_features]

    # reorder to align with columntransformer
    X_train = X_train[categorical_features + numerical_features]
    X_test = X_test[categorical_features + numerical_features]

    # resolve model name
    model_name = resolve_model_name(model_name)

    # xgboost requires specifying which features are categorical from the data
    model_params = {} if model_params is None else model_params.copy()
    if model_name == "xgboost":
        model_params.setdefault(
            "feature_types",
            ["c" if x in categorical_features else "q" for x in X_train.columns],
        )

    cat_encoder, num_scaler = get_preprocessors(model_name)
    transformer = ColumnTransformer(
        transformers=[
            ("cat", cat_encoder, categorical_features),
            ("num", num_scaler, numerical_features),
        ],
        remainder="drop",
    )
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    if task == "regression":
        target_preprocessor = StandardScaler()  # required for interpretable error
        y_train_transformed = target_preprocessor.fit_transform(y_train.to_frame())
        y_test_transformed = target_preprocessor.transform(y_test.to_frame())
        y_train_transformed = y_train_transformed.ravel()
        y_test_transformed = y_test_transformed.ravel()
    else:
        target_preprocessor = LabelEncoder()  # required for contiguous encoding
        target_preprocessor.fit(pd.concat([y_train, y_test], ignore_index=True))
        y_train_transformed = target_preprocessor.transform(y_train)
        y_test_transformed = target_preprocessor.transform(y_test)

    # build model
    model = build_ml_model(model_name, task, model_params, random_state=random_state)

    # train model
    model.fit(X_train_transformed, y_train_transformed)

    if task == "multiclass":
        preds = model.predict_proba(X_test_transformed)
        pred_labels = model.predict(X_test_transformed)
    elif task == "binary":
        preds = model.predict_proba(X_test_transformed)[:, 1]
        pred_labels = model.predict(X_test_transformed)
    elif task == "regression":
        preds = model.predict(X_test_transformed)
        pred_labels = preds
    else:
        raise ValueError(f"Task {task} not supported")

    if score_fns is not None:
        return score_ml_predictions(
            y_test_transformed,
            preds,
            pred_labels,
            score_fns,
            task,
        )

    return preds


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
            if task == "multiclass":
                value = fn(y_true, preds, average="micro", multi_class="ovr")
            else:
                value = fn(y_true, preds)
        elif key == "f1":
            if task == "multiclass":
                value = fn(y_true, pred_labels, average="weighted")
            else:
                value = fn(y_true, pred_labels)
        elif key == "accuracy":
            value = fn(y_true, pred_labels)
        else:
            value = fn(y_true, preds)

        scores[key] = float(value) if np.isscalar(value) else value
    return scores


def resolve_score_fn(score_fn):
    if isinstance(score_fn, str):
        score_fn = score_fn.lower()
        score_fns = {
            "auc": roc_auc_score,
            "roc_auc": roc_auc_score,
            "roc_auc_score": roc_auc_score,
            "f1": f1_score,
            "f1_score": f1_score,
            "accuracy": accuracy_score,
            "accuracy_score": accuracy_score,
            "r2": r2_score,
            "r2_score": r2_score,
            "rmse": root_mean_squared_error,
            "root_mean_squared_error": root_mean_squared_error,
        }
        if score_fn not in score_fns:
            raise ValueError(f"Score function {score_fn} not supported")
        return score_fn, score_fns[score_fn]

    if isinstance(score_fn, tuple):
        if len(score_fn) != 2:
            raise ValueError("Score function tuples must be (name, callable)")
        name, fn = score_fn
        return str(name), fn

    return getattr(score_fn, "__name__", score_fn.__class__.__name__), score_fn


def score_key(score_name: str):
    score_name = score_name.lower()
    aliases = {
        "roc_auc": "auc",
        "roc_auc_score": "auc",
        "f1_score": "f1",
        "accuracy_score": "accuracy",
        "r2_score": "r2",
        "root_mean_squared_error": "rmse",
    }
    return aliases.get(score_name, score_name)


def get_default_tuning_score_fns(task: str):
    if task == "regression":
        return [r2_score]
    elif task in ["binary", "multiclass"]:
        return [roc_auc_score]
    else:
        raise ValueError(f"Task {task} not supported")


def score_ml_model(model, X_test, y_test, task: str = "binary"):
    if task == "multiclass":
        preds = model.predict_proba(X_test)
        pred_labels = model.predict(X_test)
        return score_ml_predictions(
            y_test,
            preds,
            pred_labels,
            [roc_auc_score, f1_score, accuracy_score],
            task,
        )
    if task == "binary":
        preds = model.predict_proba(X_test)[:, 1]
        pred_labels = model.predict(X_test)
        return score_ml_predictions(
            y_test,
            preds,
            pred_labels,
            [roc_auc_score, f1_score, accuracy_score],
            task,
        )
    elif task == "regression":
        preds = model.predict(X_test)
        return score_ml_predictions(
            y_test,
            preds,
            preds,
            [r2_score, root_mean_squared_error],
            task,
        )
    else:
        raise ValueError(f"Task {task} not supported")


def resolve_model_name(model_name: str):
    model_name = model_name.lower()
    for r in [" ", "-", "_", ".", ","]:
        model_name = model_name.replace(r, "")

    if model_name == "rf" or model_name.startswith("randomforest"):
        return "randomforest"
    elif model_name == "dt" or model_name.startswith("decisiontree"):
        return "decisiontree"
    elif model_name in [
        "lr",
        "linearregression",
        "logisticregression",
        "ridge",
        "lasso",
        "elasticnet",
    ]:
        return "linearregression"
    elif model_name in [
        "svm",
        "svc",
        "svr",
    ] or model_name.startswith("supportvector"):
        return "svm"
    elif model_name.startswith("xgb"):
        return "xgboost"
    else:
        raise ValueError(f"Model {model_name} not supported")


def build_ml_model(
    model_name: str,
    task: str = "binary",
    model_params: dict = None,
    random_state: int = 0,
):

    if model_name == "xgboost":
        model = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        default_params = {
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
        default_params = {
            "random_state": random_state,
        }
    elif model_name == "decisiontree":
        model = (
            DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
        )
        default_params = {
            "random_state": random_state,
        }
    elif model_name == "linearregression":
        model = ElasticNet if task == "regression" else LogisticRegression
        default_params = {
            "random_state": random_state,
        }
    elif model_name == "svm":
        if task == "regression":
            model = SVR
            default_params = {}
        else:
            model = SVC
            default_params = {
                "random_state": random_state,
                "probability": True,
            }
    else:
        raise ValueError(f"Model {model_name} not supported")

    model_params = {} if model_params is None else model_params.copy()

    default_params.update(model_params)
    return model(**default_params)


def get_preprocessors(model_name: str):
    if model_name in ["linearregression", "svm"]:
        num_scaler = StandardScaler()
        cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    else:
        # tree-based methods don't need scaling and can use ordinal encoding
        num_scaler = "passthrough"
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    return cat_encoder, num_scaler


def get_hyperparams(model_name: str, task: str, trial: optuna.Trial):
    if model_name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.35, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.1, 10.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
    elif model_name == "randomforest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 4, 8, 16, 32, 64]
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 16, log=True
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16, log=True),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.5, 0.75, 1.0]
            ),
        }
        return params
    elif model_name == "decisiontree":
        return {
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 4, 8, 16, 32, 64]
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 16, log=True
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16, log=True),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", 0.5, 0.75, 1.0]
            ),
        }
    elif model_name == "linearregression":
        if task == "regression":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
        else:
            params = {
                "C": trial.suggest_float("C", 1e-4, 1000.0, log=True),
            }
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
            return params
    elif model_name == "svm":
        params = {
            "C": trial.suggest_float("C", 1e-4, 1000.0, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "rbf", "poly", "sigmoid"]
            ),
        }
        if params["kernel"] in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 10.0, log=True)
        if params["kernel"] == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        if params["kernel"] in ["poly", "sigmoid"]:
            params["coef0"] = trial.suggest_float("coef0", -1.0, 1.0)
        return params
    else:
        raise ValueError(f"Model {model_name} not supported")
