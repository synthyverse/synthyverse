"""Hyperparameter search spaces used by evaluation metrics."""


def xgboost_hyperparams(trial):
    """Optuna search space for XGBoost models."""
    return {
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 30, 150),
    }


def _classifier_hyperparams(trial, model_name_lc: str) -> dict:
    """Optuna search spaces for common sklearn classifiers."""
    if model_name_lc == "logisticregression":
        return {
            "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
        }
    if model_name_lc == "randomforestclassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
        }
    if model_name_lc == "extratreesclassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
        }
    if model_name_lc == "gradientboostingclassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
    if model_name_lc == "histgradientboostingclassifier":
        return {
            "max_iter": trial.suggest_int("max_iter", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
    if model_name_lc == "svc":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        }
    if model_name_lc == "linearsvc":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        }
    if model_name_lc == "kneighborsclassifier":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        }
    if model_name_lc == "mlpclassifier":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64)]
            ),
            "max_iter": trial.suggest_int("max_iter", 50, 500),
        }
    raise ValueError(
        f"No predefined tuning space for sklearn classifier '{model_name_lc}'. "
        "Please pass fixed model_params or extend hyperparameters.py."
    )


def _regressor_hyperparams(trial, model_name_lc: str) -> dict:
    """Optuna search spaces for common sklearn regressors."""
    if model_name_lc == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        }
    if model_name_lc == "lasso":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        }
    if model_name_lc == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if model_name_lc == "randomforestregressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
        }
    if model_name_lc == "extratreesregressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
        }
    if model_name_lc == "gradientboostingregressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
    if model_name_lc == "histgradientboostingregressor":
        return {
            "max_iter": trial.suggest_int("max_iter", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
    if model_name_lc == "svr":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        }
    if model_name_lc == "kneighborsregressor":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        }
    if model_name_lc == "mlpregressor":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64)]
            ),
            "max_iter": trial.suggest_int("max_iter", 50, 500),
        }
    raise ValueError(
        f"No predefined tuning space for sklearn regressor '{model_name_lc}'. "
        "Please pass fixed model_params or extend hyperparameters.py."
    )


def sklearn_hyperparams(trial, model_name: str, estimator_type: str) -> dict:
    """Get Optuna search space for a sklearn model.

    Args:
        trial: Optuna trial.
        model_name: sklearn estimator class name.
        estimator_type: one of {"classifier", "regressor"}.
    """
    model_name_lc = model_name.lower()
    if estimator_type == "classifier":
        return _classifier_hyperparams(trial, model_name_lc)
    if estimator_type == "regressor":
        return _regressor_hyperparams(trial, model_name_lc)
    raise ValueError(f"Unsupported estimator_type: {estimator_type}")
