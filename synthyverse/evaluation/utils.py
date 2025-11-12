from sklearn.metrics import (
    roc_auc_score,
    root_mean_squared_error,
    r2_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import optuna


def precision_recall(
    y_true,
    y_pred,
):
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred)


def get_accuracy_metric(name: str):
    # capture as many variants of the metric name as possible
    if name.lower().startswith("precision") and name.lower().endswith("recall"):
        return precision_recall, "precision-recall"
    elif name.lower().startswith("f1"):
        return f1_score, "f1"
    elif name.lower().startswith("acc"):
        return accuracy_score, "accuracy"
    elif (
        name.lower().startswith("r2")
        or name.lower().startswith("r^2")
        or name.lower().startswith("rsquared")
    ):
        return r2_score, "r2"
    elif name.lower().startswith("rmse") or (
        name.lower().startswith("root") and name.lower().endswith("error")
    ):
        return root_mean_squared_error, "rmse"
    elif name.lower().startswith("roc") and name.lower().endswith("auc"):
        return (
            lambda y_true, y_score: roc_auc_score(
                y_true, y_score, average="micro", multi_class="ovr"
            ),
            "roc_auc",
        )
    else:
        raise ValueError(f"Metric {name} not supported")


def xgboost_hyperparams(trial):
    return {
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 30, 150),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 20.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
