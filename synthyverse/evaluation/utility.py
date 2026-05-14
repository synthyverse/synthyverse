import json
import os

import pandas as pd

from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    f1_score,
    accuracy_score,
)
from .ml import ml_task, HYPERPARAM_SAVE_DIR, tune_ml_model


class MLE:
    """Machine Learning Efficacy from configurable ML models.

    Measures how well synthetic data can be used for downstream machine learning
    tasks compared to real data.

    Args:
        target_column (str): Name of the target column. Default: "target".
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        random_state (int): Random seed for reproducibility. Default: 0.
        train_set (str): Which dataset to train on ("synthetic" for TSTR, "real" for TRTS). Evaluates on the opposite set. Default: "synthetic".
        model_name (str): Estimator family. Supported values include "xgboost",
            "randomforest", "decisiontree", "linearregression", and "svm",
            including common aliases. Every model except for XGBoost is a scikit-learn model. Default: "xgboost".
        model_params (dict): Model parameters passed to the selected estimator.
        tune (bool): Whether to tune hyperparameters. Default: False.
        tuning_trials (int): Number of Optuna trials for hyperparameter tuning. Default: 32.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import MLE
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> X_val = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = MLE(
        ...     target_column="target",
        ...     discrete_features=discrete_features,
        ...     train_set="synthetic",
        ...     tune=True,
        ...     random_state=42
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_test, X_syn, X_val=X_val)
    """

    name = "mle"

    def __init__(
        self,
        target_column: str = "target",
        discrete_features: list = None,
        random_state: int = 0,
        train_set: str = "synthetic",  # whether to compute TSTR or TRTS
        model_name: str = "xgboost",
        model_params: dict = None,
        tune: bool = False,
        tuning_trials: int = 32,
    ):
        super().__init__()
        self.random_state = random_state
        self.tune = tune
        self.tuning_trials = tuning_trials
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )
        self.target_column = target_column
        self.train_set = train_set
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_syn_test: pd.DataFrame = None,
    ):
        """Evaluate synthetic data utility using machine learning efficacy.

        Args:
            X_train: Real training data as a pandas DataFrame.
            X_test: Real test data as a pandas DataFrame.
            X_syn: Synthetic training data as a pandas DataFrame.
            X_val: Optional validation data used when tune=True.
            X_syn_test: Optional synthetic test data used when train_set="real".

        Returns:
            dict: Dictionary with metric scores for the configured train/test
                direction. Keys have the form
                "mle.train_<train_set>_test_<test_set>.<score>".
        """

        assert not (
            X_syn_test is None and self.train_set == "real"
        ), "X_syn_test must be provided when train_set=real, since we need to evaluate on a synthetic test set."

        self.task = (
            "regression"
            if self.target_column not in self.discrete_features
            else "binary"
        )
        if self.task != "regression" and X_train[self.target_column].nunique() > 2:
            self.task = "multiclass"

        x_train = X_syn.copy() if self.train_set == "synthetic" else X_train.copy()
        x_train = x_train.drop(columns=[self.target_column])
        y_train = (
            X_syn[self.target_column].copy()
            if self.train_set == "synthetic"
            else X_train[self.target_column].copy()
        )

        if self.tune:
            # load tuned params from file if it exists
            model_slug = self.model_name.replace(" ", "_")
            param_file = f"{HYPERPARAM_SAVE_DIR}/mle_{model_slug}.json"
            if os.path.exists(param_file):
                with open(param_file, "r") as f:
                    params = json.load(f)
            else:
                # tune params and save to file
                assert X_val is not None, "X_val must be provided when tune=True."
                x_val = (
                    X_syn.iloc[-len(X_val) :].copy()
                    if self.train_set == "real"
                    else X_val.copy()
                )
                x_val = x_val.drop(columns=[self.target_column])
                y_val = (
                    X_syn.iloc[-len(X_val) :][self.target_column].copy()
                    if self.train_set == "real"
                    else X_val[self.target_column].copy()
                )

                params = tune_ml_model(
                    x_train,
                    x_val,
                    y_train,
                    y_val,
                    self.discrete_features,
                    self.task,
                    self.model_name,
                    self.tuning_trials,
                    self.random_state,
                )
                os.makedirs(os.path.dirname(param_file), exist_ok=True)
                with open(param_file, "w") as f:
                    json.dump(params, f)

        else:
            params = self.model_params

        synthetic_test = X_syn_test if X_syn_test is not None else X_syn
        x_test = synthetic_test.copy() if self.train_set == "real" else X_test.copy()
        x_test = x_test.drop(columns=[self.target_column])
        y_test = (
            synthetic_test[self.target_column].copy()
            if self.train_set == "real"
            else X_test[self.target_column].copy()
        )

        score_fns = (
            [r2_score, root_mean_squared_error]
            if self.task == "regression"
            else [roc_auc_score, f1_score, accuracy_score]
        )
        scores = ml_task(
            x_train,
            x_test,
            y_train,
            y_test,
            self.discrete_features,
            self.task,
            self.model_name,
            params,
            random_state=self.random_state,
            score_fns=score_fns,
        )
        result = {}
        for key, value in scores.items():
            result[f"{self.name}.{key}"] = value
        return result
