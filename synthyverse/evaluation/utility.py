import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from .ml import (
    HYPERPARAM_SAVE_DIR,
    ml_task,
    resolve_model_name,
    score_ml_predictions,
    split_validation,
    tune_ml_model,
)


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
            For XGBoost, passing ``early_stopping_rounds`` enables early stopping
            and requires ``val_size > 0``.
        tune (bool): Whether to tune hyperparameters on real train/validation
            data. Hyperparameter tuning is skipped when XGBoost early stopping
            is enabled. Default: False.
        tuning_trials (int): Number of Optuna trials for hyperparameter tuning. Default: 32.
        val_size (float): Fraction of real and synthetic training rows reserved
            for validation when tuning or early stopping needs it. Default: 0.2.
        hyperparam_save_dir (str): Directory used to cache tuned hyperparameters.
            Default: ``HYPERPARAM_SAVE_DIR``.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import MLE
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
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
        >>> results = metric.evaluate(X_train, X_test, X_syn)
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
        val_size: float = 0.2,
        hyperparam_save_dir: str = HYPERPARAM_SAVE_DIR,
    ):
        super().__init__()
        self.random_state = random_state
        self.tune = tune
        self.tuning_trials = tuning_trials
        self.val_size = val_size
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )
        self.target_column = target_column
        self.train_set = train_set
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.hyperparam_save_dir = os.fspath(hyperparam_save_dir)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ):
        """Evaluate synthetic data utility using machine learning efficacy.

        Args:
            X_train: Real training data as a pandas DataFrame.
            X_test: Real test data as a pandas DataFrame.
            X_syn: Synthetic training data as a pandas DataFrame.

        Returns:
            dict: Dictionary with metric scores for the configured train/test
                direction plus a real-train/real-test baseline. Baseline keys
                have the form "mle.baseline.<score>".
        """

        self.task = (
            "regression"
            if self.target_column not in self.discrete_features
            else "binary"
        )
        if self.task != "regression" and X_train[self.target_column].nunique() > 2:
            self.task = "multiclass"

        model_name = resolve_model_name(self.model_name)
        uses_xgboost_early_stopping = (
            model_name == "xgboost"
            and self.model_params.get("early_stopping_rounds") is not None
        )
        model_slug = self.model_name.replace(" ", "_")
        param_file = os.path.join(self.hyperparam_save_dir, f"mle_{model_slug}.json")
        needs_val = uses_xgboost_early_stopping or (
            self.tune
            and not uses_xgboost_early_stopping
            and not os.path.exists(param_file)
        )

        X_train_fit = X_train.copy()
        X_syn_fit = X_syn.copy()
        X_val = None
        X_syn_val = None
        constant_label = None
        if (
            self.task != "regression"
            and self.train_set == "synthetic"
            and X_syn[self.target_column].nunique() == 1
        ):
            constant_label = X_syn[self.target_column].iloc[0]

        if needs_val:
            stratify = self.task != "regression"
            X_train_fit, X_val, _, _ = split_validation(
                X_train,
                X_train[self.target_column],
                self.val_size,
                self.random_state,
                stratify=stratify,
            )
            if constant_label is None:
                X_syn_split = X_syn
                if self.task != "regression":
                    label_counts = X_syn_split[self.target_column].value_counts()
                    X_syn_split = X_syn_split[
                        X_syn_split[self.target_column].isin(
                            label_counts[label_counts > 1].index
                        )
                    ]
                X_syn_fit, X_syn_val, _, _ = split_validation(
                    X_syn_split,
                    X_syn_split[self.target_column],
                    self.val_size,
                    self.random_state,
                    stratify=stratify,
                )
                if self.train_set == "synthetic" and (
                    X_syn_fit[self.target_column].nunique() == 1
                ):
                    constant_label = X_syn_fit[self.target_column].iloc[0]

        if self.tune and not uses_xgboost_early_stopping:
            # load tuned params from file if it exists
            if os.path.exists(param_file):
                with open(param_file, "r") as f:
                    params = json.load(f)
            else:
                # tune params and save to file
                params = tune_ml_model(
                    X_train_fit.drop(columns=[self.target_column]),
                    X_val.drop(columns=[self.target_column]),
                    X_train_fit[self.target_column],
                    X_val[self.target_column],
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

        x_train_data = X_syn_fit if self.train_set == "synthetic" else X_train_fit
        score_fns = (
            [r2_score, root_mean_squared_error]
            if self.task == "regression"
            else [roc_auc_score, f1_score, accuracy_score]
        )

        if needs_val:
            val_data = X_syn_val if self.train_set == "real" else X_val
            x_val = val_data.drop(columns=[self.target_column])
            y_val = val_data[self.target_column]
        else:
            x_val = None
            y_val = None

        x_train = x_train_data.drop(columns=[self.target_column])
        y_train = x_train_data[self.target_column].copy()
        x_test = (
            X_syn.sample(n=len(X_test), random_state=self.random_state).copy()
            if self.train_set == "real"
            else X_test.copy()
        )
        y_test = x_test[self.target_column].copy()
        x_test = x_test.drop(columns=[self.target_column])
        task = self.task
        if task == "multiclass" and y_train.nunique() == 2:
            task = "binary"

        if constant_label is None:
            scores = ml_task(
                x_train,
                x_test,
                y_train,
                y_test,
                self.discrete_features,
                task,
                self.model_name,
                params,
                random_state=self.random_state,
                score_fns=score_fns,
                X_val=x_val,
                y_val=y_val,
            )
        else:
            target_preprocessor = LabelEncoder()
            target_preprocessor.fit(
                pd.concat(
                    [
                        X_train[self.target_column],
                        y_test,
                        pd.Series([constant_label]),
                    ]
                )
            )
            y_test_transformed = target_preprocessor.transform(y_test)
            constant_label_transformed = target_preprocessor.transform(
                [constant_label]
            )[0]
            pred_labels = np.full(len(y_test), constant_label_transformed)
            if self.task == "multiclass":
                preds = np.zeros((len(y_test), len(target_preprocessor.classes_)))
                preds[:, constant_label_transformed] = 1.0
            else:
                preds = pred_labels.astype(float)
            scores = score_ml_predictions(
                y_test_transformed,
                preds,
                pred_labels,
                score_fns,
                task,
            )
        result = {}
        for key, value in scores.items():
            result[f"{self.name}.{key}"] = value
        # add baseline scores (TRTR)
        if needs_val:
            x_val = X_val.copy().drop(columns=[self.target_column])
            y_val = X_val[self.target_column].copy()
        else:
            x_val = None
            y_val = None

        baseline_scores = ml_task(
            X_train_fit.drop(columns=[self.target_column]),
            X_test.copy().drop(columns=[self.target_column]),
            X_train_fit[self.target_column].copy(),
            X_test[self.target_column].copy(),
            self.discrete_features,
            self.task,
            self.model_name,
            params,
            random_state=self.random_state,
            score_fns=score_fns,
            X_val=x_val,
            y_val=y_val,
        )
        for key, value in baseline_scores.items():
            result[f"{self.name}.baseline.{key}"] = value
        return result
