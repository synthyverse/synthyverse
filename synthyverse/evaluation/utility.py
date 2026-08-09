import pandas as pd

from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    f1_score,
    accuracy_score,
)
from .ml import (
    ml_task,
    resolve_model_name,
    split_validation,
)
from .base import BaseMetric


class MLE(BaseMetric):
    """Machine Learning Efficacy from configurable ML models.

    Measures how well synthetic data can be used for downstream machine learning
    tasks compared to real data.

    Args:
        target_column (str): Name of the target column. Default: "target".
        random_state (int): Random seed for reproducibility. Default: 0.
        train_set (str): Which dataset to train on ("synthetic" for TSTR, "real" for TRTS). Evaluates on the opposite set. Default: "synthetic".
        model_name (str): Estimator family. Supported values include "xgboost",
            "randomforest", "decisiontree", "linearregression", and "svm",
            including common aliases. Every model except for XGBoost is a scikit-learn model. Default: "xgboost".
        model_params (dict): Model parameters passed to the selected estimator.
            For XGBoost, passing ``early_stopping_rounds`` enables early stopping
            and requires ``val_size > 0``.
        val_size (float): Fraction of the fitting train set reserved for
            validation when XGBoost early stopping needs it. Default: 0.15.
        include_baseline (bool): Whether to compute and include the real-train/
            real-test baseline. Default: True.

    Outputs:
        Keys are "mle.<score>" plus "mle.baseline.<score>" when
        include_baseline is True.

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
        ...     train_set="synthetic",
        ...     random_state=42
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_syn, X_test=X_test, discrete_features=discrete_features)
    """

    name = "mle"

    def __init__(
        self,
        target_column: str = "target",
        random_state: int = 0,
        train_set: str = "synthetic",  # whether to compute TSTR or TRTS
        model_name: str = "xgboost",
        model_params: dict = None,
        val_size: float = 0.15,
        include_baseline: bool = True,
    ):
        super().__init__(random_state=random_state)
        self.val_size = val_size
        self.include_baseline = include_baseline
        self.target_column = target_column
        if train_set not in {"synthetic", "real"}:
            raise ValueError("train_set must be either 'synthetic' or 'real'.")
        self.train_set = train_set
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        if X_test is None:
            raise ValueError("MLE requires X_test.")

        self.task = "regression"
        if self.target_column in discrete_features:
            self.task = (
                "multiclass" if X[self.target_column].nunique() > 2 else "binary"
            )

        model_name = resolve_model_name(self.model_name)
        uses_xgboost_early_stopping = (
            model_name == "xgboost"
            and self.model_params.get("early_stopping_rounds") is not None
        )

        X_train_fit = X
        X_syn_fit = X_syn
        X_val = None
        X_syn_val = None

        if uses_xgboost_early_stopping:
            stratify = self.task != "regression"
            X_train_fit, X_val, _, _ = split_validation(
                X,
                X[self.target_column],
                self.val_size,
                self.random_state,
                stratify=stratify,
            )
            if self.train_set == "synthetic" and (
                self.task == "regression" or X_syn[self.target_column].nunique() > 1
            ):
                X_syn_fit, X_syn_val, _, _ = split_validation(
                    X_syn,
                    X_syn[self.target_column],
                    self.val_size,
                    self.random_state,
                    stratify=stratify,
                )

        x_train_data = X_syn_fit if self.train_set == "synthetic" else X_train_fit
        if self.train_set == "real" and len(X_syn) < len(X_test):
            raise ValueError(
                "MLE with train_set='real' requires X_syn to contain at least "
                f"as many rows as X_test ({len(X_syn)} < {len(X_test)})."
            )
        x_test_data = (
            X_syn.sample(n=len(X_test), random_state=self.random_state)
            if self.train_set == "real"
            else X_test
        )
        y_train = x_train_data[self.target_column]
        y_test = x_test_data[self.target_column]
        task = self.task
        if task == "multiclass" and y_train.nunique() == 2:
            task = "binary"

        score_fns = (
            [r2_score, root_mean_squared_error]
            if self.task == "regression"
            else [roc_auc_score, f1_score, accuracy_score]
        )

        if uses_xgboost_early_stopping and not (
            task != "regression" and y_train.nunique() == 1
        ):
            val_data = X_syn_val if self.train_set == "synthetic" else X_val
            x_val = val_data.drop(columns=[self.target_column])
            y_val = val_data[self.target_column]
        else:
            x_val = None
            y_val = None

        scores = ml_task(
            x_train_data.drop(columns=[self.target_column]),
            x_test_data.drop(columns=[self.target_column]),
            y_train,
            y_test,
            discrete_features,
            task,
            self.model_name,
            self.model_params,
            random_state=self.random_state,
            score_fns=score_fns,
            X_val=x_val,
            y_val=y_val,
        )
        result = {f"{self.name}.{key}": value for key, value in scores.items()}

        if not self.include_baseline:
            return result

        # add baseline scores (TRTR)
        if uses_xgboost_early_stopping:
            x_val = X_val.drop(columns=[self.target_column])
            y_val = X_val[self.target_column]
        else:
            x_val = None
            y_val = None

        baseline_scores = ml_task(
            X_train_fit.drop(columns=[self.target_column]),
            X_test.drop(columns=[self.target_column]),
            X_train_fit[self.target_column],
            X_test[self.target_column],
            discrete_features,
            self.task,
            self.model_name,
            self.model_params,
            random_state=self.random_state,
            score_fns=score_fns,
            X_val=x_val,
            y_val=y_val,
        )
        result.update(
            {
                f"{self.name}.baseline.{key}": value
                for key, value in baseline_scores.items()
            }
        )
        return result
