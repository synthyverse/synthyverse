import inspect
import json
import os

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.utils import all_estimators

from .hyperparameters import sklearn_hyperparams, xgboost_hyperparams


class MLE:
    """Machine Learning Efficacy from configurable ML models.

    Measures how well synthetic data can be used for downstream machine learning
    tasks compared to real data.

    Args:
        X_val (pd.DataFrame, optional): Validation data for hyperparameter tuning. Default: None.
        target_column (str): Name of the target column. Default: "target".
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        random_state (int): Random seed for reproducibility. Default: 0.
        train_set (str): Which dataset to train on ("synthetic" for TSTR, "real" for TRTS). Default: "synthetic".
        model_name (str): Estimator name. Use "xgboost" for native XGBoost,
            or any sklearn estimator class name discoverable via all_estimators.
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
        ...     X_val=X_val,
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
    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_target_column = True
    needs_random_state = True
    needs_val_set = True

    def __init__(
        self,
        X_val: pd.DataFrame = None,
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
        self.X_val = X_val
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )
        self.target_column = target_column
        self.train_set = train_set
        self.model_name = model_name
        self.model_name_lc = model_name.lower()
        self.uses_xgboost = self.model_name_lc in {
            "xgboost",
            "xgb",
            "xgbclassifier",
            "xgbregressor",
        }

        self.model_params = model_params if model_params is not None else {}
        self.prefix = f"mle.train-{self.train_set}-test-{'real' if self.train_set == 'synthetic' else 'synthetic'}"

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        """Evaluate synthetic data utility using machine learning efficacy.

        Args:
            train: Training data as a pandas DataFrame.
            test: Test data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with MLE metric scores. Includes both synthetic-to-real
                and real-to-real baseline scores.
        """

        self.feature_names = [x for x in train.columns if x != self.target_column]
        self.categorical_features = [
            col for col in self.feature_names if col in self.discrete_features
        ]
        self.numerical_features = [
            col for col in self.feature_names if col not in self.discrete_features
        ]

        if self.target_column in self.discrete_features:
            all_labels = pd.concat(
                (train[self.target_column], test[self.target_column])
            )
            if self.X_val is not None:
                all_labels = pd.concat((all_labels, self.X_val[self.target_column]))

            self.num_classes = len(np.unique(all_labels))
            self.objective = (
                "multi:softprob" if self.num_classes > 2 else "binary:logistic"
            )
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)
        else:
            self.num_classes = None
            self.objective = "reg:squarederror"
            self.scaler = StandardScaler()
            self.scaler.fit(train[[self.target_column]])
        if self.tune:
            assert self.X_val is not None, "X_val must be provided when tune=True."
            # try to load params from file - if it doesn't exist, we tune and save the params
            task_type = (
                "classifier"
                if self.target_column in self.discrete_features
                else "regressor"
            )
            model_slug = self.model_name_lc.replace(" ", "_")
            param_file = f"synthyverse_hyperparams_tuned/mle_{task_type}_{model_slug}.json"
            if os.path.exists(param_file):
                with open(param_file, "r") as f:
                    params = json.load(f)
            else:
                params = self._tune(train)  # tune on RD only
                os.makedirs(os.path.dirname(param_file), exist_ok=True)
                with open(param_file, "w") as f:
                    json.dump(params, f)

            return self._evaluate(train, test, sd, params)
        else:
            return self._evaluate(train, test, sd, self.model_params)

    def _evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
        model_params: dict = None,
    ):
        if self.train_set == "synthetic":
            scores = self._ml_experiment(sd[: len(train)], test, model_params)
        else:
            scores = self._ml_experiment(train, sd[-len(test) :], model_params)

        outputs = {}
        outputs.update({f"{self.prefix}.{k}": v for k, v in scores.items()})

        scores = self._ml_experiment(train, test, model_params)
        outputs.update({f"mle.train-real-test-real.{k}": v for k, v in scores.items()})

        return outputs

    def _ml_experiment(
        self, train: pd.DataFrame, test: pd.DataFrame, model_params: dict = None
    ):

        y_tr = train[self.target_column].to_numpy(copy=False)
        y_te = test[self.target_column].to_numpy(copy=False)
        x_tr = train.drop(columns=[self.target_column])
        x_te = test.drop(columns=[self.target_column])
        if self.uses_xgboost or "histgradientboosting" in self.model_name_lc:
            x_tr, x_te = self._prepare_native_categorical_inputs(x_tr, x_te)

        if self.target_column in self.discrete_features:
            y_tr = self.label_encoder.transform(y_tr)
            y_te = self.label_encoder.transform(y_te)
            model = self._build_model(model_params)
            model.fit(x_tr, y_tr)
            preds, hard_preds = self._predict_classification(model, x_te)
        else:
            y_tr = self.scaler.transform(y_tr.reshape(-1, 1)).squeeze()
            y_te = self.scaler.transform(y_te.reshape(-1, 1)).squeeze()
            model = self._build_model(model_params)
            model.fit(x_tr, y_tr)
            preds = model.predict(x_te)
            hard_preds = None

        scores = self.score_fn(y_te, preds, hard_preds=hard_preds)
        return scores

    def _tune(self, train: pd.DataFrame):
        estimator_type = (
            "classifier" if self.target_column in self.discrete_features else "regressor"
        )

        def objective(trial: optuna.Trial):
            if self.uses_xgboost:
                params = xgboost_hyperparams(trial)
                params.update(
                    {"objective": self.objective, "random_state": self.random_state}
                )
            else:
                params = sklearn_hyperparams(trial, self.model_name, estimator_type)

            scores = self._ml_experiment(train, self.X_val, params)
            return (
                scores["auc"]
                if self.target_column in self.discrete_features
                else scores["r2"]
            )

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            direction="maximize",
        )
        study.optimize(
            objective,
            n_trials=self.tuning_trials,
            show_progress_bar=True,
        )

        return study.best_params.copy()

    def _supports_random_state(self, estimator_cls) -> bool:
        try:
            sig = inspect.signature(estimator_cls)
        except (TypeError, ValueError):
            return False
        return "random_state" in sig.parameters

    def _filter_supported_params(self, estimator_cls, params: dict) -> dict:
        params = {} if params is None else params.copy()
        try:
            sig = inspect.signature(estimator_cls)
        except (TypeError, ValueError):
            return params

        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return params

        allowed = set(sig.parameters.keys())
        return {k: v for k, v in params.items() if k in allowed}

    def _resolve_sklearn_estimator(self, estimator_type: str):
        estimators = {
            name.lower(): cls for name, cls in all_estimators(type_filter=estimator_type)
        }
        cls = estimators.get(self.model_name_lc)
        if cls is None:
            raise ValueError(
                f"Unknown {estimator_type} model_name '{self.model_name}'. "
                f"Please use a valid sklearn estimator name from all_estimators "
                f"or 'xgboost'."
            )
        return cls

    def _needs_numeric_scaling(self, model_cls) -> bool:
        name = model_cls.__name__.lower()
        no_scale_tokens = [
            "tree",
            "forest",
            "boosting",
            "bagging",
            "randomforest",
            "extratrees",
            "histgradientboosting",
            "isolationforest",
        ]
        return not any(token in name for token in no_scale_tokens)

    def _uses_native_histgb(self, model_cls) -> bool:
        return "histgradientboosting" in model_cls.__name__.lower()

    def _build_preprocessor(self, model_cls):
        numeric_transformer = (
            StandardScaler()
            if self._needs_numeric_scaling(model_cls)
            else "passthrough"
        )
        return ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
                ("numerical", numeric_transformer, self.numerical_features),
            ],
            remainder="drop",
        )

    def _build_model(self, model_params: dict = None):
        model_params = {} if model_params is None else model_params.copy()
        if self.uses_xgboost:
            model_params.setdefault("random_state", self.random_state)
            model_params.setdefault("objective", self.objective)
            model_params.setdefault("tree_method", "hist")
            if self.target_column in self.discrete_features:
                return xgb.XGBClassifier(
                    **model_params,
                    enable_categorical=True,
                )
            return xgb.XGBRegressor(
                **model_params,
                enable_categorical=True,
            )

        estimator_type = (
            "classifier" if self.target_column in self.discrete_features else "regressor"
        )
        model_cls = self._resolve_sklearn_estimator(estimator_type)

        if self._supports_random_state(model_cls):
            model_params.setdefault("random_state", self.random_state)
        model_params = self._filter_supported_params(model_cls, model_params)
        if self._uses_native_histgb(model_cls):
            model_params.setdefault("categorical_features", self.categorical_features)
            return model_cls(**model_params)

        estimator = model_cls(**model_params)
        preprocessor = self._build_preprocessor(model_cls)
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    def _prepare_native_categorical_inputs(
        self, x_tr: pd.DataFrame, x_te: pd.DataFrame
    ):
        x_tr = x_tr.copy()
        x_te = x_te.copy()
        for col in self.categorical_features:
            x_tr[col] = x_tr[col].astype("category")
            x_te[col] = pd.Categorical(x_te[col], categories=x_tr[col].cat.categories)
        return x_tr, x_te

    def _predict_classification(self, model, x_te: pd.DataFrame):
        hard_preds = model.predict(x_te)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(x_te)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_te)
            if np.ndim(scores) == 1:
                probs_pos = 1.0 / (1.0 + np.exp(-scores))
                preds = np.column_stack([1.0 - probs_pos, probs_pos])
            else:
                shifted = scores - np.max(scores, axis=1, keepdims=True)
                exp_scores = np.exp(shifted)
                denom = np.sum(exp_scores, axis=1, keepdims=True)
                preds = exp_scores / np.clip(denom, 1e-12, None)
        else:
            classes = np.arange(self.num_classes) if self.num_classes else np.array([0, 1])
            idx = np.searchsorted(classes, hard_preds)
            idx = np.clip(idx, 0, len(classes) - 1)
            preds = np.zeros((len(hard_preds), len(classes)))
            preds[np.arange(len(hard_preds)), idx] = 1.0

        if self.num_classes == 2:
            preds = preds[:, 1]
        return preds, hard_preds

    def score_fn(self, y, preds, hard_preds=None):
        if self.target_column in self.discrete_features:
            if self.objective == "multi:softprob":
                if hard_preds is None:
                    hard_preds = np.argmax(preds, axis=1)
                return {
                    "auc": roc_auc_score(
                        y, preds, average="weighted", multi_class="ovr"
                    ),
                    "f1": f1_score(y, hard_preds, average="weighted"),
                    "accuracy": accuracy_score(y, hard_preds),
                }
            else:
                if hard_preds is None:
                    hard_preds = (preds > 0.5).astype(int)
                return {
                    "auc": roc_auc_score(y, preds),
                    "f1": f1_score(y, hard_preds, average="binary"),
                    "accuracy": accuracy_score(y, hard_preds),
                }
        else:
            return {"r2": r2_score(y, preds), "rmse": root_mean_squared_error(y, preds)}
