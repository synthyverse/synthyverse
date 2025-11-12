import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    r2_score,
    f1_score,
    accuracy_score,
    root_mean_squared_error,
)
from .utils import xgboost_hyperparams
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os
import optuna


class MLE:
    """
    Machine Learning Efficacy from an XGBoost classifier.
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
        discrete_features: list = [],
        random_state: int = 0,
        train_set: str = "synthetic",  # whether to compute TSTR or TRTS
        model_params: dict = {"max_depth": 3, "tree_method": "hist"},
        tune: bool = False,
        tuning_trials: int = 32,
    ):
        super().__init__()
        self.random_state = random_state
        self.tune = tune
        self.tuning_trials = tuning_trials
        self.X_val = X_val
        self.discrete_features = discrete_features
        self.target_column = target_column
        self.train_set = train_set
        self.model_params = model_params
        self.model_params.update(
            {
                "random_state": self.random_state,
            }
        )
        self.prefix = f"mle.train-{self.train_set}-test-{'real' if self.train_set == 'synthetic' else 'synthetic'}"

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
    ):

        self.feature_types = [
            "c" if col in self.discrete_features else "q"
            for col in [x for x in train.columns if x != self.target_column]
        ]
        self.feature_names = [x for x in train.columns if x != self.target_column]

        if self.target_column in self.discrete_features:
            self.objective = (
                "multi:softprob"
                if len(
                    np.unique(
                        pd.concat(
                            (
                                train[self.target_column],
                                test[self.target_column],
                                sd[self.target_column],
                            )
                        )
                    )
                )
                > 2
                else "binary:logistic"
            )
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(
                pd.concat(
                    (
                        train[self.target_column],
                        test[self.target_column],
                        sd[self.target_column],
                    )
                )
            )
        else:
            self.objective = "reg:squarederror"
            self.scaler = StandardScaler()
            (
                self.scaler.fit(
                    pd.concat(
                        (
                            train[[self.target_column]],
                            test[[self.target_column]],
                            sd[[self.target_column]],
                        )
                    )
                )
                if self.X_val is None
                else self.scaler.fit(
                    pd.concat(
                        (
                            train[[self.target_column]],
                            test[[self.target_column]],
                            sd[[self.target_column]],
                            self.X_val[[self.target_column]],
                        )
                    )
                )
            )

        self.model_params.update({"objective": self.objective})

        if self.tune:
            assert self.X_val is not None, "X_val must be provided when tune=True."
            # try to load params from file - if it doesn't exist, we tune and save the params
            param_file = "synthyverse_hyperparams_tuned/mle.json"
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
        x_tr = train.drop(columns=[self.target_column]).to_numpy(copy=False)
        x_te = test.drop(columns=[self.target_column]).to_numpy(copy=False)

        if self.target_column in self.discrete_features:

            y_tr = self.label_encoder.transform(y_tr)
            y_te = self.label_encoder.transform(y_te)
        else:
            y_tr = self.scaler.transform(y_tr.reshape(-1, 1)).squeeze()
            y_te = self.scaler.transform(y_te.reshape(-1, 1)).squeeze()

        num_boost_round = (
            model_params["n_estimators"]
            if "n_estimators" in model_params.keys()
            else 100
        )

        dmatrix = xgb.QuantileDMatrix(
            data=x_tr,
            label=y_tr,
            feature_types=self.feature_types,
            feature_names=self.feature_names,
            enable_categorical=True,
        )

        model = xgb.train(
            model_params,
            dmatrix,
            num_boost_round=num_boost_round,
        )
        dmatrix = xgb.QuantileDMatrix(
            data=x_te,
            feature_types=self.feature_types,
            feature_names=self.feature_names,
            enable_categorical=True,
            ref=dmatrix,
        )
        preds = model.predict(
            dmatrix,
        )
        scores = self.score_fn(y_te, preds)
        return scores

    def _tune(self, train: pd.DataFrame):

        def objective(trial: optuna.Trial):
            params = xgboost_hyperparams(trial)
            params.update(
                {"objective": self.objective, "random_state": self.random_state}
            )

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

        best = study.best_params.copy()
        best.update(
            {
                "tree_method": "hist",
                "random_state": self.random_state,
                "objective": self.objective,
            }
        )

        return best

    def score_fn(self, y, preds):
        if self.target_column in self.discrete_features:
            return {
                "auc": roc_auc_score(y, preds, average="micro", multi_class="ovr"),
                "f1": f1_score(
                    y,
                    (
                        np.argmax(preds, axis=1)
                        if self.objective == "multi:softprob"
                        else (preds > 0.5).astype(int)
                    ),
                ),
                "accuracy": accuracy_score(
                    y,
                    (
                        np.argmax(preds, axis=1)
                        if self.objective == "multi:softprob"
                        else (preds > 0.5).astype(int)
                    ),
                ),
            }
        else:
            return {"r2": r2_score(y, preds), "rmse": root_mean_squared_error(y, preds)}
