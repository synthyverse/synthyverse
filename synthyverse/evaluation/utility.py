import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from ..utils.xgb_utils import get_xgb_tree_method


class MLE:
    """
    Machine Learning Efficacy from a XGB classifier.
    AUC score for discrete target columns, R^2 score for continuous target columns.
    """

    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_target_column = True
    needs_random_state = True

    def __init__(
        self,
        target_column: str = "target",
        discrete_features: list = [],
        random_state: int = 0,
        train_set: str = "synthetic",
    ):
        super().__init__()
        self.random_state = random_state
        self.discrete_features = discrete_features
        self.target_column = target_column
        self.train_set = train_set

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
    ):

        y_tr = train[self.target_column]
        y_te = test[self.target_column]
        y_s = sd[self.target_column]
        x_tr = train.drop(columns=[self.target_column])
        x_te = test.drop(columns=[self.target_column])
        x_s = sd.drop(columns=[self.target_column])

        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]
        discrete_features = [
            col for col in self.discrete_features if col != self.target_column
        ]

        x_tr[numerical_features], x_te[numerical_features], x_s[numerical_features] = (
            x_tr[numerical_features].astype(float),
            x_te[numerical_features].astype(float),
            x_s[numerical_features].astype(float),
        )
        x_tr[discrete_features], x_te[discrete_features], x_s[discrete_features] = (
            x_tr[discrete_features].astype("category"),
            x_te[discrete_features].astype("category"),
            x_s[discrete_features].astype("category"),
        )

        if self.target_column in self.discrete_features:
            le = LabelEncoder()
            le.fit(pd.concat((y_tr, y_te, y_s)))
            y_tr = le.transform(y_tr)
            y_te = le.transform(y_te)
            y_s = le.transform(y_s)
            model = XGBClassifier(
                tree_method=get_xgb_tree_method(),
                enable_categorical=True,
                random_state=self.random_state,
                max_depth=3,
            )
        else:
            model = XGBRegressor(
                tree_method=get_xgb_tree_method(),
                enable_categorical=True,
                random_state=self.random_state,
                max_depth=3,
            )

        if self.train_set == "synthetic":
            model.fit(x_s[: len(x_tr)], y_s[: len(x_tr)])
            score = self._get_score(y_te, x_te, model)
        else:
            model.fit(x_tr, y_tr)
            score = self._get_score(y_s[-len(x_te) :], x_s[-len(x_te) :], model)

        # also add trtr score
        model.fit(x_tr, y_tr)
        score_trtr = self._get_score(y_te, x_te, model)

        return {
            f"mle.train-{self.train_set}-test-{'real' if self.train_set == 'synthetic' else 'synthetic'}.{'auc' if self.target_column in self.discrete_features else 'r2'}": float(
                score
            ),
            f"mle.train-real-test-real.{'auc' if self.target_column in self.discrete_features else 'r2'}": float(
                score_trtr
            ),
        }

    def _get_score(self, y_te, X_te, model):
        if self.target_column in self.discrete_features:
            preds = model.predict_proba(X_te)
            if np.unique(y_te).shape[0] > 2:
                score = roc_auc_score(y_te, preds, multi_class="ovr", average="micro")
            else:
                score = roc_auc_score(y_te, preds[:, 1])
        else:
            preds = model.predict(X_te)
            score = r2_score(y_te, preds)
        return score
