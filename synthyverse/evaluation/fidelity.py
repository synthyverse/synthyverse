import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from ..utils.xgb_utils import get_xgb_tree_method
from ..utils.oneclass import OneClassLayer
from ..utils.utils import suppress_print
import torch
from sklearn.neighbors import NearestNeighbors
from sdmetrics.reports.single_table import QualityReport


class ClassifierTest:
    """
    AUC score of XGB classifier which aims to distinguish synthetic from real data.
    """

    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_random_state = True

    def __init__(
        self,
        discrete_features: list = [],
        random_state: int = 0,
    ):
        super().__init__()
        self.random_state = random_state
        self.discrete_features = discrete_features

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
    ):

        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]

        X = pd.concat((train, sd[: len(train)]))
        X = X.reset_index(drop=True)
        X[numerical_features] = X[numerical_features].astype(float)
        X[self.discrete_features] = X[self.discrete_features].astype("category")
        y = pd.concat(
            (
                pd.Series(0, index=list(range(len(train))), name="y"),
                pd.Series(1, index=list(range(len(train))), name="y"),
            )
        )
        y = y.reset_index(drop=True)

        model = XGBClassifier(
            tree_method=get_xgb_tree_method(),
            enable_categorical=True,
            random_state=self.random_state,
            max_depth=3,
        )

        model.fit(X, y)

        X_te = pd.concat((test, sd[-len(test) :]))
        X_te[numerical_features] = X_te[numerical_features].astype(float)
        X_te[self.discrete_features] = X_te[self.discrete_features].astype("category")
        y_te = pd.concat(
            (
                pd.Series(0, index=list(range(len(test))), name="y"),
                pd.Series(1, index=list(range(len(test))), name="y"),
            )
        )
        y_te = y_te.reset_index(drop=True)

        preds = model.predict_proba(X_te)
        score = roc_auc_score(y_te, preds[:, 1])

        return {f"classifiertest.auc": float(score)}


class AlphaPrecisionBetaRecallAuthenticity:
    """
    alpha-Precision, Beta-Recall, Authenticity score from the Alaa et al. paper.
    """

    data_requirement = "train_preprocessed"
    needs_random_state = True

    def __init__(
        self,
        discrete_features: list = [],
        random_state: int = 0,
    ):
        super().__init__()
        self.random_state = random_state
        self.discrete_features = discrete_features

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):

        OC_params = {
            "input_dim": rd.shape[1],
            "rep_dim": rd.shape[1],
            "num_layers": 4,
            "num_hidden": 32,
            "activation": "ReLU",
            "dropout_prob": 0.2,
            "dropout_active": False,
            "LossFn": "SoftBoundary",
            "lr": 2e-3,
            "epochs": 1000,
            "warm_up_epochs": 20,
            "train_prop": 1.0,
            "weight_decay": 2e-3,
        }
        OC_hyperparams = {"Radius": 1, "nu": 1e-2}
        OC_hyperparams["center"] = (
            torch.ones(OC_params["rep_dim"]) * 10
        )  # *10 is what is used in synthcity
        OC_model = OneClassLayer(params=OC_params, hyperparams=OC_hyperparams)
        OC_model.fit(rd.values, verbosity=True)
        real = OC_model.predict(rd.values)
        syn = OC_model.predict(sd.values)
        emb_center = OC_model.c.detach().cpu().numpy()

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        Radii = np.quantile(np.sqrt(np.sum((real - emb_center) ** 2, axis=1)), alphas)

        synth_center = np.mean(syn, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        synth_to_center = np.sqrt(np.sum((syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real)
        real_to_real, _ = nbrs_real.kneighbors(real)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(syn)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real)

        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = syn[real_to_synth_args]

        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        )
        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(Radii)):
            precision_audit_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_audit_mask)

            beta_coverage = np.mean(
                (
                    (real_to_synth <= real_to_real)
                    * (real_synth_closest_d <= closest_synth_Radii[k])
                )
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)

        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)

        return {
            "alphaprecision.oc.score": float(Delta_precision_alpha),
            "betacoverage.oc.score": float(Delta_coverage_beta),
            "authenticity.oc.score": float(authenticity),
        }


class Similarity:
    """
    Column Shapes and Column Pair Trends from the SDMetrics library.
    Indicates quality of marginal distributions and correlations in synthetic data, respectively.
    """

    data_requirement = "train"
    needs_discrete_features = True

    def __init__(
        self,
        discrete_features: list = [],
    ):
        super().__init__()
        self.discrete_features = discrete_features

    @suppress_print
    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        dtypes = [
            "categorical" if x in self.discrete_features else "numerical"
            for x in rd.columns
        ]
        metadata = {k: {"sdtype": v} for k, v in zip(rd.columns, dtypes)}
        metadata = {"columns": metadata}
        metadata["primary_key"] = "index"

        report = QualityReport()
        report.generate(rd, sd, metadata)
        scores = report.get_properties()

        return {
            "similarity.shape": float(
                scores.loc[scores["Property"] == "Column Shapes", "Score"]
            ),
            "similarity.trend": float(
                scores.loc[scores["Property"] == "Column Pair Trends", "Score"]
            ),
        }
