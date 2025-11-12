import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sdmetrics.reports.single_table import QualityReport
import optuna
import json
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from .utils import xgboost_hyperparams


class ClassifierTest:
    """
    AUC score of XGB classifier which aims to distinguish synthetic from real data.
    """

    name = "classifier_test"

    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_random_state = True
    needs_val_set = True

    def __init__(
        self,
        X_val: pd.DataFrame = None,
        discrete_features: list = [],
        random_state: int = 0,
        clf_params: dict = {
            "objective": "binary:logistic",
            "max_depth": 3,
            "tree_method": "hist",
        },
        tune: bool = False,
        tuning_trials: int = 32,
    ):
        super().__init__()
        self.random_state = random_state
        self.discrete_features = discrete_features
        self.X_val = X_val
        self.tune = tune
        self.tuning_trials = tuning_trials

        self.clf_params = clf_params
        self.clf_params.update(
            {
                "random_state": self.random_state,
            }
        )

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
    ):

        if self.tune:
            assert self.X_val is not None, "X_val must be provided when tune=True."
            # try to load params from file - if it doesn't exist, we tune and save the params
            param_file = "synthyverse_hyperparams_tuned/classifier_test.json"
            if os.path.exists(param_file):
                with open(param_file, "r") as f:
                    params = json.load(f)
            else:
                params = self._tune(train, sd)
                os.makedirs(os.path.dirname(param_file), exist_ok=True)
                with open(param_file, "w") as f:
                    json.dump(params, f)

            return self._evaluate(train, test, sd, params)
        else:
            return self._evaluate(train, test, sd, self.clf_params)

    def _evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
        clf_params: dict = None,
    ):

        feature_types = [
            "c" if col in self.discrete_features else "q" for col in train.columns
        ]
        feature_names = train.columns.tolist()

        # training
        y = np.concatenate((np.zeros(len(train)), np.ones(len(train))))
        x = (
            pd.concat((train, sd[: len(train)]))
            .reset_index(drop=True)
            .to_numpy(copy=False)
        )

        dmatrix = xgb.QuantileDMatrix(
            data=x,
            label=y,
            feature_types=feature_types,
            feature_names=feature_names,
            enable_categorical=True,
        )
        num_boost_round = (
            clf_params["n_estimators"] if "n_estimators" in clf_params.keys() else 100
        )
        model = xgb.train(clf_params, dmatrix, num_boost_round=num_boost_round)

        # evaluation
        y = np.concatenate((np.zeros(len(test)), np.ones(len(test))))
        x = (
            pd.concat((test, sd[-len(test) :]))
            .reset_index(drop=True)
            .to_numpy(copy=False)
        )
        dmatrix = xgb.QuantileDMatrix(
            data=x,
            feature_types=feature_types,
            feature_names=feature_names,
            enable_categorical=True,
            ref=dmatrix,
        )
        preds = model.predict(dmatrix)

        score = roc_auc_score(y, preds)

        return {f"classifiertest.auc": float(score)}

    def _tune(self, train: pd.DataFrame, sd: pd.DataFrame):

        def objective(trial: optuna.Trial):
            params = xgboost_hyperparams(trial)
            params.update(
                {"objective": "binary:logistic", "random_state": self.random_state}
            )

            return self._evaluate(train, self.X_val, sd, params)[f"classifiertest.auc"]

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
                "objective": "binary:logistic",
                "tree_method": "hist",
                "random_state": self.random_state,
            }
        )

        return best


class AlphaPrecisionBetaRecallAuthenticity:
    """
    alpha-Precision, Beta-Recall, Authenticity score from the Alaa et al. paper.
    """

    name = "prauth"
    data_requirement = "train"
    needs_discrete_features = True

    def __init__(self, discrete_features: list = []):
        super().__init__()
        self.discrete_features = discrete_features

    def evaluate(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        numerical_features = [
            col for col in rd.columns if col not in self.discrete_features
        ]

        # one hot and standard scale
        onehot_encoder = OneHotEncoder(sparse_output=False)
        onehot_encoder.fit(
            pd.concat([rd[self.discrete_features], sd[self.discrete_features]])
        )
        scaler = MinMaxScaler()
        scaler.fit(rd[numerical_features])

        data = {}
        for df, name in zip([rd, sd], ["rd", "sd"]):
            cat = onehot_encoder.transform(df[self.discrete_features])
            cat = cat / 2  # scaling for Gower distance
            num = scaler.transform(df[numerical_features])
            data[name] = np.concatenate((cat, num), axis=1)

        x_rd = data["rd"]
        x_sd = data["sd"]

        emb_center = np.mean(x_rd, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        # Radii = np.quantile(np.sqrt(np.sum((x_rd - emb_center) ** 2, axis=1)), alphas)
        # use L1 distance to get Gower-type distance
        Radii = np.quantile(np.sum(np.abs(x_rd - emb_center), axis=1), alphas)

        synth_center = np.mean(x_sd, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        # synth_to_center = np.sqrt(np.sum((x_sd - emb_center) ** 2, axis=1))
        # use L1 distance to get Gower-type distance
        synth_to_center = np.sum(np.abs(x_sd - emb_center), axis=1)

        # use L1 distance to get Gower-type distance
        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=1).fit(x_rd)
        real_to_real, _ = nbrs_real.kneighbors(x_rd)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=1).fit(x_sd)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(x_rd)

        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = x_sd[real_to_synth_args]

        # real_synth_closest_d = np.sqrt(
        #     np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        # )
        # use L1 distance to get Gower-type distance
        real_synth_closest_d = np.sum(np.abs(real_synth_closest - synth_center), axis=1)

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
            "alphaprecision.naive.score": float(Delta_precision_alpha),
            "betacoverage.naive.score": float(Delta_coverage_beta),
            "authenticity.naive.score": float(authenticity),
        }


class Similarity:
    """
    Column Shapes and Column Pair Trends from the SDMetrics library.
    Indicates quality of marginal distributions and correlations in synthetic data, respectively.
    """

    name = "similarity"
    data_requirement = "train"
    needs_discrete_features = True

    def __init__(
        self,
        discrete_features: list = [],
    ):
        super().__init__()
        self.discrete_features = discrete_features

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
        report.generate(rd, sd, metadata, verbose=False)
        scores = report.get_properties()

        return {
            "similarity.shape": float(
                scores.loc[scores["Property"] == "Column Shapes", "Score"]
            ),
            "similarity.trend": float(
                scores.loc[scores["Property"] == "Column Pair Trends", "Score"]
            ),
        }


class ImputationMAE_MAD:
    name = "mae_mad"
    data_requirement = "train"
    needs_discrete_features = True

    def __init__(self, discrete_features: list = []):
        self.__dict__.update(locals())

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        # average over #observations BEFORE OHE
        n_miss = int(rd.shape[0] * rd.shape[1])

        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(pd.concat([rd[self.discrete_features], sd[self.discrete_features]]))

        numerical_features = [x for x in rd.columns if x not in self.discrete_features]

        scaler = MinMaxScaler()
        scaler.fit(rd[numerical_features])

        data = {}
        for df, name in zip([rd, sd], ["rd", "sd"]):
            cat = ohe.transform(df[self.discrete_features]) / 2
            num = scaler.transform(df[numerical_features])
            data[name] = np.concatenate((num, cat), axis=1)

        mad = (
            np.sum(
                np.absolute(data["sd"] - np.median(data["sd"], axis=0, keepdims=False))
            )
            / n_miss
        )

        mae = np.sum(np.absolute(data["rd"] - data["sd"])) / n_miss

        return {
            "imputation.mae": mae,
            "imputation.mad": mad,
        }
