import inspect
import json
import os

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sdmetrics.reports.single_table import QualityReport
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.utils import all_estimators
from .hyperparameters import sklearn_hyperparams, xgboost_hyperparams
from scipy.stats import spearmanr, pearsonr, chi2_contingency, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


class ClassifierTest:
    """AUC score of a classifier that distinguishes synthetic from real data.

    Lower scores indicate better quality synthetic data (harder to distinguish from real).

    Args:
        X_val (pd.DataFrame, optional): Validation data for hyperparameter tuning. Default: None.
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        random_state (int): Random seed for reproducibility. Default: 0.
        model_name (str): Classifier name. Use "xgboost" for native XGBoost,
            or any sklearn classifier class name discoverable via all_estimators.
        clf_params (dict): Classifier parameters passed to the selected estimator.
        tune (bool): Whether to tune hyperparameters. Default: False.
        tuning_trials (int): Number of Optuna trials for hyperparameter tuning. Default: 32.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import ClassifierTest
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> X_val = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = ClassifierTest(
        ...     X_val=X_val,
        ...     discrete_features=discrete_features,
        ...     tune=True,
        ...     random_state=42
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_test, X_syn)
    """

    name = "classifier_test"

    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_random_state = True
    needs_val_set = True

    def __init__(
        self,
        X_val: pd.DataFrame = None,
        discrete_features: list = None,
        random_state: int = 0,
        model_name: str = "xgboost",
        clf_params: dict = None,
        tune: bool = False,
        tuning_trials: int = 32,
    ):
        super().__init__()
        self.random_state = random_state
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )
        self.X_val = X_val
        self.tune = tune
        self.tuning_trials = tuning_trials
        self.model_name = model_name
        self.model_name_lc = model_name.lower()
        self.uses_xgboost = self.model_name_lc in {
            "xgboost",
            "xgb",
            "xgbclassifier",
        }

        self.clf_params = clf_params if clf_params is not None else {}
    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sd: pd.DataFrame,
    ):
        """Evaluate synthetic data using classifier test.

        Args:
            train: Training data as a pandas DataFrame.
            test: Test data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with "classifiertest.auc" key and AUC score value.
        """

        if self.tune:
            assert self.X_val is not None, "X_val must be provided when tune=True."
            # try to load params from file - if it doesn't exist, we tune and save the params
            model_slug = self.model_name_lc.replace(" ", "_")
            param_file = (
                f"synthyverse_hyperparams_tuned/classifier_test_{model_slug}.json"
            )
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

        categorical_features = [
            col for col in train.columns if col in self.discrete_features
        ]
        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]

        # training
        y = np.concatenate((np.zeros(len(train)), np.ones(len(train))))
        x = pd.concat((train, sd[: len(train)])).reset_index(drop=True)
        if self.uses_xgboost or "histgradientboosting" in self.model_name_lc:
            x = self._prepare_native_categorical_input(x, categorical_features)
        x_train_model = x

        model = self._build_classifier(
            clf_params=clf_params,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        model.fit(x, y)

        # evaluation
        y = np.concatenate((np.zeros(len(test)), np.ones(len(test))))
        x = pd.concat((test, sd[-len(test) :])).reset_index(drop=True)
        if self.uses_xgboost or "histgradientboosting" in self.model_name_lc:
            x = self._prepare_native_categorical_input(
                x, categorical_features, categories_ref=x_train_model
            )
        preds = self._predict_binary_scores(model, x)

        score = roc_auc_score(y, preds)

        return {f"classifiertest.auc": float(score)}

    def _tune(self, train: pd.DataFrame, sd: pd.DataFrame):

        def objective(trial: optuna.Trial):
            if self.uses_xgboost:
                params = xgboost_hyperparams(trial)
                params.update(
                    {"objective": "binary:logistic", "random_state": self.random_state}
                )
            else:
                params = sklearn_hyperparams(
                    trial, self.model_name, estimator_type="classifier"
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

    def _resolve_classifier(self):
        estimators = {
            name.lower(): cls for name, cls in all_estimators(type_filter="classifier")
        }
        cls = estimators.get(self.model_name_lc)
        if cls is None:
            raise ValueError(
                f"Unknown classifier model_name '{self.model_name}'. "
                f"Please use a valid sklearn classifier name from all_estimators "
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

    def _build_preprocessor(self, model_cls, categorical_features, numerical_features):
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
                    categorical_features,
                ),
                ("numerical", numeric_transformer, numerical_features),
            ],
            remainder="drop",
        )

    def _build_classifier(
        self,
        clf_params: dict,
        categorical_features: list,
        numerical_features: list,
    ):
        clf_params = {} if clf_params is None else clf_params.copy()
        if self.uses_xgboost:
            clf_params.setdefault("random_state", self.random_state)
            clf_params.setdefault("objective", "binary:logistic")
            clf_params.setdefault("tree_method", "hist")
            return xgb.XGBClassifier(
                **clf_params,
                enable_categorical=True,
            )

        model_cls = self._resolve_classifier()
        if self._supports_random_state(model_cls):
            clf_params.setdefault("random_state", self.random_state)
        clf_params = self._filter_supported_params(model_cls, clf_params)
        if self._uses_native_histgb(model_cls):
            clf_params.setdefault("categorical_features", categorical_features)
            return model_cls(**clf_params)

        estimator = model_cls(**clf_params)
        preprocessor = self._build_preprocessor(
            model_cls, categorical_features, numerical_features
        )
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    def _prepare_native_categorical_input(
        self,
        x: pd.DataFrame,
        categorical_features: list,
        categories_ref: pd.DataFrame = None,
    ):
        x = x.copy()
        for col in categorical_features:
            if categories_ref is None:
                x[col] = x[col].astype("category")
            else:
                x[col] = pd.Categorical(x[col], categories=categories_ref[col].cat.categories)
        return x

    def _predict_binary_scores(self, model, x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]

        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)
            if np.ndim(scores) > 1:
                scores = scores[:, 0]
            return 1.0 / (1.0 + np.exp(-scores))

        return model.predict(x).astype(float)


class AlphaPrecisionBetaRecallAuthenticity:
    """Alpha-Precision, Beta-Recall, Authenticity score.

    Paper: "How faithful is your synthetic data? sample-level metrics for evaluating and auditing generative models" by Alaa et al. (2022).

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import AlphaPrecisionBetaRecallAuthenticity
        >>>
        >>> # Prepare data
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = AlphaPrecisionBetaRecallAuthenticity(
        ...     discrete_features=discrete_features
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_real, X_syn)
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
        """Evaluate synthetic data using alpha-precision, beta-recall, and authenticity.

        Args:
            rd: Real data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with keys:
                - "alphaprecision.naive.score": Alpha-precision score
                - "betacoverage.naive.score": Beta-coverage score
                - "authenticity.naive.score": Authenticity score
        """
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


class ShapeTrend:
    """Column Shapes and Column Pair Trends from the SDMetrics library (https://docs.sdv.dev/sdmetrics/)

    Indicates quality of marginal distributions and correlations in synthetic data,
    respectively.

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import ShapeTrend
        >>>
        >>> # Prepare data
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = ShapeTrend(discrete_features=discrete_features)
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_real, X_syn)
    """

    name = "shapetrend"
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
        """Evaluate synthetic data using SDMetrics shape and trend scores.

        Args:
            rd: Real data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with keys:
                - "shapetrend.shape": Column shapes score
                - "shapetrend.trend": Column pair trends score
        """
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

        shape = scores.loc[scores["Property"] == "Column Shapes", "Score"]
        trend = scores.loc[scores["Property"] == "Column Pair Trends", "Score"]

        return {
            "shapetrend.shape": float(shape.iloc[0]),
            "shapetrend.trend": float(trend.iloc[0]),
        }


class Marginals:
    """Per-column distributional distance between real and synthetic marginals.

    Computes a distance metric for each column independently and returns
    the average distance over numerical and categorical features separately.
    Supported distance functions: Wasserstein (wsd), Jensen-Shannon divergence
    (jsd), Kolmogorov-Smirnov statistic (ks), and Total Variation distance (tvd).
    For histogram-based metrics (jsd, tvd) on numerical features, values are
    discretized into equal-width bins before comparison.

    Lower scores indicate better fidelity to the real marginals.

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        numerical_distance (str): Distance metric for numerical features.
            One of "wsd", "jsd", "ks", or "tvd". Default: "wsd".
        categorical_distance (str): Distance metric for categorical features.
            One of "jsd", "tvd", "wsd", or "ks". Default: "jsd".
        n_bins_numerical (int): Number of equal-width bins used when discretizing
            numerical features for jsd/tvd. Must be >= 2. Default: 30.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import Marginals
        >>>
        >>> # Prepare data
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = Marginals(
        ...     discrete_features=discrete_features,
        ...     numerical_distance="wsd",
        ...     categorical_distance="jsd",
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_real, X_syn)
    """

    name = "marginals"
    data_requirement = "train"
    needs_discrete_features = True

    def __init__(
        self,
        discrete_features: list = [],
        numerical_distance: str = "wsd",
        categorical_distance: str = "jsd",
        n_bins_numerical: int = 30,
    ):
        super().__init__()
        self.discrete_features = discrete_features

        self.numerical_distance = self._check_distance(numerical_distance)
        self.categorical_distance = self._check_distance(categorical_distance)

        self.n_bins_numerical = int(n_bins_numerical)
        if self.n_bins_numerical < 2:
            raise ValueError("n_bins_numerical must be >= 2")

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        """Evaluate synthetic data by comparing marginal distributions.

        Args:
            rd: Real data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with keys:
                - "marginals.<numerical_distance>": Mean distance over numerical features
                - "marginals.<categorical_distance>": Mean distance over categorical features
        """
        rd = rd.copy()
        sd = sd.copy()

        numerical_features = [c for c in rd.columns if c not in self.discrete_features]

        dist_func = {
            "wsd": self._wsd,
            "jsd": self._jsd,
            "ks": self._ks,
            "tvd": self._tvd,
        }

        # For numerical JSD/TVD we need to discretize to compare histograms
        if self.numerical_distance in ["jsd", "tvd"]:
            if len(numerical_features) > 0:
                n_bins = min(len(rd), self.n_bins_numerical)
                # KBinsDiscretizer expects 2D; it will bin per-feature
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="uniform"
                )
                discretizer.fit(rd[numerical_features])
                rd[numerical_features] = discretizer.transform(rd[numerical_features])
                sd[numerical_features] = discretizer.transform(sd[numerical_features])

        num = []
        for col in numerical_features:
            num.append(dist_func[self.numerical_distance](rd[col], sd[col]))

        cat = []
        for col in self.discrete_features:
            cat.append(dist_func[self.categorical_distance](rd[col], sd[col]))

        return {
            f"marginals.{self.numerical_distance}": (
                float(np.mean(num)) if len(num) else np.nan
            ),
            f"marginals.{self.categorical_distance}": (
                float(np.mean(cat)) if len(cat) else np.nan
            ),
        }

    def _jsd(self, s1: pd.Series, s2: pd.Series) -> float:
        p = s1.value_counts(normalize=True, dropna=False)
        q = s2.value_counts(normalize=True, dropna=False)
        support = p.index.union(q.index)
        p = p.reindex(support, fill_value=0.0).to_numpy(dtype=float)
        q = q.reindex(support, fill_value=0.0).to_numpy(dtype=float)
        return float(jensenshannon(p, q, base=2))

    def _tvd(self, s1: pd.Series, s2: pd.Series) -> float:
        p = s1.value_counts(normalize=True, dropna=False)
        q = s2.value_counts(normalize=True, dropna=False)
        support = p.index.union(q.index)
        p = p.reindex(support, fill_value=0.0).to_numpy(dtype=float)
        q = q.reindex(support, fill_value=0.0).to_numpy(dtype=float)
        return float(0.5 * np.abs(p - q).sum())

    def _ks(self, s1: pd.Series, s2: pd.Series) -> float:
        x = s1.to_numpy()
        y = s2.to_numpy()
        # ks_2samp handles ties; returns statistic in [0, 1]
        return float(ks_2samp(x, y, alternative="two-sided", mode="auto").statistic)

    def _wsd(self, s1: pd.Series, s2: pd.Series) -> float:
        mu = float(np.mean(s1))
        sigma = float(np.std(s1))
        if sigma == 0.0 or not np.isfinite(sigma):
            return float(wasserstein_distance(s1, s2))
        s1z = (s1 - mu) / sigma
        s2z = (s2 - mu) / sigma
        return float(wasserstein_distance(s1z, s2z))

    def _check_distance(self, distance: str) -> str:
        distance = distance.lower()
        if distance in ["wsd", "wasserstein", "ws"]:
            return "wsd"
        elif distance in ["jsd", "jensenshannon", "js"]:
            return "jsd"
        elif distance in [
            "ks",
            "kstest",
            "kolmogorov",
            "kolmogorovsmirnov",
            "kolmogorov-smirnov",
        ]:
            return "ks"
        elif distance in [
            "tv",
            "tvd",
            "totalvariation",
            "total_variation",
            "total-variation",
        ]:
            return "tvd"
        else:
            raise ValueError(f"Invalid distance: {distance}")


class Correlations:
    """Pairwise correlation matrix difference between real and synthetic data.

    Builds a full correlation matrix for both real and synthetic data and
    returns the L2 norm of their absolute difference. Correlation type is
    chosen automatically per feature pair: Spearman/Pearson for
    numerical-numerical, Cramer's V for categorical-categorical, and the
    correlation ratio (eta-squared) for mixed pairs.

    Lower scores indicate better preservation of feature dependencies.

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        numerical_correlation (str): Correlation method for numerical-numerical pairs.
            One of "spearman" or "pearson". Default: "spearman".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import Correlations
        >>>
        >>> # Prepare data
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = Correlations(
        ...     discrete_features=discrete_features,
        ...     numerical_correlation="spearman",
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_real, X_syn)
    """

    name = "correlations"
    data_requirement = "train"
    needs_discrete_features = True

    def __init__(
        self,
        discrete_features: list = [],
        numerical_correlation: str = "spearman",
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.numerical_correlation = numerical_correlation

    def evaluate(self, rd: pd.DataFrame, sd: pd.DataFrame):
        """Evaluate synthetic data by comparing pairwise correlation matrices.

        Args:
            rd: Real data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with key:
                - "correlations.l2": L2 norm of the absolute difference between
                  the real and synthetic correlation matrices
        """
        rd = rd.copy()
        sd = sd.copy()

        cols = rd.columns.tolist()
        n = len(cols)

        C_rd = np.zeros((n, n))
        C_sd = np.zeros((n, n))

        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if j < i:
                    C_rd[i, j] = C_rd[j, i]
                    C_sd[i, j] = C_sd[j, i]
                    continue

                corr_type = self._get_corr_type(ci, cj)

                C_rd[i, j] = self._get_corr(rd[ci], rd[cj], corr_type)
                C_sd[i, j] = self._get_corr(sd[ci], sd[cj], corr_type)

        diff = np.abs(C_rd - C_sd)
        l2 = np.linalg.norm(diff)

        return {
            "correlations.l2": float(l2),
        }

    # ---------- helpers ----------

    def _get_corr_type(self, c1: str, c2: str) -> str:
        d1 = c1 in self.discrete_features
        d2 = c2 in self.discrete_features
        if not d1 and not d2:
            return "numerical"
        if d1 and d2:
            return "categorical"
        return "mixed"

    def _get_corr(self, s1: pd.Series, s2: pd.Series, corr_type: str) -> float:
        if corr_type == "numerical":
            return self._num_corr(s1, s2)
        elif corr_type == "categorical":
            return self._cat_corr(s1, s2)
        elif corr_type == "mixed":
            # correlation ratio: categorical explains numerical
            if s1.name in self.discrete_features:
                return self._mixed_corr(s1, s2)
            else:
                return self._mixed_corr(s2, s1)
        else:
            raise ValueError(f"Invalid corr_type: {corr_type}")

    def _num_corr(self, s1: pd.Series, s2: pd.Series) -> float:
        if self.numerical_correlation == "spearman":
            return float(spearmanr(s1, s2).statistic)
        elif self.numerical_correlation == "pearson":
            return float(pearsonr(s1, s2).statistic)
        else:
            raise ValueError(
                f"Invalid numerical correlation: {self.numerical_correlation}"
            )

    def _cat_corr(self, s1: pd.Series, s2: pd.Series) -> float:
        ct = pd.crosstab(s1, s2)
        if ct.size == 0:
            return 0.0
        chi2 = chi2_contingency(ct, correction=False)[0]
        n = ct.values.sum()
        r, k = ct.shape
        denom = n * (min(r - 1, k - 1))
        if denom == 0:
            return 0.0
        return float(np.sqrt(chi2 / denom))

    def _mixed_corr(self, cat: pd.Series, num: pd.Series) -> float:
        """
        η^2 = Var(E[X|Y]) / Var(X)
        """
        x = num.to_numpy()
        y = cat.to_numpy()

        if np.var(x) == 0:
            return 0.0

        df = pd.DataFrame({"x": x, "y": y})
        means = df.groupby("y")["x"].mean()
        counts = df.groupby("y")["x"].count()

        grand_mean = x.mean()
        num = np.sum(counts * (means - grand_mean) ** 2)
        den = np.sum((x - grand_mean) ** 2)

        if den == 0:
            return 0.0
        return float(num / den)
