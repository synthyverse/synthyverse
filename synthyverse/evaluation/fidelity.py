# Third-party notice: selected fidelity metrics include upstream-derived code.
# See THIRD_PARTY_NOTICES.md for attribution, NOTICE, and modification details.
import numpy as np
import pandas as pd
import os
import re
import warnings
from collections import Counter
from itertools import combinations
from math import ceil

# from geomloss import SamplesLoss
# import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score

from scipy.stats import (
    spearmanr,
    pearsonr,
    chi2_contingency,
)

from .base import BaseMetric
from .ml import cv_ml_task, resolve_model_name
from .distances import (
    FastGowerNN,
    js_robust,
    kld,
    knn_distances,
    ksd,
    l1,
    l2 as l2_distance,
    pairwise_l1,
    tvd,
    wsd,
)
from .preprocessing import (
    bin_numerical_columns,
    empirical_discrete_distribution,
    fast_gower_transform,
    gower_like_transform,
)


def _metric_key_part(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


class ClassifierTwoSampleTest(BaseMetric):
    """ROCAUC score of a classifier that distinguishes synthetic from real data.

    Lower scores indicate better quality synthetic data (harder to distinguish from real).

    Args:
        random_state (int): Random seed for reproducibility. Default: 0.
        model_name (str): Classifier family. Supported values include "xgboost",
            "randomforest", "decisiontree", "linearregression", and "svm",
            including some common aliases. Every model except for XGBoost is a scikit-learn classifier. Default: "xgboost".
        model_params (dict): Classifier parameters passed to the selected estimator.
            If None, XGBoost uses {"n_estimators": 500, "early_stopping_rounds": 30}
            and other models use no extra parameters.
        nfold (int): Number of cross-validation folds. Default: 3.

    Outputs:
        "c2st.auc"; lower scores indicate better fidelity.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import ClassifierTwoSampleTest
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = ClassifierTwoSampleTest(
        ...     random_state=42
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_syn, discrete_features=discrete_features)
    """

    name = "c2st"

    def __init__(
        self,
        model_name: str = "xgboost",
        model_params: dict = None,
        nfold: int = 3,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.nfold = int(nfold)
        if self.nfold < 2:
            raise ValueError("nfold must be >= 2.")
        self.model_name = model_name
        if model_params is None:
            self.model_params = (
                {"n_estimators": 500, "early_stopping_rounds": 30}
                if resolve_model_name(model_name) == "xgboost"
                else {}
            )
        else:
            self.model_params = model_params.copy()

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        x_train = pd.concat([X, X_syn], ignore_index=True).copy()
        y_train = pd.concat(
            [pd.Series([0] * len(X)), pd.Series([1] * len(X_syn))],
            ignore_index=True,
        )
        minority_class_size = min(len(X), len(X_syn))
        if minority_class_size < 2:
            raise ValueError(
                "ClassifierTwoSampleTest requires at least two real and two "
                "synthetic rows for stratified cross-validation."
            )
        nfold = min(self.nfold, minority_class_size)

        score = cv_ml_task(
            x_train,
            y_train,
            discrete_features,
            task="binary",
            model_name=self.model_name,
            model_params=self.model_params,
            random_state=self.random_state,
            score_fn="auc",
            nfold=nfold,
        )["auc"]

        return {
            f"{self.name}.auc": score,
        }


class AlphaPrecisionBetaRecall(BaseMetric):
    """Integrated alpha-precision / beta-coverage scores.

    Computes alpha-precision and beta-coverage curves over a grid of alpha
    values, then returns the integrated Delta-precision-alpha and
    Delta-coverage-beta scores from Alaa et al. (2022).

    Paper: "How faithful is your synthetic data? sample-level metrics for evaluating and auditing generative models" by Alaa et al. (2022).

    Based on the implementation from the synthcity Python library: https://github.com/vanderschaarlab/synthcity/.

    Args:
        k (int): Number of nearest neighbors to use in Beta-Recall. Default: 5.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        Integrated scores under
        ``"alphaprecisionbetarecall.precision"`` and
        ``"alphaprecisionbetarecall.recall"``.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import AlphaPrecisionBetaRecall
        >>>
        >>> # Prepare data
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = AlphaPrecisionBetaRecall(
        ...     k=5
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_real, X_syn, discrete_features=discrete_features)
    """

    name = "alphaprecisionbetarecall"

    def __init__(self, k: int = 5, random_state: int = 0):
        super().__init__(random_state=random_state)
        self.k = int(k)
        if self.k < 1:
            raise ValueError("AlphaPrecisionBetaRecall requires k >= 1.")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        if self.k >= len(X):
            raise ValueError(
                "AlphaPrecisionBetaRecall requires k to be smaller than the "
                f"number of real rows ({self.k} >= {len(X)})."
            )
        data = gower_like_transform(
            {"rd": X, "sd": X_syn},
            reference_data=X,
            discrete_features=discrete_features,
            categorical_fit_data=[X, X_syn],
        )

        x_rd = data["rd"]
        x_sd = data["sd"]
        nn_data = fast_gower_transform(
            {"rd": X, "sd": X_syn},
            reference_data=X,
            discrete_features=discrete_features,
            categorical_fit_data=[X, X_syn],
        )
        rd_metric = nn_data["rd"]
        sd_metric = nn_data["sd"]
        emb_center = np.mean(x_rd, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        # Radii = np.quantile(np.sqrt(np.sum((x_rd - emb_center) ** 2, axis=1)), alphas)
        # Use L1 distance in the mixed-type feature space.
        Radii = np.quantile(l1(x_rd, emb_center), alphas)

        synth_center = np.mean(x_sd, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        # synth_to_center = np.sqrt(np.sum((x_sd - emb_center) ** 2, axis=1))
        # Use L1 distance in the mixed-type feature space.
        synth_to_center = l1(x_sd, emb_center)

        nbrs_real = FastGowerNN(categorical_cols=discrete_features).fit(rd_metric)
        real_to_real, _ = nbrs_real.kneighbors(X=None, k=self.k)

        nbrs_synth = FastGowerNN(categorical_cols=discrete_features).fit(sd_metric)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(rd_metric, k=1)

        real_to_real = real_to_real[:, self.k - 1].reshape(-1)
        real_to_synth = real_to_synth.reshape(-1)
        real_to_synth_args = real_to_synth_args.reshape(-1)

        real_synth_closest = x_sd[real_to_synth_args]

        # real_synth_closest_d = np.sqrt(
        #     np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        # )
        # Use L1 distance in the mixed-type feature space.
        real_synth_closest_d = l1(real_synth_closest, synth_center)

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

        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)

        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)

        return {
            f"{self.name}.precision": float(Delta_precision_alpha),
            f"{self.name}.recall": float(Delta_coverage_beta),
        }


class PRDC(BaseMetric):
    """Precision, Recall, Density, and Coverage for tabular synthetic data.

    Paper: "Reliable fidelity and diversity metrics for generative models" by Naeem et al. (2020).

    Args:
        k (int): Number of nearest neighbours used to estimate each
            sample's manifold radius. Default: 5.
        n_jobs (int): Number of parallel jobs for sklearn pairwise distances.
            Default: -1.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        "prdc.precision", "prdc.recall", "prdc.density", and
        "prdc.coverage".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import PRDC
        >>>
        >>> metric = PRDC(k=5)
        >>> results = metric.evaluate(X_train, X_syn, discrete_features=["category_col"])
    """

    name = "prdc"

    def __init__(
        self,
        k: int = 5,
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.nearest_k = int(k)
        self.n_jobs = int(n_jobs)

        if self.nearest_k < 1:
            raise ValueError("k must be >= 1.")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        if len(X) <= self.nearest_k or len(X_syn) <= self.nearest_k:
            raise ValueError(
                "PRDC requires k to be smaller than both the real and "
                "synthetic sample sizes."
            )

        data = gower_like_transform(
            {"real": X, "syn": X_syn},
            reference_data=X,
            discrete_features=discrete_features,
            categorical_fit_data=[X, X_syn],
        )

        real_features = data["real"]
        syn_features = data["syn"]
        if not np.isfinite(real_features).all() or not np.isfinite(syn_features).all():
            raise ValueError("PRDC requires finite metric features.")

        real_radii = knn_distances(real_features, self.nearest_k, n_jobs=self.n_jobs)
        syn_radii = knn_distances(syn_features, self.nearest_k, n_jobs=self.n_jobs)
        distance_real_syn = pairwise_l1(real_features, syn_features, n_jobs=self.n_jobs)

        real_manifold_membership = distance_real_syn < np.expand_dims(real_radii, 1)
        syn_manifold_membership = distance_real_syn < np.expand_dims(syn_radii, 0)

        precision = real_manifold_membership.any(axis=0).mean()
        recall = syn_manifold_membership.any(axis=1).mean()
        density = real_manifold_membership.sum(axis=0).mean() / float(self.nearest_k)
        coverage = (distance_real_syn.min(axis=1) < real_radii).mean()

        return {
            f"{self.name}.precision": float(precision),
            f"{self.name}.recall": float(recall),
            f"{self.name}.density": float(density),
            f"{self.name}.coverage": float(coverage),
        }


# class Wasserstein:
#     """Multivariate Wasserstein distance between real and synthetic samples.

#     Sinkhorn approximation of the Wasserstein-1 distance using a Gower-like cost function.

#     Args:
#         discrete_features (list): List of discrete/categorical feature names. Default: [].
#         blur (float): Entropic regularization scale passed to GeomLoss
#             ``SamplesLoss``. Smaller values are closer to exact optimal
#             transport but can be slower or less stable. Default: 0.001.
#         scaling (float): GeomLoss epsilon-scaling ratio. Default: 0.5.
#         debias (bool): Whether to use the debiased Sinkhorn divergence form,
#             which returns zero for identical empirical distributions. Default: True.
#         backend (str): GeomLoss backend. The default "online" backend streams
#             pairwise costs through KeOps and avoids materializing the full
#             sample-by-sample cost matrix. Use "tensorized" only for small
#             datasets or debugging. Default: "online".
#     Example:
#         >>> import pandas as pd
#         >>> from synthyverse.evaluation import Wasserstein
#         >>>
#         >>> metric = Wasserstein(discrete_features=["category_col"])
#         >>> results = metric.evaluate(X_train, X_syn)
#     """

#     name = "wasserstein"

#     def __init__(
#         self,
#         discrete_features: list = None,
#         blur: float = 0.001,
#         scaling: float = 0.5,
#         debias: bool = True,
#         backend: str = "online",
#     ):
#         super().__init__()
#         self.discrete_features = (
#             list(discrete_features) if discrete_features is not None else []
#         )
#         self.blur = float(blur)
#         self.scaling = float(scaling)
#         self.debias = bool(debias)
#         self.backend = backend

#         if self.blur <= 0.0:
#             raise ValueError("blur must be > 0 for GeomLoss SamplesLoss.")
#         if self.scaling <= 0.0 or self.scaling >= 1.0:
#             raise ValueError("scaling must be in (0, 1).")
#         if self.backend not in {"online", "multiscale", "tensorized"}:
#             raise ValueError(
#                 'backend must be one of "online", "multiscale", or "tensorized".'
#             )

#     def evaluate(self, X_train: pd.DataFrame, X_syn: pd.DataFrame):
#         """Evaluate synthetic data using multivariate Wasserstein distance.

#         Args:
#             X_train: Real training data as a pandas DataFrame.
#             X_syn: Synthetic data as a pandas DataFrame.

#         Returns:
#             dict: Dictionary with key:
#                 - "wasserstein.w1": Wasserstein-1 distance with L1 ground cost
#         """
#         if len(X_train) == 0 or len(X_syn) == 0:
#             raise ValueError("Wasserstein requires non-empty X_train and X_syn.")

#         data = gower_like_transform(
#             {"train": X_train, "syn": X_syn},
#             reference_data=X_train,
#             discrete_features=self.discrete_features,
#             categorical_fit_data=[X_train, X_syn],
#         )

#         x_train = data["train"]
#         x_syn = data["syn"]
#         if not np.isfinite(x_train).all() or not np.isfinite(x_syn).all():
#             raise ValueError("Wasserstein requires finite metric features.")

#         return {
#             f"{self.name}.w1": float(self._samples_loss(x_train, x_syn)),
#         }

#     def _samples_loss(self, x_train: np.ndarray, x_syn: np.ndarray) -> float:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         tensor_kwargs = {"dtype": torch.float32, "device": device}

#         x_train_t = torch.as_tensor(np.ascontiguousarray(x_train), **tensor_kwargs)
#         x_syn_t = torch.as_tensor(np.ascontiguousarray(x_syn), **tensor_kwargs)

#         loss = SamplesLoss(
#             loss="sinkhorn",
#             p=1,
#             blur=self.blur,
#             scaling=self.scaling,
#             cost=self._l1_cost(),
#             debias=self.debias,
#             backend=self.backend,
#         )

#         with torch.no_grad():
#             value = loss(x_train_t, x_syn_t)
#         return value.detach().cpu().item()

#     @staticmethod
#     def _l1_cost_matrix(x, y):
#         return (x[:, :, None, :] - y[:, None, :, :]).abs().sum(dim=-1)

#     def _l1_cost(self):
#         if self.backend == "tensorized":
#             return self._l1_cost_matrix
#         return "Sum(Abs(X-Y))"


class ShapeTrend(BaseMetric):
    """Low-level implementation of the Column Shape and Column Pair Trend scores
    from the SDMetrics library (https://docs.sdv.dev/sdmetrics/).


    Args:
        numerical_correlation (str): Correlation method for numerical-numerical pairs.
            One of "spearman" or "pearson". Default: "pearson".
        n_bins_numerical (int): Number of bins used to discretize numerical
            features for mixed-pair trends. Must be >= 2. Default: 20.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        "shapetrend.<split>.shape" and "shapetrend.<split>.trend".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import ShapeTrend
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = ShapeTrend()
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X=X_train, X_syn=X_syn, discrete_features=discrete_features)
        >>> results_with_test = metric.evaluate(
        ...     X=X_train, X_syn=X_syn, X_test=X_test, discrete_features=discrete_features
        ... )
    """

    name = "shapetrend"

    def __init__(
        self,
        numerical_correlation: str = "pearson",
        n_bins_numerical: int = 20,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.numerical_correlation = numerical_correlation.lower()
        self.n_bins_numerical = int(n_bins_numerical)

        if self.numerical_correlation not in {"pearson", "spearman"}:
            raise ValueError('numerical_correlation must be "pearson" or "spearman"')
        if self.n_bins_numerical < 2:
            raise ValueError("n_bins_numerical must be >= 2")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        self.discrete_features = discrete_features
        result = {}
        splits = [("train", X, X_syn)]
        if X_test is not None:
            splits.append(("test", X_test, X_syn))

        for split, real, syn in splits:
            scores = self._evaluate_split(real, syn)
            result.update({f"{self.name}.{split}.{k}": v for k, v in scores.items()})
        return result

    def _evaluate_split(self, real: pd.DataFrame, syn: pd.DataFrame) -> dict:
        rd = real.copy()
        sd = syn[real.columns].copy()
        cols = rd.columns.tolist()
        numerical_features = [c for c in cols if c not in self.discrete_features]
        categorical_features = [c for c in cols if c in self.discrete_features]

        shape_scores = []
        for col in cols:
            if col in self.discrete_features:
                rd_dist, sd_dist = empirical_discrete_distribution(
                    rd[col].to_numpy(),
                    sd[col].to_numpy(),
                    dropna=True,
                )
                shape_scores.append(1.0 - tvd(rd_dist, sd_dist))
            else:
                shape_scores.append(1.0 - ksd(rd[col].to_numpy(), sd[col].to_numpy()))
        shape = float(np.mean(shape_scores)) if shape_scores else np.nan

        rd_binned, sd_binned = rd, sd
        if numerical_features and categorical_features:
            rd_binned, sd_binned = bin_numerical_columns(
                rd,
                sd,
                numerical_features,
                self.n_bins_numerical,
            )

        trend_scores = []
        for ci, cj in combinations(cols, 2):
            trend_scores.append(self._trend_score(rd, sd, rd_binned, sd_binned, ci, cj))

        if trend_scores:
            trend = float(np.mean(trend_scores))
        else:
            warnings.warn(
                "ShapeTrend trend is undefined for fewer than two columns.",
                RuntimeWarning,
                stacklevel=2,
            )
            trend = np.nan
        return {
            "shape": float(shape),
            "trend": float(trend),
        }

    def _trend_score(
        self,
        rd: pd.DataFrame,
        sd: pd.DataFrame,
        rd_binned: pd.DataFrame,
        sd_binned: pd.DataFrame,
        c1: str,
        c2: str,
    ) -> float:
        c1_cat = c1 in self.discrete_features
        c2_cat = c2 in self.discrete_features

        if not c1_cat and not c2_cat:
            corr_rd = self._num_corr(rd[c1], rd[c2])
            corr_sd = self._num_corr(sd[c1], sd[c2])
            return 1.0 - abs(corr_sd - corr_rd) / 2.0

        real_x = rd[c1] if c1_cat else rd_binned[c1]
        syn_x = sd[c1] if c1_cat else sd_binned[c1]
        real_y = rd[c2] if c2_cat else rd_binned[c2]
        syn_y = sd[c2] if c2_cat else sd_binned[c2]

        return 1.0 - self._contingency_tvd(real_x, real_y, syn_x, syn_y)

    def _num_corr(self, s1: pd.Series, s2: pd.Series) -> float:
        if s1.nunique() < 2 or s2.nunique() < 2:
            return 0.0

        if self.numerical_correlation == "spearman":
            corr = spearmanr(s1, s2).statistic
        else:
            corr = pearsonr(s1, s2).statistic

        return float(corr)

    def _contingency_tvd(
        self,
        real_x: pd.Series,
        real_y: pd.Series,
        syn_x: pd.Series,
        syn_y: pd.Series,
    ) -> float:
        row_order = pd.Index(pd.concat([real_x, syn_x], ignore_index=True).unique())
        col_order = pd.Index(pd.concat([real_y, syn_y], ignore_index=True).unique())

        real_table = pd.crosstab(real_x, real_y, normalize=True)
        syn_table = pd.crosstab(syn_x, syn_y, normalize=True)
        real_table = real_table.reindex(
            index=row_order, columns=col_order, fill_value=0.0
        )
        syn_table = syn_table.reindex(
            index=row_order, columns=col_order, fill_value=0.0
        )
        return tvd(
            real_table.to_numpy().ravel(),
            syn_table.to_numpy().ravel(),
        )


class Marginals(BaseMetric):
    """Per-column distributional distance between real and synthetic marginals.

    Computes distance metrics for each column independently and returns
    the average distances over numerical and categorical features separately.
    Numerical distance functions: Wasserstein (wsd), Jensen-Shannon distance
    (jsd), Kolmogorov-Smirnov statistic (ks), Total Variation distance (tvd),
    and Kullback-Leibler divergence (kld).
    Categorical distance functions: Jensen-Shannon distance (jsd), Total
    Variation distance (tvd), and Kullback-Leibler divergence (kld).
    For histogram-based metrics (jsd, tvd, kld) on numerical features, values are
    discretized into equal-width bins before comparison.

    Lower scores indicate better fidelity to the real marginals.

    Args:
        n_bins_numerical (int): Number of equal-width bins used when discretizing
            numerical features for jsd/tvd/kld. Must be >= 2. Default: 20.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        "marginals.<split>.num.<distance>" and
        "marginals.<split>.cat.<distance>".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import Marginals
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = Marginals()
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X=X_train, X_syn=X_syn, X_test=X_test, discrete_features=discrete_features)
    """

    name = "marginals"
    numerical_distances = ("wsd", "jsd", "ks", "tvd", "kld")
    categorical_distances = ("jsd", "tvd", "kld")

    def __init__(
        self,
        n_bins_numerical: int = 20,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.n_bins_numerical = int(n_bins_numerical)
        if self.n_bins_numerical < 2:
            raise ValueError("n_bins_numerical must be >= 2")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        self.discrete_features = discrete_features
        self._bin_reference = X
        result = {}
        splits = [("train", X, X_syn)]
        if X_test is not None:
            splits.append(("test", X_test, X_syn))

        for split, real, syn in splits:
            scores = self._evaluate_split(real, syn)
            result.update({f"{self.name}.{split}.{k}": v for k, v in scores.items()})
        return result

    def _evaluate_split(self, real: pd.DataFrame, syn: pd.DataFrame) -> dict:
        rd = real.copy()
        sd = syn[real.columns].copy()

        numerical_features = [c for c in rd.columns if c not in self.discrete_features]

        dist_func = {
            "wsd": wsd,
            "jsd": js_robust,
            "kld": kld,
            "ks": ksd,
            "tvd": tvd,
        }

        rd_binned, sd_binned = bin_numerical_columns(
            rd,
            sd,
            numerical_features,
            min(len(self._bin_reference), self.n_bins_numerical),
            skip_constant=True,
            fit_data=self._bin_reference,
        )

        result = {}
        for distance in self.numerical_distances:
            num = []
            for col in numerical_features:
                if distance in ["jsd", "tvd", "kld"]:
                    rd_dist, sd_dist = empirical_discrete_distribution(
                        rd_binned[col].to_numpy(),
                        sd_binned[col].to_numpy(),
                    )
                    num.append(dist_func[distance](sd_dist, rd_dist))
                else:
                    num.append(
                        dist_func[distance](
                            sd[col].to_numpy(),
                            rd[col].to_numpy(),
                        )
                    )
            result[f"num.{distance}"] = float(np.mean(num)) if len(num) else np.nan

        for distance in self.categorical_distances:
            cat = []
            for col in self.discrete_features:
                rd_dist, sd_dist = empirical_discrete_distribution(
                    rd[col].to_numpy(),
                    sd[col].to_numpy(),
                )
                cat.append(dist_func[distance](sd_dist, rd_dist))
            result[f"cat.{distance}"] = float(np.mean(cat)) if len(cat) else np.nan

        return result


class Correlations(BaseMetric):
    """Pairwise correlation matrix difference between real and synthetic data.

    Builds a full correlation matrix for both real and synthetic data and
    returns the L2 norm of their absolute difference. Correlation type is
    chosen automatically per feature pair: Spearman/Pearson for
    numerical-numerical, Cramer's V for categorical-categorical, and the
    correlation ratio (eta-squared) for mixed pairs.

    Lower scores indicate better preservation of feature dependencies.

    Args:
        numerical_correlation (str): Correlation method for numerical-numerical pairs.
            One of "spearman" or "pearson". Default: "pearson".
        img_save_path (str, optional): Directory where correlation matrix plots
            will be saved. If a file path is provided, its basename is used as a
            prefix. Default: None.
        file_format (str): Image file format passed to matplotlib. Default: "png".
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        "correlations.<split>.l2" and optional
        "correlations.<split>.img_paths".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import Correlations
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = Correlations(
        ...     numerical_correlation="spearman",
        ...     img_save_path="results/correlations",
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X=X_train, X_syn=X_syn, discrete_features=discrete_features)
        >>> results_with_test = metric.evaluate(
        ...     X=X_train, X_syn=X_syn, X_test=X_test, discrete_features=discrete_features
        ... )
    """

    name = "correlations"

    def __init__(
        self,
        numerical_correlation: str = "pearson",
        img_save_path: str = None,
        file_format: str = "png",
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.numerical_correlation = numerical_correlation.lower()
        if self.numerical_correlation not in {"pearson", "spearman"}:
            raise ValueError('numerical_correlation must be "pearson" or "spearman"')
        self.img_save_path = img_save_path
        self.file_format = file_format.lower().lstrip(".")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        self.discrete_features = discrete_features
        result = {}
        splits = [("train", X, X_syn)]
        if X_test is not None:
            splits.append(("test", X_test, X_syn))

        for split, real, syn in splits:
            scores = self._evaluate_split(real, syn, split)
            result.update({f"{self.name}.{split}.{k}": v for k, v in scores.items()})
        return result

    def _evaluate_split(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        split: str,
    ) -> dict:
        rd = real.copy()
        sd = syn[real.columns].copy()

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
                if i == j:
                    C_rd[i, j] = 1.0
                    C_sd[i, j] = 1.0
                    continue

                corr_type = self._get_corr_type(ci, cj)

                C_rd[i, j] = self._get_corr(rd[ci], rd[cj], corr_type)
                C_sd[i, j] = self._get_corr(sd[ci], sd[cj], corr_type)

        diff = np.abs(C_rd - C_sd)
        score = l2_distance(diff)

        result = {
            "l2": score,
        }

        if self.img_save_path:
            result["img_paths"] = self._save_corr_plots(C_rd, C_sd, diff, cols, split)

        return result

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
        data = pd.concat([s1, s2], axis=1).dropna()
        if (
            len(data) < 2
            or data.iloc[:, 0].nunique() <= 1
            or data.iloc[:, 1].nunique() <= 1
        ):
            return 0.0

        if self.numerical_correlation == "spearman":
            corr = spearmanr(data.iloc[:, 0], data.iloc[:, 1]).statistic
        elif self.numerical_correlation == "pearson":
            corr = pearsonr(data.iloc[:, 0], data.iloc[:, 1]).statistic
        else:
            raise ValueError(
                f"Invalid numerical correlation: {self.numerical_correlation}"
            )
        return float(corr) if np.isfinite(corr) else 0.0

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

    def _save_corr_plots(
        self,
        C_rd: np.ndarray,
        C_sd: np.ndarray,
        diff: np.ndarray,
        cols: list,
        split: str,
    ) -> list:
        save_dir, prefix, file_format = self._resolve_img_save_path()
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"{prefix}{split}_matrices.{file_format}")

        self._plot_corr_matrices(C_rd, C_sd, diff, cols, path)

        return [os.path.abspath(path)]

    def _resolve_img_save_path(self) -> tuple[str, str, str]:
        path = os.fspath(self.img_save_path)
        root, ext = os.path.splitext(path)

        if ext:
            save_dir = os.path.dirname(path) or "."
            prefix = f"{os.path.basename(root)}_"
            file_format = ext.lstrip(".")
        else:
            save_dir = path
            prefix = "correlations_"
            file_format = self.file_format

        return save_dir, prefix, file_format

    def _plot_corr_matrices(
        self,
        C_rd: np.ndarray,
        C_sd: np.ndarray,
        diff: np.ndarray,
        cols: list,
        path: str,
    ):
        fig, axes = plt.subplots(
            1, 3, figsize=self._heatmap_figsize(cols, 3), constrained_layout=True
        )
        for i, (ax, matrix, title) in enumerate(
            zip(axes[:2], [C_rd, C_sd], ["Real", "Synthetic"])
        ):
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                xticklabels=cols,
                yticklabels=cols if i == 0 else False,
                cbar=False,
            )
            ax.set_title(title)
            self._format_heatmap_axis(ax)

        sns.heatmap(
            diff,
            ax=axes[2],
            cmap="viridis",
            vmin=0,
            square=True,
            xticklabels=cols,
            yticklabels=False,
            cbar=False,
        )
        axes[2].set_title("Absolute Difference")
        self._format_heatmap_axis(axes[2])

        fig.colorbar(
            axes[1].collections[0],
            ax=axes[:2].tolist(),
            shrink=0.8,
            ticks=np.linspace(-1, 1, 5),
        )
        fig.colorbar(axes[2].collections[0], ax=axes[2], shrink=0.8)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _heatmap_figsize(self, cols: list, n_panels: int) -> tuple[float, float]:
        size = max(5.0, min(18.0, 0.45 * len(cols) + 2.5))
        return (size * n_panels, size)

    def _format_heatmap_axis(self, ax):
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")


class ARM(BaseMetric):
    """Association Rule Mining preservation using Apriori.

    Mines association rules in the real training data and synthetic data, then
    compares exact rule matches. Rules are represented as ``antecedent ->
    consequent`` over tabular ``column=value`` items. Numerical columns are
    discretized into equal-width bins fitted on the real training data before
    mining.

    Higher precision and recall indicate better preservation of association
    rules found in the real data.

    Args:
        min_support (float or int): Minimum itemset support for Apriori. Floats
            in (0, 1] are interpreted as a fraction of rows; integers are
            interpreted as absolute row counts. Default: 0.1.
        min_confidence (float): Minimum confidence for generated rules. Default: 0.8.
        n_bins_numerical (int): Number of equal-width bins for numerical features.
            Must be >= 2. Default: 5.
        max_itemset_size (int): Maximum frequent itemset size mined by Apriori.
            Must be >= 2. Default: 2.
        print_rule_differences (bool): Whether to print missed and hallucinated
            rules after evaluation. Default: True.
        max_rules_to_print (int or None): Maximum number of missed and
            hallucinated rules to print per category. None prints all. Default: 25.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        "arm.precision", "arm.recall", "arm.n_rules_real", and
        "arm.n_rules_syn".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import ARM
        >>>
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> metric = ARM(
        ...     min_support=0.05,
        ...     min_confidence=0.7,
        ... )
        >>>
        >>> results = metric.evaluate(X_real, X_syn, discrete_features=discrete_features)
    """

    name = "arm"

    def __init__(
        self,
        min_support=0.1,
        min_confidence: float = 0.8,
        n_bins_numerical: int = 5,
        max_itemset_size: int = 2,
        print_rule_differences: bool = True,
        max_rules_to_print: int = 25,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.min_support = min_support
        self.min_confidence = float(min_confidence)
        self.n_bins_numerical = int(n_bins_numerical)
        self.max_itemset_size = int(max_itemset_size)
        self.print_rule_differences = bool(print_rule_differences)
        self.max_rules_to_print = max_rules_to_print

        if isinstance(self.min_support, float) and not (0.0 < self.min_support <= 1.0):
            raise ValueError("min_support as a float must be in (0, 1]")
        if isinstance(self.min_support, int) and self.min_support < 1:
            raise ValueError("min_support as an int must be >= 1")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if self.n_bins_numerical < 2:
            raise ValueError("n_bins_numerical must be >= 2")
        if self.max_itemset_size < 2:
            raise ValueError("max_itemset_size must be >= 2")
        if self.max_rules_to_print is not None and self.max_rules_to_print < 1:
            raise ValueError("max_rules_to_print must be >= 1 or None")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        self.discrete_features = discrete_features
        rd, sd = self._prepare_data(X, X_syn)
        real_rules = self._mine_rules(rd)
        syn_rules = self._mine_rules(sd)

        shared_rules = real_rules.intersection(syn_rules)
        precision = len(shared_rules) / len(syn_rules) if syn_rules else np.nan
        recall = len(shared_rules) / len(real_rules) if real_rules else np.nan

        missed_rules = real_rules - syn_rules
        hallucinated_rules = syn_rules - real_rules
        if self.print_rule_differences:
            self._print_rule_differences(missed_rules, hallucinated_rules)

        return {
            f"{self.name}.precision": float(precision),
            f"{self.name}.recall": float(recall),
            f"{self.name}.n_rules_real": int(len(real_rules)),
            f"{self.name}.n_rules_syn": int(len(syn_rules)),
        }

    def _prepare_data(
        self, X_train: pd.DataFrame, X_syn: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rd = X_train.copy()
        sd = X_syn.copy()

        numerical_features = [c for c in rd.columns if c not in self.discrete_features]
        if numerical_features:
            n_bins = min(len(rd), self.n_bins_numerical)
            if n_bins >= 2:
                rd, sd = bin_numerical_columns(
                    rd,
                    sd,
                    numerical_features,
                    n_bins,
                )
            else:
                rd[numerical_features] = 0
                sd[numerical_features] = 0

        for col in rd.columns:
            rd[col] = self._canonicalize_column(rd[col], col in numerical_features)
            sd[col] = self._canonicalize_column(sd[col], col in numerical_features)

        return rd, sd

    def _canonicalize_column(self, s: pd.Series, is_numerical: bool) -> pd.Series:
        values = s.astype("object").where(s.notna(), "__nan__")
        if is_numerical:
            return values.map(
                lambda x: "__nan__" if x == "__nan__" else f"bin_{int(x)}"
            )
        return values.map(str)

    def _mine_rules(self, X: pd.DataFrame) -> set:
        transactions = self._to_transactions(X)
        itemset_support = self._apriori(transactions)
        rules = set()

        for itemset, itemset_count in itemset_support.items():
            if len(itemset) < 2:
                continue

            items = sorted(itemset)
            for antecedent_size in range(1, len(items)):
                for antecedent in combinations(items, antecedent_size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_count = itemset_support.get(antecedent, 0)
                    if antecedent_count == 0:
                        continue

                    confidence = itemset_count / antecedent_count
                    if confidence >= self.min_confidence:
                        rules.add(
                            (tuple(sorted(antecedent)), tuple(sorted(consequent)))
                        )

        return rules

    def _print_rule_differences(
        self,
        missed_rules: set,
        hallucinated_rules: set,
    ):
        print(f"[arm] Missed rules (real only): {len(missed_rules)}")
        self._print_rules(missed_rules)
        print(f"[arm] Hallucinated rules (synthetic only): {len(hallucinated_rules)}")
        self._print_rules(hallucinated_rules)

    def _print_rules(self, rules: set):
        sorted_rules = self._sort_rules(rules)
        if self.max_rules_to_print is not None:
            rules_to_print = sorted_rules[: self.max_rules_to_print]
        else:
            rules_to_print = sorted_rules

        for rule in rules_to_print:
            print(f"[arm]   {self._format_rule(rule)}")

        n_hidden = len(sorted_rules) - len(rules_to_print)
        if n_hidden > 0:
            print(f"[arm]   ... {n_hidden} more")

    def _sort_rules(self, rules: set) -> list:
        return sorted(rules, key=self._format_rule)

    def _format_rule(self, rule: tuple) -> str:
        antecedent, consequent = rule
        lhs = " AND ".join(self._format_item(item) for item in antecedent)
        rhs = " AND ".join(self._format_item(item) for item in consequent)
        return f"{lhs} -> {rhs}"

    def _format_item(self, item: tuple) -> str:
        col, value = item
        return f"{col}={value}"

    def _to_transactions(self, X: pd.DataFrame) -> list[frozenset]:
        transactions = []
        for _, row in X.iterrows():
            transaction = frozenset((col, row[col]) for col in X.columns)
            transactions.append(transaction)
        return transactions

    def _apriori(self, transactions: list[frozenset]) -> dict:
        min_count = self._min_support_count(len(transactions))
        item_counts = Counter()
        for transaction in transactions:
            item_counts.update(frozenset([item]) for item in transaction)

        frequent_level = {
            itemset: count
            for itemset, count in item_counts.items()
            if count >= min_count
        }
        frequent_itemsets = dict(frequent_level)

        k = 2
        while frequent_level and k <= self.max_itemset_size:
            candidates = self._generate_candidates(set(frequent_level.keys()), k)
            candidate_counts = Counter()
            for transaction in transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1

            frequent_level = {
                itemset: count
                for itemset, count in candidate_counts.items()
                if count >= min_count
            }
            frequent_itemsets.update(frequent_level)
            k += 1

        return frequent_itemsets

    def _generate_candidates(self, previous_level: set, k: int) -> set:
        candidates = set()
        previous_list = sorted(previous_level, key=lambda x: tuple(sorted(x)))
        for i, itemset_a in enumerate(previous_list):
            for itemset_b in previous_list[i + 1 :]:
                candidate = itemset_a | itemset_b
                if len(candidate) != k:
                    continue
                if all(
                    frozenset(subset) in previous_level
                    for subset in combinations(candidate, k - 1)
                ):
                    candidates.add(candidate)
        return candidates

    def _min_support_count(self, n_rows: int) -> int:
        if isinstance(self.min_support, float):
            return max(1, ceil(self.min_support * n_rows))
        return int(self.min_support)


class NMI(BaseMetric):
    """Pairwise normalized mutual information preservation.

    Paper: "A Sobering Look at Tabular Data Generation via Probabilistic Circuits" by Scassola et al. (2026).

    Computes normalized mutual information (NMI) for every feature pair in the
    real and synthetic data, then returns the weighted average of the per-pair
    preservation score ``1 - abs(NMI_real - NMI_synthetic)``. Each pair is
    weighted by ``abs(NMI_real + NMI_synthetic)`` and weights are normalized to
    sum to 1.

    Numerical features are discretized into equal-width bins before computing
    NMI. Higher scores indicate better preservation of feature dependencies.

    Args:
        n_bins_numerical (int): Number of equal-width bins used when discretizing
            numerical features. Must be >= 2. Default: 20.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        "nmi.<split>.score".

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import NMI
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = NMI()
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X=X_train, X_syn=X_syn, discrete_features=discrete_features)
        >>> results_with_test = metric.evaluate(
        ...     X=X_train, X_syn=X_syn, X_test=X_test, discrete_features=discrete_features
        ... )
    """

    name = "nmi"

    def __init__(
        self,
        n_bins_numerical: int = 20,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.n_bins_numerical = int(n_bins_numerical)
        if self.n_bins_numerical < 2:
            raise ValueError("n_bins_numerical must be >= 2")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        self.discrete_features = discrete_features
        self._bin_reference = X
        result = {}
        splits = [("train", X, X_syn)]
        if X_test is not None:
            splits.append(("test", X_test, X_syn))

        for split, real, syn in splits:
            scores = self._evaluate_split(real, syn)
            result.update({f"{self.name}.{split}.{k}": v for k, v in scores.items()})
        return result

    def _evaluate_split(self, real: pd.DataFrame, syn: pd.DataFrame) -> dict:
        rd = real.copy()
        sd = syn[real.columns].copy()

        cols = rd.columns.tolist()
        if len(cols) < 2:
            return {"score": np.nan}

        numerical_features = [c for c in cols if c not in self.discrete_features]

        if numerical_features:
            rd, sd = bin_numerical_columns(
                rd,
                sd,
                numerical_features,
                min(len(self._bin_reference), self.n_bins_numerical),
                fit_data=self._bin_reference,
            )

        scores = []
        weights = []
        for i, ci in enumerate(cols):
            for cj in cols[i + 1 :]:
                nmi_rd = self._nmi(rd[ci], rd[cj])
                nmi_sd = self._nmi(sd[ci], sd[cj])
                scores.append(1.0 - abs(nmi_rd - nmi_sd))
                weights.append(abs(nmi_rd + nmi_sd))

        scores = np.array(scores)
        weights = np.array(weights)
        weight_sum = weights.sum()
        if weight_sum == 0.0:
            score = np.mean(scores)
        else:
            score = np.sum((weights / weight_sum) * scores)

        return {"score": float(score)}

    def _nmi(self, s1: pd.Series, s2: pd.Series) -> float:
        x = self._to_codes(s1)
        y = self._to_codes(s2)

        h_x = self._entropy(x)
        h_y = self._entropy(y)
        denom = np.sqrt(h_x * h_y)
        if denom == 0.0:
            return 0.0

        return float(mutual_info_score(x, y) / denom)

    def _to_codes(self, s: pd.Series) -> np.ndarray:
        labels = s.astype("object").where(s.notna(), "__nan__")
        return pd.Categorical(labels).codes

    def _entropy(self, x: np.ndarray) -> float:
        counts = np.bincount(x[x >= 0])
        if counts.sum() == 0:
            return 0.0
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())


class DomainConstraint(BaseMetric):
    """Evaluate boolean pandas expressions on real and synthetic data.

    Args:
        constraint_list (list): List of expressions accepted by DataFrame.eval.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        Per-constraint real/synthetic truth rates and synthetic violation counts.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import DomainConstraint
        >>> from synthyverse.evaluation.eval import TabularMetricEvaluator
        >>>
        >>> # Prepare data
        >>> X_real = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>>
        >>> # Define constraints as boolean DataFrame.eval expressions
        >>> constraints = [
        ...     "age >= 18",
        ...     "systolic_bp > diastolic_bp",
        ... ]
        >>>
        >>> # Create metric
        >>> metric = DomainConstraint(constraint_list=constraints)
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_real, X_syn)
    """

    name = "domainconstraint"

    def __init__(self, constraint_list: list, random_state: int = 0):
        super().__init__(random_state=random_state)
        self.constraint_list = constraint_list

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        result = {}
        for index, constraint in enumerate(self.constraint_list):
            sanitized_constraint = _metric_key_part(
                str(constraint)
                .replace(">=", "_gte_")
                .replace("<=", "_lte_")
                .replace("==", "_eq_")
                .replace("!=", "_neq_")
                .replace("=", "_eq_")
                .replace(">", "_gt_")
                .replace("<", "_lt_")
            )
            constraint_key = f"{index}_{sanitized_constraint}"
            syn_constraint = X_syn.eval(constraint)
            result[f"{self.name}.{constraint_key}_real"] = X.eval(constraint).mean()
            result[f"{self.name}.{constraint_key}_syn"] = syn_constraint.mean()
            result[f"{self.name}.{constraint_key}_syn_n_violations"] = int(
                (~syn_constraint).sum()
            )
        return result


class FeatureWisePlots(BaseMetric):
    """Save one multi-panel plot comparing real training and synthetic data.

    Numerical features are plotted as overlaid density histograms. Discrete
    features are plotted as side-by-side normalized frequency bars.

    Args:
        img_save_path (str): Directory where the feature plot file will be saved.
        bins (int): Number of bins for numerical histograms. Default: 20.
        max_categories (int): Maximum number of categorical levels to show
            before grouping the remainder into "__other__". Default: 20.
        file_format (str): Image file format passed to matplotlib. Default: "png".
        dpi (int): Saved figure resolution. Default: 150.
        random_state (int): Random seed for reproducibility. Default: 0.

    Outputs:
        Absolute path of the saved plot file under
        ``"featurewiseplots.save_dir"``.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import FeatureWisePlots
        >>>
        >>> metric = FeatureWisePlots(
        ...     img_save_path="results/featurewise_plots",
        ... )
        >>> results = metric.evaluate(X_train, X_syn, discrete_features=["category_col"])
    """

    name = "featurewiseplots"

    def __init__(
        self,
        img_save_path: str,
        bins: int = 20,
        max_categories: int = 20,
        file_format: str = "png",
        dpi: int = 150,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.img_save_path = img_save_path
        self.bins = int(bins)
        self.max_categories = int(max_categories)
        self.file_format = file_format.lower().lstrip(".")
        self.dpi = int(dpi)

        if not self.img_save_path:
            raise ValueError("img_save_path must be a non-empty path")
        if self.bins < 1:
            raise ValueError("bins must be >= 1")
        if self.max_categories < 1:
            raise ValueError("max_categories must be >= 1")

    def _evaluate(
        self,
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
        X_test: pd.DataFrame = None,
        discrete_features: list = None,
    ):
        self.discrete_features = discrete_features
        missing = [col for col in X.columns if col not in X_syn.columns]
        if missing:
            raise ValueError(f"X_syn is missing columns: {missing}")

        os.makedirs(self.img_save_path, exist_ok=True)
        cols = X.columns.tolist()

        if not cols:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.text(0.5, 0.5, "No features", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            n_cols = min(3, len(cols))
            n_rows = ceil(len(cols) / n_cols)
            fig_width = 7 * n_cols
            fig_height = 4.8 * n_rows
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_width, fig_height),
                squeeze=False,
            )
            axes = axes.ravel()
            legend_handles = []
            legend_labels = []

            for ax, col in zip(axes, cols):
                self._plot_feature(ax, X, X_syn, col)
                handles, labels = self._pop_axis_legend(ax)
                for handle, label in zip(handles, labels):
                    if label not in legend_labels:
                        legend_handles.append(handle)
                        legend_labels.append(label)

            for ax in axes[len(cols) :]:
                ax.remove()

            if legend_handles:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper center",
                    ncol=len(legend_labels),
                    frameon=False,
                )

        path = self._save_figure(fig)

        return {f"{self.name}.save_dir": path}

    def _plot_feature(
        self,
        ax,
        X_train: pd.DataFrame,
        X_syn: pd.DataFrame,
        col: str,
    ):
        if self._is_categorical(col, X_train[col], X_syn[col]):
            self._plot_categorical(ax, X_train[col], X_syn[col], col)
        else:
            self._plot_numerical(ax, X_train[col], X_syn[col], col)

    def _is_categorical(
        self,
        col: str,
        real: pd.Series,
        syn: pd.Series,
    ) -> bool:
        if col in self.discrete_features:
            return True
        return not (
            pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(syn)
        )

    def _plot_numerical(
        self,
        ax,
        real: pd.Series,
        syn: pd.Series,
        col: str,
    ):
        plot_df = pd.concat(
            [
                pd.DataFrame({"value": real, "dataset": "Real"}),
                pd.DataFrame({"value": syn, "dataset": "Synthetic"}),
            ],
            ignore_index=True,
        ).dropna(subset=["value"])

        if len(plot_df):
            sns.histplot(
                data=plot_df,
                x="value",
                hue="dataset",
                stat="density",
                common_norm=False,
                bins=self.bins,
                kde=all(
                    values["value"].nunique() > 1
                    for _, values in plot_df.groupby("dataset")
                ),
                element="step",
                ax=ax,
            )
        else:
            ax.text(0.5, 0.5, "No non-missing values", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    def _plot_categorical(
        self,
        ax,
        real: pd.Series,
        syn: pd.Series,
        col: str,
    ):
        real_values = self._prepare_categories(real)
        syn_values = self._prepare_categories(syn)
        order = self._category_order(real_values, syn_values)

        if len(order) > self.max_categories:
            visible = set(order[: self.max_categories - 1])
            real_values = real_values.where(real_values.isin(visible), "__other__")
            syn_values = syn_values.where(syn_values.isin(visible), "__other__")
            order = order[: self.max_categories - 1] + ["__other__"]

        plot_df = pd.concat(
            [
                self._category_proportions(real_values, "Real", order),
                self._category_proportions(syn_values, "Synthetic", order),
            ],
            ignore_index=True,
        )

        sns.barplot(
            data=plot_df,
            x="value",
            y="proportion",
            hue="dataset",
            order=order,
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    def _prepare_categories(self, values: pd.Series) -> pd.Series:
        return values.astype("object").where(values.notna(), "__nan__").map(str)

    def _category_order(self, real: pd.Series, syn: pd.Series) -> list:
        counts = pd.concat([real, syn], ignore_index=True).value_counts(dropna=False)
        return counts.index.astype(str).tolist()

    def _category_proportions(
        self,
        values: pd.Series,
        dataset: str,
        order: list,
    ) -> pd.DataFrame:
        proportions = values.value_counts(normalize=True, dropna=False)
        proportions = proportions.reindex(order, fill_value=0.0)
        return pd.DataFrame(
            {
                "value": proportions.index.astype(str),
                "proportion": proportions.to_numpy(dtype=float),
                "dataset": dataset,
            }
        )

    def _pop_axis_legend(self, ax) -> tuple[list, list]:
        legend = ax.get_legend()
        if legend is None:
            return [], []

        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = getattr(legend, "legendHandles", [])
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        return list(handles), labels

    def _save_figure(self, fig) -> str:
        path = os.path.join(self.img_save_path, f"featurewiseplots.{self.file_format}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return os.path.abspath(path)
