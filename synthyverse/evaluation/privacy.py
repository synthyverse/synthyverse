import pandas as pd
import numpy as np
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.stats import gaussian_kde, rankdata
from .ml import ml_task
from .preprocessing import gower_like_transform


def _metric_key_part(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


def lift_at_k(y_true, y_score, k=0.1):
    """
    Return member enrichment in the highest-scored k fraction of records.

    Lift is precision@k divided by the overall positive-class prevalence, so
    a random ranking has expected lift 1.0.
    """
    if not 0 < k <= 1:
        raise ValueError("k must be in the interval (0, 1].")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.nan_to_num(np.asarray(y_score, dtype=float))
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(y_true) == 0:
        return 0.0

    prevalence = np.mean(y_true == 1)
    if prevalence == 0:
        return 0.0

    top_k_count = max(1, int(np.ceil(k * len(y_true))))
    top_k_indices = np.argsort(y_score)[-top_k_count:]
    precision_at_k = np.mean(y_true[top_k_indices] == 1)
    return float(precision_at_k / prevalence)


def tpr_at_fpr(y_true, y_score, max_fpr=0.1):
    """
    Return the highest true positive rate achievable at or below max_fpr.

    This is a thresholded attack metric: it measures member recall while
    constraining the fraction of non-members incorrectly flagged as members.
    """
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in the interval [0, 1].")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.nan_to_num(np.asarray(y_score, dtype=float))
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(y_true) == 0:
        return 0.0
    if not np.any(y_true == 1) or not np.any(y_true == 0):
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= max_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


class DCR:
    """Distance to Closest Record (DCR) privacy metrics.

    Measures whether synthetic records are more often closer to a training record than an independent test record.

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        subsample_test_size (bool): Whether to subsample the training set and synthetic set to the test set size. Prevents biasing DCR due to different sample sizes. If used, multiple iterations of the DCR score are computed and aggregated, to ensure the metric is based on all training and synthetic records. Default: True.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import DCR
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = DCR(
        ...     discrete_features=discrete_features,
        ...     subsample_test_size=True
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_test, X_syn)
    """

    name = "dcr"

    def __init__(
        self,
        discrete_features: list = None,
        subsample_test_size: bool = True,
        random_state: int = 0,
    ):
        super().__init__()
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )
        self.subsample_test_size = subsample_test_size
        self.random_state = random_state

    def evaluate(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, X_syn: pd.DataFrame
    ):
        """Evaluate synthetic data privacy using DCR metrics.

        Args:
            X_train: Training data as a pandas DataFrame.
            X_test: Test data as a pandas DataFrame.
            X_syn: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with keys:
                - "dcr.score": DCR score such that higher scores indicate better privacy
                - "dcr.train": Proportion closer to train
                - "dcr.test": Proportion closer to test
                - "dcr.quantile_002": Proportion closer to train than the 2% test-to-train distance quantile
                - "dcr.quantile_005": Proportion closer to train than the 5% test-to-train distance quantile
                - "dcr.nndr_train": Mean NNDR score from synthetic records to train records
                - "dcr.nndr_train_002": 2% NNDR quantile from synthetic records to train records
                - "dcr.nndr_train_005": 5% NNDR quantile from synthetic records to train records
                - "dcr.nndr_test": Mean NNDR score from synthetic records to test records
                - "dcr.nndr_test_002": 2% NNDR quantile from synthetic records to test records
                - "dcr.nndr_test_005": 5% NNDR quantile from synthetic records to test records
                - "dcr.nndr_ratio": Mean pointwise ratio of each synthetic row's train NNDR score to its test NNDR score
                - "dcr.nndr_ratio_002": 2% quantile of pointwise synthetic train/test NNDR ratios
                - "dcr.nndr_ratio_005": 5% quantile of pointwise synthetic train/test NNDR ratios

        Raises:
            AssertionError: If test set is larger than train set.
        """
        # compare training set to same size synthetic set
        train, test, sd = X_train, X_test, X_syn

        assert len(test) <= len(
            train
        ), "Test set must be smaller than or equal to train size to compute DCR"

        num_rows_subsample = len(test) if self.subsample_test_size else len(train)
        if len(sd) < num_rows_subsample:
            raise ValueError(
                "Synthetic data must contain at least "
                f"{num_rows_subsample} rows to compute DCR with the current "
                "subsampling settings."
            )

        data = gower_like_transform(
            {"train": train, "test": test, "syn": sd.iloc[: len(train)]},
            reference_data=train,
            discrete_features=self.discrete_features,
            categorical_fit_data=[train, test, sd],
        )

        # Optionally subsample train and synthetic data to match the test size.
        if len(data["train"]) < 2 or len(data["test"]) < 2 or num_rows_subsample < 2:
            raise ValueError(
                "DCR with NNDR requires at least two train rows and two test rows "
                "after applying subsampling."
            )
        num_iterations = int(np.ceil(len(data["train"]) / num_rows_subsample))
        metric_values = {}

        rng = np.random.default_rng(self.random_state)

        def choose(x: np.ndarray, n: int) -> np.ndarray:
            return x[rng.choice(len(x), n, replace=False)]

        def nearest_distances(
            query: np.ndarray,
            reference: np.ndarray,
            n_neighbors: int = 1,
        ):
            if len(reference) < n_neighbors:
                raise ValueError(
                    f"Need at least {n_neighbors} reference records to compute "
                    "nearest-neighbor distances."
                )
            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="cityblock",
                n_jobs=-1,
            ).fit(reference)
            distances, _ = nbrs.kneighbors(query)
            if n_neighbors == 1:
                distances = distances.ravel()
            return distances

        def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            return np.divide(
                numerator,
                denominator,
                out=np.ones(len(numerator), dtype=float),
                where=denominator > 0,
            )

        def nearest_distance_ratio(distances: np.ndarray) -> np.ndarray:
            return safe_ratio(distances[:, 0], distances[:, 1])

        def distribution_metrics(prefix: str, values: np.ndarray) -> dict:
            return {
                prefix: np.mean(values),
                f"{prefix}_002": np.quantile(values, 0.02),
                f"{prefix}_005": np.quantile(values, 0.05),
            }

        def collect(metrics: dict):
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(value)

        for _ in range(num_iterations):
            syn_curr = choose(data["syn"], num_rows_subsample)
            train_curr = choose(data["train"], num_rows_subsample)
            test_curr = choose(data["test"], min(len(data["test"]), num_rows_subsample))

            d_s_tr_neighbors = nearest_distances(
                syn_curr,
                train_curr,
                n_neighbors=2,
            )
            d_s_tr = d_s_tr_neighbors[:, 0]
            # align test set size to never exceed subsampled set size
            d_s_te_neighbors = nearest_distances(syn_curr, test_curr, n_neighbors=2)
            d_s_te = d_s_te_neighbors[:, 0]
            d_te_tr = nearest_distances(test_curr, train_curr)

            closer_to_train_ = np.mean(d_s_tr < d_s_te)
            closer_to_test_ = 1 - closer_to_train_

            nndr_train_scores = nearest_distance_ratio(d_s_tr_neighbors)
            nndr_test_scores = nearest_distance_ratio(d_s_te_neighbors)
            nndr_ratio_scores = safe_ratio(nndr_train_scores, nndr_test_scores)

            iteration_metrics = {
                "score": min(1, closer_to_test_ * 2),
                "train": closer_to_train_,
                "test": closer_to_test_,
                "quantile_002": np.mean(d_s_tr < np.quantile(d_te_tr, 0.02)),
                "quantile_005": np.mean(d_s_tr < np.quantile(d_te_tr, 0.05)),
            }
            iteration_metrics.update(
                distribution_metrics("nndr_train", nndr_train_scores)
            )
            iteration_metrics.update(
                distribution_metrics("nndr_test", nndr_test_scores)
            )
            iteration_metrics.update(
                distribution_metrics("nndr_ratio", nndr_ratio_scores)
            )

            collect(iteration_metrics)

        return {
            f"{self.name}.{name}": float(np.mean(values))
            for name, values in metric_values.items()
        }


class AIA:
    """Attribute Inference Attack (AIA) privacy metric.

    Trains a supervised ML model on synthetic data to infer each sensitive
    feature from quasi-identifiers, then evaluates the inferred sensitive
    feature values on real data. Higher performance indicates higher attribute
    disclosure risk.

    Args:
        quasi_identifiers (list): Feature names used by the attacker. If None,
            all non-sensitive features are used. If sensitive_features is also
            None, all other features are used for each target feature.
        sensitive_features (list): Sensitive feature names to infer. If None,
            all features are evaluated as sensitive features.
        discrete_features (list): List of discrete/categorical feature names.
            Used as the authoritative source for classification vs. regression
            targets and quasi-identifier preprocessing.
        model_name (str): Model family. Supported values include "xgboost",
            "randomforest", "decisiontree", "linearregression", and "svm",
            including some common aliases. Every model except for XGBoost is a scikit-learn model. Default: "xgboost".
        model_params (dict): Model parameters passed to the selected estimator.
        random_state (int): Random seed for reproducibility. Default: 0.
    """

    name = "aia"

    def __init__(
        self,
        quasi_identifiers: list = None,
        sensitive_features: list = None,
        discrete_features: list = None,
        model_name: str = "xgboost",
        model_params: dict = None,
        random_state: int = 0,
    ):
        super().__init__()
        self.quasi_identifiers = (
            list(quasi_identifiers) if quasi_identifiers is not None else None
        )
        self.sensitive_features = (
            list(sensitive_features) if sensitive_features is not None else None
        )
        self.discrete_features = (
            list(discrete_features) if discrete_features is not None else []
        )
        self.model_name = model_name
        self.model_params = {} if model_params is None else model_params.copy()
        self.random_state = random_state

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_syn: pd.DataFrame,
    ):
        """Evaluate AIA on real training data using models trained on synthetic data.

        Args:
            X_train: Real training data as a pandas DataFrame.
            X_syn: Synthetic data used to train the attribute inference models.

        Returns:
            dict: Dictionary with per-sensitive-feature attack scores. Keys have
                the form "aia.<sensitive_feature>.<score>".
        """
        self._validate_columns(X_train, X_syn)

        sensitive_features = self._sensitive_features(X_train)
        base_quasi_identifiers = self._base_quasi_identifiers(
            X_train, sensitive_features
        )

        results = {}
        for sensitive_feature in sensitive_features:
            quasi_identifiers = [
                feature
                for feature in base_quasi_identifiers
                if feature != sensitive_feature
            ]
            if not quasi_identifiers:
                raise ValueError(
                    "AIA requires at least one quasi-identifier for sensitive "
                    f"feature '{sensitive_feature}'."
                )

            x_attack_train = X_syn[quasi_identifiers].copy()
            y_attack_train = X_syn[sensitive_feature].copy()
            x_attack_test = X_train[quasi_identifiers].copy()
            y_attack_test = X_train[sensitive_feature].copy()

            task = self._task_for_sensitive_feature(sensitive_feature, y_attack_train)
            discrete_quasi_identifiers = [
                feature
                for feature in quasi_identifiers
                if self._is_discrete_feature(feature)
            ]

            score_fns = (
                [r2_score, root_mean_squared_error]
                if task == "regression"
                else [roc_auc_score]
            )
            scores = ml_task(
                x_attack_train,
                x_attack_test,
                y_attack_train,
                y_attack_test,
                discrete_quasi_identifiers,
                task,
                self.model_name,
                self.model_params,
                random_state=self.random_state,
                score_fns=score_fns,
            )

            prefix = f"{self.name}.{_metric_key_part(sensitive_feature)}"
            results.update(
                {f"{prefix}.{metric}": value for metric, value in scores.items()}
            )

        return results

    def _validate_columns(self, X_train: pd.DataFrame, X_syn: pd.DataFrame):
        missing_from_syn = [col for col in X_train.columns if col not in X_syn.columns]
        if missing_from_syn:
            raise ValueError(
                "Synthetic data is missing columns required for AIA: "
                f"{missing_from_syn}."
            )

        for attr_name in ("quasi_identifiers", "sensitive_features"):
            features = getattr(self, attr_name)
            if features is None:
                continue
            missing = [
                feature for feature in features if feature not in X_train.columns
            ]
            if missing:
                raise ValueError(
                    f"AIA {attr_name} contains columns not present in real data: "
                    f"{missing}."
                )

    def _sensitive_features(self, X_train: pd.DataFrame) -> list:
        if self.sensitive_features is None:
            return X_train.columns.tolist()
        return self.sensitive_features.copy()

    def _base_quasi_identifiers(
        self, X_train: pd.DataFrame, sensitive_features: list
    ) -> list:
        if self.quasi_identifiers is not None:
            return self.quasi_identifiers.copy()

        if self.sensitive_features is None:
            return X_train.columns.tolist()

        return [
            feature for feature in X_train.columns if feature not in sensitive_features
        ]

    def _task_for_sensitive_feature(
        self, sensitive_feature: str, y_train: pd.Series
    ) -> str:
        if self._is_discrete_feature(sensitive_feature):
            task = "binary"
            if y_train.nunique() > 2:
                task = "multiclass"
            return task
        return "regression"

    def _is_discrete_feature(self, feature: str) -> bool:
        return feature in self.discrete_features


@dataclass
class MIAData:
    """Datasets used by a membership inference attack."""

    reference: pd.DataFrame
    synthetic: pd.DataFrame
    non_members: pd.DataFrame
    members: pd.DataFrame
    x_eval: pd.DataFrame
    y_eval: pd.Series


class MIA(ABC):
    """Base class for membership inference attacks.

    The base class owns the common attack protocol: create reference,
    non-member, member, and synthetic sets; compute attack-specific membership
    scores on the aligned member/non-member set; then evaluate those scores.

    Args:
        ref_prop (float): Proportion of X_test used as the attacker reference set.
            The remaining X_test rows are evaluation non-members. Default: 0.5.
        member_prop (float): Proportion of X_train available as evaluation
            members. Default: 1.0.
        repeats (int): Number of repeated evaluations when subsample=True.
            Default: 1.
        subsample (bool): Whether to subsample synthetic and member sets in each
            repeat. If False, all selected members and synthetic records are used
            once. Default: False.
        random_state (int): Random seed for reproducibility. Default: 0.
    """

    score_metrics = (
        "auc",
        "lift_010",
        "lift_005",
        "lift_001",
        "tpr_at_fpr_010",
        "tpr_at_fpr_005",
        "tpr_at_fpr_001",
    )

    def __init__(
        self,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        repeats: int = 1,
        subsample: bool = False,
        random_state: int = 0,
    ):
        if repeats < 1:
            raise ValueError("repeats must be at least 1.")
        self.ref_prop = ref_prop
        self.member_prop = member_prop
        self.repeats = repeats
        self.subsample = subsample
        self.random_state = random_state

    def evaluate(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, X_syn: pd.DataFrame
    ):
        """Evaluate membership inference risk.

        Args:
            X_train: Real training data whose rows are treated as members.
            X_test: Independent real test data split into reference records and
                evaluation non-members.
            X_syn: Synthetic data available to the attacker.

        Returns:
            dict: Dictionary with attack AUC and lift-at-k scores. Keys have the
                form "<attack_name>.<score>".
        """
        scores = {}
        n_repeats = self.repeats if self.subsample else 1
        for repeat_idx in range(n_repeats):
            seed = self.random_state + repeat_idx
            mia_data = self._create_mia_data(X_train, X_test, X_syn, seed)
            membership_scores = np.asarray(self._membership_scores(mia_data, seed))

            if len(membership_scores) != len(mia_data.y_eval):
                raise ValueError(
                    "Membership attack returned a different number of scores than "
                    "evaluation records."
                )

            scores[repeat_idx] = {
                "auc": roc_auc_score(mia_data.y_eval, membership_scores),
                "lift_010": lift_at_k(mia_data.y_eval, membership_scores, 0.10),
                "lift_005": lift_at_k(mia_data.y_eval, membership_scores, 0.05),
                "lift_001": lift_at_k(mia_data.y_eval, membership_scores, 0.01),
                "tpr_at_fpr_010": tpr_at_fpr(
                    mia_data.y_eval, membership_scores, 0.10
                ),
                "tpr_at_fpr_005": tpr_at_fpr(
                    mia_data.y_eval, membership_scores, 0.05
                ),
                "tpr_at_fpr_001": tpr_at_fpr(
                    mia_data.y_eval, membership_scores, 0.01
                ),
            }

        avg_scores = {
            metric: float(
                np.mean([run_scores[metric] for run_scores in scores.values()])
            )
            for metric in self.score_metrics
        }

        return {f"{self.name}.{metric}": value for metric, value in avg_scores.items()}

    def _create_mia_data(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        syn: pd.DataFrame,
        seed: int,
    ) -> MIAData:
        n_ref = int(self.ref_prop * len(test))
        n_nonmembers = len(test) - n_ref
        n_member_pool = min(len(train), int(len(train) * self.member_prop))

        if n_ref <= 0:
            raise ValueError("ref_prop is too small: no reference samples selected.")
        if n_nonmembers <= 0:
            raise ValueError("ref_prop is too large: no evaluation non-members left.")
        if len(syn) <= 0:
            raise ValueError("Synthetic data must contain at least one sample.")
        if len(train) <= 0:
            raise ValueError("Training data must contain at least one sample.")
        if n_member_pool <= 0:
            raise ValueError("member_prop is too small: no member samples selected.")
        if self.subsample and len(syn) < n_ref:
            raise ValueError(
                f"Not enough synthetic samples: need at least {n_ref}, got {len(syn)}."
            )
        if self.subsample and n_member_pool < n_nonmembers:
            raise ValueError(
                "Not enough member samples: need at least "
                f"{n_nonmembers}, got {n_member_pool}."
            )

        shuffled_test = test.sample(frac=1, random_state=seed).reset_index(drop=True)
        reference = shuffled_test.iloc[:n_ref].reset_index(drop=True)
        non_members = shuffled_test.iloc[n_ref:].reset_index(drop=True)

        member_pool = (
            train.sample(frac=1, random_state=seed)
            .iloc[:n_member_pool]
            .reset_index(drop=True)
        )
        if self.subsample:
            members = member_pool.sample(n=n_nonmembers, random_state=seed).reset_index(
                drop=True
            )
            synthetic = syn.sample(n=n_ref, random_state=seed).reset_index(drop=True)
        else:
            members = member_pool
            synthetic = syn.reset_index(drop=True)

        x_eval = pd.concat((non_members, members), ignore_index=True)
        y_eval = pd.Series(
            np.r_[
                np.zeros(len(non_members), dtype=int),
                np.ones(len(members), dtype=int),
            ]
        )

        return MIAData(
            reference=reference,
            synthetic=synthetic,
            non_members=non_members,
            members=members,
            x_eval=x_eval,
            y_eval=y_eval,
        )

    @abstractmethod
    def _membership_scores(self, mia_data: MIAData, seed: int) -> np.ndarray:
        """Return membership scores for the prepared evaluation records."""


class DOMIAS(MIA):
    """DOMIAS membership inference attack metric.

    Paper: "Membership inference attacks against synthetic data through overfitting detection" by van Breugel et al. (2023).

    DOMIAS compares multivariate density of the attack records under the synthetic and reference distributions.
    Uses Gaussian KDE on PCA-transformed data for density estimation.

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        ref_prop (float): Proportion of test set to use as reference for density estimation. Default: 0.5.
        member_prop (float): Proportion of train set to use as members. Default: 1.0.
        n_components (int or float): Number of PCA components. Float in (0,1] = variance target,
            int = exact components. Default: 0.99.
        subsample (bool): Whether to subsample synthetic and member sets to match
            reference and evaluation non-member sizes. Default: False.
        repeats (int): Number of repeated evaluations when subsampling records.
            Default: 1.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.evaluation import DOMIAS
        >>>
        >>> # Prepare data
        >>> X_train = pd.DataFrame(...)
        >>> X_test = pd.DataFrame(...)
        >>> X_syn = pd.DataFrame(...)
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create metric
        >>> metric = DOMIAS(
        ...     discrete_features=discrete_features,
        ...     ref_prop=0.5,
        ...     n_components=0.95,
        ...     random_state=42
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_test, X_syn)
    """

    name = "mia.domias"

    def __init__(
        self,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        n_components: int = 0.99,
        random_state: int = 0,
        discrete_features: list = None,
        subsample: bool = False,
        repeats: int = 1,
    ):
        super().__init__(
            ref_prop=ref_prop,
            member_prop=member_prop,
            repeats=repeats,
            subsample=subsample,
            random_state=random_state,
        )
        self.n_components = n_components
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )

    def _membership_scores(self, mia_data: MIAData, seed: int) -> np.ndarray:
        syn = mia_data.synthetic.copy()
        reference_set = mia_data.reference.copy()
        numerical_features = [x for x in syn.columns if x not in self.discrete_features]

        # preprocess before PCA
        transformer = ColumnTransformer(
            transformers=[
                (
                    "ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self.discrete_features,
                ),
                ("scaler", StandardScaler(), numerical_features),
            ],
            remainder="passthrough",
        )
        transformer.fit(pd.concat((syn, reference_set), ignore_index=True))
        synth_set = transformer.transform(syn)
        reference_set = transformer.transform(reference_set)

        # dimensionality reduction
        embedder = PCA(n_components=self.n_components, random_state=seed)

        # Fit on syn and reference to avoid evaluation-data leakage.
        embedder.fit(np.concatenate((synth_set, reference_set)))
        synth_set = embedder.transform(synth_set)
        reference_set = embedder.transform(reference_set)

        # standardize for gaussian KDE
        scaler = StandardScaler()
        scaler.fit(np.concatenate((synth_set, reference_set)))
        synth_set = scaler.transform(synth_set)
        reference_set = scaler.transform(reference_set)

        synth_kde = gaussian_kde(synth_set.T)
        reference_kde = gaussian_kde(reference_set.T)

        x_eval = transformer.transform(mia_data.x_eval)
        x_eval = embedder.transform(x_eval)
        x_eval = scaler.transform(x_eval)
        p_g = synth_kde(x_eval.T)
        p_r = reference_kde(x_eval.T)
        p_rel = p_g / (p_r + 1e-10)
        return np.nan_to_num(p_rel)


class DPI(MIA):
    """Data Plagiarism Index membership inference attack.

    Paper: "Data plagiarism index: Characterizing the privacy risk of data-copying in tabular generative models" by Ward et al. (2024)

    DPI scores an attack record through the ratio of synthetic/reference samples in its local neighborhood.


    Args:
        k (int): Number of nearest neighbors from the combined reference and
            synthetic pool. Default: 20.
        discrete_features (list): List of discrete/categorical feature names.
            Default: [].
        ref_prop (float): Proportion of test set to use as attacker reference
            non-members. Default: 0.5.
        member_prop (float): Proportion of train set to use as members.
            Default: 1.0.
        repeats (int): Number of repeated evaluations when subsampling records.
            Default: 1.
        subsample (bool): Whether to subsample synthetic and member sets to
            match reference and evaluation non-member sizes. Default: False.
        random_state (int): Random seed for reproducibility. Default: 0.
    """

    name = "mia.dpi"

    def __init__(
        self,
        k: int = 20,
        discrete_features: list = None,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        repeats: int = 1,
        subsample: bool = False,
        random_state: int = 0,
    ):
        super().__init__(
            ref_prop=ref_prop,
            member_prop=member_prop,
            repeats=repeats,
            subsample=subsample,
            random_state=random_state,
        )
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )

    def _membership_scores(self, mia_data: MIAData, seed: int) -> np.ndarray:
        reference = mia_data.reference.copy()
        synthetic = mia_data.synthetic.copy()

        if len(reference) + len(synthetic) < self.k:
            raise ValueError(
                f"Need at least k={self.k} combined reference and synthetic "
                "records to compute DPI."
            )

        data = gower_like_transform(
            {
                "reference": reference,
                "synthetic": synthetic,
                "x_eval": mia_data.x_eval,
            },
            reference_data=reference,
            discrete_features=self.discrete_features,
            categorical_fit_data=[reference, synthetic],
            categorical_handle_unknown="ignore",
        )

        reference_array = data["reference"]
        synthetic_array = data["synthetic"]
        neighbor_pool = np.concatenate((reference_array, synthetic_array), axis=0)

        x_eval_array = data["x_eval"]
        _, indices = KDTree(neighbor_pool, metric="manhattan").query(
            x_eval_array,
            k=self.k,
        )

        reference_counts = np.sum(indices < len(reference_array), axis=1).astype(float)
        synthetic_counts = self.k - reference_counts

        return np.divide(
            synthetic_counts,
            reference_counts,
            out=np.zeros(len(synthetic_counts), dtype=float),
            where=reference_counts > 0,
        )


class ClassifierMIA(MIA):
    """Classifier-based membership inference attack metric.

    Trains a classifier to distinguish between the synthetic and reference distributions.

    Args:
        ref_prop (float): Proportion of test set to use as attacker reference
            non-members. Default: 0.5.
        model_name (str): Model used for the attack classifier. Default: "randomforest".
        model_params (dict): Optional parameters for the attack classifier.
        discrete_features (list): List of discrete/categorical feature names.
            Default: [].
        member_prop (float): Proportion of train set to use as members.
            Default: 1.0.
        repeats (int): Number of repeated evaluations when subsampling records.
            Default: 1.
        subsample (bool): Whether to subsample the synthetic set to the same size
            as the attacker reference set, and the member set to the same size as
            the evaluation non-member set. If False, all synthetic records and all
            members are used, and the metric is evaluated once. Default: False.
        random_state (int): Random seed for reproducibility. Default: 0.
    """

    name = "mia.classifier"

    def __init__(
        self,
        ref_prop: float = 0.5,
        model_name: str = "randomforest",
        model_params: dict = None,
        discrete_features: list = None,
        member_prop: float = 1.0,
        repeats: int = 1,
        subsample: bool = False,
        random_state: int = 0,
    ):
        super().__init__(
            ref_prop=ref_prop,
            member_prop=member_prop,
            repeats=repeats,
            subsample=subsample,
            random_state=random_state,
        )
        self.categorical_features = (
            [] if discrete_features is None else discrete_features.copy()
        )
        self.model_params = {} if model_params is None else model_params.copy()
        self.model_name = model_name

    def _membership_scores(self, mia_data: MIAData, seed: int) -> np.ndarray:
        x_train = pd.concat((mia_data.reference, mia_data.synthetic), ignore_index=True)
        y_train = pd.Series(
            np.r_[
                np.zeros(len(mia_data.reference), dtype=int),
                np.ones(len(mia_data.synthetic), dtype=int),
            ]
        )
        return ml_task(
            x_train,
            mia_data.x_eval,
            y_train,
            mia_data.y_eval,
            self.categorical_features,
            "binary",
            self.model_name,
            self.model_params,
            random_state=seed,
        )


class EnsembleMIA(MIA):
    """Ensemble membership inference attack metric.

    Ensembles the membership scores from multiple MIA implementations.

    Args:
        include_mia (dict): Mapping of MIA names to constructor parameters.
            Supported keys are "classifier_mia", "dpi", "domias", and their
            metric-name aliases "mia.classifier", "mia.dpi", and "mia.domias".
            Default: {"classifier_mia": {}, "dpi": {}, "domias": {}}.
        ensemble (str): How to combine component MIA scores. "mean" min-max
            normalizes each component score vector and averages them.
            "rank_avg" averages normalized within-component ranks. Default:
            "rank_avg".
        discrete_features (list): List of discrete/categorical feature names.
            Passed to component attacks unless overridden in include_mia.
        ref_prop (float): Proportion of test set to use as attacker reference
            non-members. Default: 0.5.
        member_prop (float): Proportion of train set to use as members.
            Default: 1.0.
        repeats (int): Number of repeated evaluations when subsampling records.
            Default: 1.
        subsample (bool): Whether to subsample synthetic/member sets in the
            shared MIA protocol. Default: False.
        random_state (int): Random seed for reproducibility. Default: 0.
    """

    name = "mia.ensemble"

    _available_mias = {
        "classifier_mia": ClassifierMIA,
        "dpi": DPI,
        "domias": DOMIAS,
        "mia.classifier": ClassifierMIA,
        "mia.dpi": DPI,
        "mia.domias": DOMIAS,
    }

    def __init__(
        self,
        include_mia: dict = None,
        ensemble: str = "rank_avg",
        discrete_features: list = None,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        repeats: int = 1,
        subsample: bool = False,
        random_state: int = 0,
    ):
        super().__init__(
            ref_prop=ref_prop,
            member_prop=member_prop,
            repeats=repeats,
            subsample=subsample,
            random_state=random_state,
        )
        self.discrete_features = (
            discrete_features if discrete_features is not None else []
        )
        self.ensemble = ensemble.strip().lower()
        if self.ensemble not in {"mean", "rank_avg"}:
            raise ValueError("ensemble must be either 'mean' or 'rank_avg'.")
        self.include_mia = (
            {
                "classifier_mia": {},
                "dpi": {},
                "domias": {},
            }
            if include_mia is None
            else include_mia
        )
        if not self.include_mia:
            raise ValueError("include_mia must contain at least one MIA.")
        self.mias = self._build_mias()

    def _build_mias(self) -> dict:
        mias = {}
        for mia_name, mia_params in self.include_mia.items():
            normalized_name = mia_name.strip().lower()
            if normalized_name not in self._available_mias:
                supported = ", ".join(sorted(self._available_mias))
                raise ValueError(
                    f"Unsupported MIA '{mia_name}'. Supported MIAs are: {supported}."
                )
            if mia_params is None:
                mia_params = {}
            if not isinstance(mia_params, dict):
                raise ValueError(
                    f"Parameters for MIA '{mia_name}' must be provided as a dict."
                )

            params = mia_params.copy()
            params.setdefault("discrete_features", self.discrete_features)
            params.setdefault("random_state", self.random_state)
            mia_cls = self._available_mias[normalized_name]
            mias[normalized_name] = mia_cls(**params)
        return mias

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        if np.isclose(max_score, min_score):
            return np.full(len(scores), 0.5, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def _rank_average_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(scores)
        if len(scores) <= 1:
            return np.full(len(scores), 0.5, dtype=float)
        ranks = rankdata(scores, method="average")
        return (ranks - 1) / (len(scores) - 1)

    def _score_membership(self, y_eval: pd.Series, scores: np.ndarray) -> dict:
        scores = np.nan_to_num(np.asarray(scores, dtype=float))
        return {
            "auc": roc_auc_score(y_eval, scores),
            "lift_010": lift_at_k(y_eval, scores, 0.10),
            "lift_005": lift_at_k(y_eval, scores, 0.05),
            "lift_001": lift_at_k(y_eval, scores, 0.01),
            "tpr_at_fpr_010": tpr_at_fpr(y_eval, scores, 0.10),
            "tpr_at_fpr_005": tpr_at_fpr(y_eval, scores, 0.05),
            "tpr_at_fpr_001": tpr_at_fpr(y_eval, scores, 0.01),
        }

    def _component_membership_scores(
        self, mia_data: MIAData, seed: int
    ) -> dict[str, np.ndarray]:
        component_scores = {}
        for mia in self.mias.values():
            scores = np.asarray(mia._membership_scores(mia_data, seed))
            if len(scores) != len(mia_data.y_eval):
                raise ValueError(
                    f"MIA '{mia.name}' returned a different number of scores than "
                    "evaluation records."
                )
            component_scores[mia.name] = scores
        return component_scores

    def _ensemble_scores(self, component_scores: dict[str, np.ndarray]) -> np.ndarray:
        if self.ensemble == "mean":
            scores = [
                self._normalize_scores(scores) for scores in component_scores.values()
            ]
        else:
            scores = [
                self._rank_average_scores(scores)
                for scores in component_scores.values()
            ]
        return np.average(scores, axis=0)

    def evaluate(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, X_syn: pd.DataFrame
    ):
        scores = {}
        n_repeats = self.repeats if self.subsample else 1
        for repeat_idx in range(n_repeats):
            seed = self.random_state + repeat_idx
            mia_data = self._create_mia_data(X_train, X_test, X_syn, seed)
            component_scores = self._component_membership_scores(mia_data, seed)
            ensemble_scores = self._ensemble_scores(component_scores)

            scores.setdefault(self.name, {})[repeat_idx] = self._score_membership(
                mia_data.y_eval, ensemble_scores
            )
            for mia_name, mia_scores in component_scores.items():
                scores.setdefault(mia_name, {})[repeat_idx] = self._score_membership(
                    mia_data.y_eval, mia_scores
                )

        results = {}
        for attack_name, repeat_scores in scores.items():
            prefix = (
                self.name
                if attack_name == self.name
                else f"{self.name}.{_metric_key_part(attack_name)}"
            )
            avg_scores = {
                metric: float(
                    np.mean(
                        [run_scores[metric] for run_scores in repeat_scores.values()]
                    )
                )
                for metric in self.score_metrics
            }
            results.update(
                {f"{prefix}.{metric}": value for metric, value in avg_scores.items()}
            )
        return results

    def _membership_scores(self, mia_data: MIAData, seed: int) -> np.ndarray:
        component_scores = self._component_membership_scores(mia_data, seed)
        return self._ensemble_scores(component_scores)
