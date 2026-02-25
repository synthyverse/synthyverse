import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import (
    pairwise_distances_argmin_min,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


class DCR:
    """Distance to Closest Record (DCR) privacy metric.

    Measures whether synthetic records are more often closer (by Gower distance) to a training record than an independent test record.
    Low scores indicate risk that synthetic data overfits to the training set, and therefore privacy risk.

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        subsample_test_size (bool): Whether to subsample the training set and synthetic set to the test set size. Prevents biasing DCR due to different sample sizes. If used, multiple iterations of the DCR score are computed and aggregated, to ensure the metric is based on all training and synthetic records. Default: True.
        max_rows (int): Maximum number of rows to use for computation to limit evaluation time. Default: 50000.

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
        ...     max_rows=50000,
        ...     subsample_test_size=True
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_test, X_syn)
    """

    name = "dcr"
    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_random_state = True

    def __init__(
        self,
        discrete_features: list = None,
        subsample_test_size: bool = True,
        max_rows: int = 50000,
        random_state: int = 0,
    ):
        super().__init__()
        self.discrete_features = discrete_features if discrete_features is not None else []
        self.subsample_test_size = subsample_test_size
        self.max_rows = max_rows
        self.random_state = random_state

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame, sd: pd.DataFrame):
        """Evaluate synthetic data privacy using DCR metric.

        DCR score is aggregated as min(1, dcr.closer_to_test * 2) to ensure higher scores indicate better privacy.

        Args:
            train: Training data as a pandas DataFrame.
            test: Test data as a pandas DataFrame.
            sd: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with keys:
                - "dcr.score": DCR score
                - "dcr.closer_to_train": Proportion closer to train
                - "dcr.closer_to_test": Proportion closer to test

        Raises:
            AssertionError: If test set is larger than train set.
        """
        # compare training set to same size synthetic set
        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]

        assert len(test) <= len(
            train
        ), "Test set must be smaller than or equal to train size to compute DCR"

        # scale the data to fixed range
        ohe = None
        if self.discrete_features:
            ohe = OneHotEncoder(sparse_output=False)
            ohe.fit(
                pd.concat(
                    [
                        train[self.discrete_features],
                        test[self.discrete_features],
                        sd[self.discrete_features],
                    ],
                    axis=0,
                )
            )

        scaler = None
        if numerical_features:
            scaler = MinMaxScaler()
            scaler.fit(train[numerical_features])

        data = {}
        for df, name in zip([train, test, sd[: len(train)]], ["train", "test", "syn"]):
            parts = []
            if ohe is not None:
                cat = ohe.transform(df[self.discrete_features])
                cat = cat / 2  # scaling for Gower distance
                parts.append(cat)
            if scaler is not None:
                num = scaler.transform(df[numerical_features])
                parts.append(num)
            data[name] = np.concatenate(parts, axis=1)

        # subsample the dataset to either max rows or test size
        num_rows_subsample = (
            min(len(data["test"]), self.max_rows)
            if self.subsample_test_size
            else self.max_rows
        )
        num_rows_subsample = min(num_rows_subsample, len(data["train"]))
        num_iterations = int(np.ceil(len(data["train"]) / num_rows_subsample))
        scores = []
        closer_to_train = []
        closer_to_test = []

        rng = np.random.default_rng(self.random_state)

        def choose(x: np.ndarray, n: int) -> np.ndarray:
            return x[rng.choice(len(x), n, replace=False)]

        for _ in range(num_iterations):
            syn_curr = choose(data["syn"], num_rows_subsample)

            _, d_s_tr = pairwise_distances_argmin_min(
                syn_curr, choose(data["train"], num_rows_subsample), metric="cityblock"
            )
            # align test set size to never exceed subsampled set size
            _, d_s_te = pairwise_distances_argmin_min(
                syn_curr,
                choose(data["test"], min(len(data["test"]), num_rows_subsample)),
                metric="cityblock",
            )

            closer_to_train_ = np.mean(d_s_tr < d_s_te)
            closer_to_test_ = 1 - closer_to_train_
            closer_to_train.append(closer_to_train_)
            closer_to_test.append(closer_to_test_)
            score = min(1, closer_to_test_ * 2)
            scores.append(score)
        return {
            "dcr.score": np.mean(scores),
            "dcr.closer_to_train": np.mean(closer_to_train),
            "dcr.closer_to_test": np.mean(closer_to_test),
        }


class DOMIAS:
    """DOMIAS membership inference attack metric.

    Measures vulnerability to membership inference attacks using density-based
    methods. Lower scores indicate better privacy. Uses KDE on PCA-transformed data for density estimation.

    For threshold-based metrics we also include a naive baseline, i.e., all records are predicted as members.

    Based on the paper "Membership inference attacks against synthetic data through overfitting detection" by van Breugel et al. (2023).

    Args:
        discrete_features (list): List of discrete/categorical feature names. Default: [].
        ref_prop (float): Proportion of test set to use as reference for density estimation. Default: 0.5.
        member_prop (float): Proportion of train set to use as members. Default: 1.0.
        n_components (Union[int, float]): Number of PCA components. Float in (0,1] = variance target,
            int = exact components. Default: 0.95.
        random_state (int): Random seed for reproducibility. Default: 0.
        metric (str): Evaluation metric. One of "roc_auc", "f1", "accuracy", "precision-recall". Default: "roc_auc".
        predict_top (float): Proportion of top predictions to consider as members. Can be lowered to increase attack precision at the cost of recall. Only relevant for threshold-based metrics, e.g., "precision-recall". Default: 0.5.

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
        ...     metric="roc_auc",
        ...     random_state=42
        ... )
        >>>
        >>> # Evaluate
        >>> results = metric.evaluate(X_train, X_test, X_syn)
    """

    name = "domias"
    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_random_state = True

    def __init__(
        self,
        discrete_features: list = None,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        n_components: (
            int | float
        ) = 0.95,  # float in (0,1] = variance target; int = exact components
        random_state: int = 0,
        metric: str = "roc_auc",
        predict_top: float = 0.5,
    ):
        super().__init__()
        self.discrete_features = discrete_features if discrete_features is not None else []
        self.ref_prop = ref_prop
        self.member_prop = member_prop
        self.n_components = n_components
        self.random_state = random_state
        self.metric = metric
        self.predict_top = predict_top

    def evaluate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        syn: pd.DataFrame,
    ):
        """Evaluate synthetic data privacy using DOMIAS membership inference attack.

        Args:
            train: Training data as a pandas DataFrame.
            test: Test data as a pandas DataFrame.
            syn: Synthetic data as a pandas DataFrame.

        Returns:
            dict: Dictionary with DOMIAS metric scores. Keys include metric name with
                configuration parameters in the name.
        """
        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]

        # ----- One-hot (fit on union to fix category spaces) -----
        onehot_encoder = None
        if self.discrete_features:
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            onehot_encoder.fit(
                pd.concat(
                    [
                        train[self.discrete_features],
                        test[self.discrete_features],
                        syn[self.discrete_features],
                    ],
                    axis=0,
                    ignore_index=True,
                )
            )

        def encode(df: pd.DataFrame) -> np.ndarray:
            if self.discrete_features:
                oh = onehot_encoder.transform(df[self.discrete_features])
                if numerical_features:
                    X = np.concatenate((df[numerical_features].to_numpy(), oh), axis=1)
                else:
                    X = oh
            else:
                X = df[numerical_features].to_numpy()
            return X

        x_train = encode(train)
        x_test = encode(test)
        x_syn = encode(syn)

        # ----- Split members / reference / non-members -----
        n_train = len(x_train)
        n_test = len(x_test)

        members = x_train[: int(n_train * self.member_prop)]
        ref_size = int(self.ref_prop * n_test)
        reference_set_raw = x_test[:ref_size]
        non_members = x_test[ref_size:]

        X_test_raw = np.concatenate((members, non_members), axis=0)
        Y_test = np.concatenate(
            [np.ones(len(members)), np.zeros(len(non_members))]
        ).astype(bool)

        # ----- PCA fit on syn + reference ONLY (no leakage from X_test_raw) -----
        fit_mat = np.concatenate((x_syn, reference_set_raw), axis=0)
        pca_full = PCA(svd_solver="full", random_state=self.random_state).fit(fit_mat)

        # target components by variance threshold or integer
        if isinstance(self.n_components, float) and 0 < self.n_components <= 1.0:
            csum = np.cumsum(pca_full.explained_variance_ratio_)
            n_var_thresh = int(np.searchsorted(csum, self.n_components) + 1)
        else:
            n_var_thresh = int(self.n_components)

        # numerical intrinsic rank (drop near-zero eigenvalues)
        lam = pca_full.explained_variance_
        lam_max = float(lam.max()) if lam.size else 0.0
        eps = np.finfo(np.float64).eps
        rank_tol = eps * max(fit_mat.shape) * max(lam_max, 1.0)
        intrinsic_rank = int((lam > rank_tol).sum())

        # cap by sample counts of each KDE fit (need d <= n-1 for covariance)
        n_syn = len(x_syn)
        n_ref = len(reference_set_raw)
        cap_by_counts = max(1, min(n_syn, n_ref) - 1)

        n_comp = max(1, min(n_var_thresh, intrinsic_rank, cap_by_counts))

        # project all relevant sets using top PCs
        W = pca_full.components_[:n_comp]  # (n_comp, D)
        mu = pca_full.mean_
        synth_set = (x_syn - mu) @ W.T
        reference_set = (reference_set_raw - mu) @ W.T
        X_test = (X_test_raw - mu) @ W.T

        # ----- Standardize AFTER PCA (fit on syn+ref only) -----
        dr_scaler = StandardScaler(with_mean=True, with_std=True).fit(
            np.vstack([synth_set, reference_set])
        )
        synth_set = dr_scaler.transform(synth_set)
        reference_set = dr_scaler.transform(reference_set)
        X_test = dr_scaler.transform(X_test)

        # ----- KDEs with Scott's rule (isotropic) -----
        kde_syn = KernelDensity(kernel="gaussian").fit(synth_set)
        kde_ref = KernelDensity(kernel="gaussian").fit(reference_set)

        logP_G = kde_syn.score_samples(X_test)
        logP_R = kde_ref.score_samples(X_test)
        logP_rel = logP_G - logP_R
        # if you need linear ratio
        P_rel = np.exp(logP_rel)

        # ----- Resolve metric function -----
        _metric_funcs = {
            "roc_auc": lambda y_true, y_score: roc_auc_score(y_true, y_score),
            "f1": f1_score,
            "accuracy": accuracy_score,
            "precision-recall": lambda y_true, y_pred: (
                precision_score(y_true, y_pred),
                recall_score(y_true, y_pred),
            ),
        }
        if self.metric not in _metric_funcs:
            raise ValueError(
                f"Metric '{self.metric}' not supported. "
                f"Choose from: {list(_metric_funcs.keys())}"
            )
        metric_func = _metric_funcs[self.metric]

        if self.metric != "roc_auc":
            threshold = np.percentile(P_rel, 100 * (1 - self.predict_top))
            P_rel = P_rel > threshold

        score = metric_func(Y_test, P_rel)

        domias = {}
        docstring = (
            f"domias.predict_top={self.predict_top}"
            f".ref_prop={self.ref_prop}"
            f".member_prop={self.member_prop}"
            f".n_components={self.n_components}"
        )

        if self.metric == "precision-recall":
            precision, recall = score
            domias[docstring + ".metric=precision"] = precision
            domias[docstring + ".metric=recall"] = recall
        else:
            domias[docstring + f".metric={self.metric}"] = score

        if self.metric != "roc_auc":
            P_rel_naive = np.ones(len(Y_test))
            naive_score = metric_func(Y_test, P_rel_naive)
            if self.metric == "precision-recall":
                naive_precision, naive_recall = naive_score
                domias[docstring + ".metric=precision.naive"] = naive_precision
                domias[docstring + ".metric=recall.naive"] = naive_recall
            else:
                domias[docstring + f".metric={self.metric}.naive"] = naive_score

        return domias
