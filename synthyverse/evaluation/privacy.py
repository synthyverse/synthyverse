import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances_argmin_min

from .utils import get_accuracy_metric


class DCR:
    name = "dcr"
    data_requirement = "train_and_test"
    needs_discrete_features = True

    def __init__(
        self,
        discrete_features: list = [],
        subsample_test_size: bool = True,
        max_rows: int = 50000,
    ):
        super().__init__()
        self.discrete_features = discrete_features
        self.subsample_test_size = subsample_test_size
        self.max_rows = max_rows

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame, sd: pd.DataFrame):
        # compare training set to same size synthetic set
        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]

        assert len(test) <= len(
            train
        ), "Test set must be smaller than or equal to train size to compute DCR"

        # scale the data to fixed range
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
        scaler = MinMaxScaler()
        scaler.fit(train[numerical_features])
        data = {}
        for df, name in zip([train, test, sd[: len(train)]], ["train", "test", "syn"]):
            cat = ohe.transform(df[self.discrete_features])
            cat = cat / 2  # scaling for Gower distance
            num = scaler.transform(df[numerical_features])
            data[name] = np.concatenate((cat, num), axis=1)

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

        def choose(x: np.ndarray, n: int) -> np.ndarray:
            return x[np.random.choice(len(x), n, replace=False)]

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
    name = "domias"
    data_requirement = "train_and_test"
    needs_discrete_features = True
    needs_random_state = True

    def __init__(
        self,
        discrete_features: list = [],
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
        self.discrete_features = discrete_features
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
        numerical_features = [
            col for col in train.columns if col not in self.discrete_features
        ]

        # ----- One-hot (fit on union to fix category spaces) -----
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

        # ----- Metrics / thresholding (unchanged) -----
        if self.metric != "roc_auc":
            threshold = np.percentile(P_rel, 100 * (1 - self.predict_top))
            P_rel = P_rel > threshold

        metric_func, self.metric = get_accuracy_metric(self.metric)
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
            # naive baseline: predict all as members
            P_rel_naive = np.ones_like(P_rel)
            naive_score = metric_func(Y_test, P_rel_naive)
            if self.metric == "precision-recall":
                naive_precision, naive_recall = naive_score
                domias[docstring + ".metric=precision.naive"] = naive_precision
                domias[docstring + ".metric=recall.naive"] = naive_recall
            else:
                domias[docstring + f".metric={self.metric}.naive"] = naive_score

        return domias
