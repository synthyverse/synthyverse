import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from .utils import get_accuracy_metric


class DCR:
    """
    Distance to Closest Record scores.
    Indicates closeness of synthetic data to the training data, and an independent holdout set.
    """

    data_requirement = "train_and_test_preprocessed"

    def __init__(
        self,
        estimates: list = [
            "mean",
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
        ],
        batch_size: int = 16000,
    ):
        super().__init__()
        self.estimates = estimates
        self.batch_size = batch_size

    def _compute_min_distances_batch(
        self, query_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute minimum distances between query_data and reference_data in batches.

        Args:
            query_data: DataFrame containing query points
            reference_data: DataFrame containing reference points

        Returns:
            Array of minimum distances for each query point
        """
        if self.batch_size is None:
            # Use original method for small datasets or when batch_size is not specified
            _, min_distances = pairwise_distances_argmin_min(
                query_data, reference_data, metric="euclidean"
            )
            return min_distances

        min_distances = []
        n_query = len(query_data)

        for i in range(0, n_query, self.batch_size):
            end_idx = min(i + self.batch_size, n_query)
            batch_query = query_data.iloc[i:end_idx]

            _, batch_min_distances = pairwise_distances_argmin_min(
                batch_query, reference_data, metric="euclidean"
            )
            min_distances.extend(batch_min_distances)

        return np.array(min_distances)

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame, sd: pd.DataFrame):

        sd = sd[: len(train)]

        # Use batch processing if batch_size is specified
        min_distances_syn = self._compute_min_distances_batch(sd, train)
        min_distances_test = self._compute_min_distances_batch(test, train)

        dictionary = {}

        for estimate in self.estimates:
            if estimate == "mean":
                score_train = min_distances_syn.mean()
                score_test = min_distances_test.mean()
            else:
                score_train = np.quantile(min_distances_syn, estimate)
                score_test = np.quantile(min_distances_test, estimate)
            dictionary.update(
                {
                    f"dcr.train.batch_size={self.batch_size}.{estimate}": float(
                        score_train
                    ),
                    f"dcr.test.batch_size={self.batch_size}.{estimate}": float(
                        score_test
                    ),
                }
            )

        return dictionary


class DOMIAS:
    data_requirement = "train_and_test_preprocessed"

    def __init__(
        self,
        ref_prop: float = 0.5,
        member_prop: float = 1.0,
        n_components: int = 0.95,
        random_state: int = 0,
        metric: str = "roc_auc",
        predict_top: float = 0.5,
    ):
        super().__init__()
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
        """
        Computes DOMIAS membership inference attack accuracy (AUROC).
        Estimates density through Gaussian KDE after dimensionality reduction.
        Code based on Synthcity library.

        ref_prop: proportion of test set used as reference set for computing RD density.
        n_components: number of dimensions to retain after reduction.
        """

        members = train[: int(len(train) * self.member_prop)]
        ref_size = int(self.ref_prop * len(test))
        reference_set, non_members = test[:ref_size], test[ref_size:]

        X_test = np.concatenate((members.to_numpy(), non_members.to_numpy()))
        Y_test = np.concatenate(
            [np.ones(members.shape[0]), np.zeros(non_members.shape[0])]
        ).astype(bool)

        embedder = PCA(n_components=self.n_components, random_state=self.random_state)

        # fit embedder on syn and reference to avoid leakage of test data which would inflate separability
        all_ = np.concatenate((syn.to_numpy(), reference_set.to_numpy()))
        all_ = embedder.fit_transform(all_)
        # project test data to same space
        all_ = np.concatenate((all_, embedder.transform(X_test)))  # type: ignore
        # standardize for gaussian KDE
        all_ = StandardScaler().fit_transform(all_)
        synth_set = all_[: len(syn)]
        reference_set = all_[len(syn) : len(syn) + len(reference_set)]
        X_test = all_[-len(X_test) :]

        kde = gaussian_kde(synth_set.T)
        P_G = kde(X_test.T)
        kde = gaussian_kde(reference_set.T)
        P_R = kde(X_test.T)
        P_rel = P_G / (P_R + 1e-10)
        P_rel = np.nan_to_num(P_rel)

        if self.metric != "roc_auc":
            threshold = np.percentile(P_rel, 100 * (1 - self.predict_top))

            P_rel = P_rel > threshold

        metric_func, self.metric = get_accuracy_metric(self.metric)
        score = metric_func(Y_test, P_rel)

        domias = {}
        docstring = f"domias.predict_top={self.predict_top}.ref_prop={self.ref_prop}.member_prop={self.member_prop}.n_components={self.n_components}"
        if self.metric == "precision-recall":
            precision, recall = score
            domias[docstring + f".metric=precision"] = precision
            domias[docstring + f".metric=recall"] = recall
        else:
            domias[docstring + f".metric={self.metric}"] = score

        if self.metric != "roc_auc":
            # add a naive score (i.e. predicting all as members)
            P_rel = np.ones_like(P_rel)
            naive_score = metric_func(Y_test, P_rel)
            if self.metric == "precision-recall":
                naive_precision, naive_recall = naive_score
                domias[docstring + f".metric=precision.naive"] = naive_precision
                domias[docstring + f".metric=recall.naive"] = naive_recall
            else:
                domias[docstring + f".metric={self.metric}.naive"] = naive_score

        return domias
