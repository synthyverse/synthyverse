import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


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
                    f"dcr.train.{estimate}": float(score_train),
                    f"dcr.test.{estimate}": float(score_test),
                }
            )

        return dictionary
