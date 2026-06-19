import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN

from ..base import BaseGenerator
from ..persistence import load_generator_state, restore_generator, save_generator_state


class SMOTEGenerator(BaseGenerator):
    """Synthetic Minority Over-sampling Technique (SMOTE) for tabular data.

    Creates synthetic samples via interpolation in feature space using
    SMOTE.

    For classification tasks, the provided target column is used directly for
    class-conditional oversampling. For regression tasks, a pseudo-binary target is
    derived by splitting the target at its median, following a strategy similar to the
    TabDDPM paper.


    Args:
        target_column (str): Name of the target column used to drive oversampling.
        k_neighbors (int): Number of nearest neighbors used during interpolation.
            Default: 5.
        n_jobs (int): Number of parallel jobs for nearest-neighbor search.
            Default: -1.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import SMOTEGenerator
        >>>
        >>> # Load data and define discrete features
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator
        >>> generator = SMOTEGenerator(
        ...     target_column="target",
        ...     k_neighbors=5,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate synthetic rows
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "smote"

    def __init__(
        self,
        target_column: str,
        k_neighbors: int = 5,
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        self.target_column = target_column
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame, discrete_features: list):

        self.is_classification = self.target_column in discrete_features

        self.X = X.copy()

        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        self.X[discrete_features] = self.ordinal_encoder.fit_transform(
            self.X[discrete_features]
        )

        if not self.is_classification:
            # pseudo outcome similar to TabDDPM paper
            self.y_train = np.where(
                self.X[self.target_column] > np.median(self.X[self.target_column]), 1, 0
            )
            self.y_train = pd.Series(self.y_train)
        else:
            self.y_train = self.X[self.target_column]
            self.X = self.X.drop(columns=[self.target_column])

        self.discrete_features = [x for x in self.X.columns if x in discrete_features]
        self.numerical_features = [
            x for x in self.X.columns if x not in discrete_features
        ]

        # SMOTE is not a model, so we don't need to fit it here

        return self

    def _generate(self, n: int):
        rng = check_random_state(self.random_state)

        if len(self.numerical_features) > 0:
            if len(self.discrete_features) > 0:
                self.smote = SMOTENC
            else:
                self.smote = SMOTE
        else:
            self.smote = SMOTEN

        # setup SMOTE
        frac_samples = n / self.X.shape[0]
        sampling_strategy = {
            k: int((1 + frac_samples) * np.sum(self.y_train == k))
            for k in np.unique(self.y_train)
        }
        obs_sum = sum(sampling_strategy.values())
        diff = obs_sum - self.y_train.shape[0]
        # if too many / too few samples would be drawn, make adjustments to randomly chosen class
        if diff != n:
            c = rng.choice(list(sampling_strategy.keys()), 1).item()
            sampling_strategy[c] += n - diff
            assert sum(sampling_strategy.values()) - self.y_train.shape[0] == n

        # compute per-class counts
        class_counts = pd.Series(self.y_train).value_counts()
        min_count = int(class_counts.min())

        # SMOTE needs at least 2 samples in any class it will oversample
        if min_count < 2:
            raise ValueError(
                f"SMOTE cannot run: smallest class has {min_count} sample(s). "
                "Need at least 2."
            )

        # cap k_neighbors so that k_neighbors <= min_count - 1
        k_eff = min(self.k_neighbors, min_count - 1)

        nearest_neighbors = NearestNeighbors(n_neighbors=k_eff, n_jobs=self.n_jobs)

        # nearest_neighbors = NearestNeighbors(
        #     n_neighbors=self.k_neighbors, n_jobs=self.n_jobs
        # )

        params = {
            "sampling_strategy": sampling_strategy,
            "random_state": self.random_state,
            "k_neighbors": nearest_neighbors,
        }
        if len(self.discrete_features) > 0 and len(self.numerical_features) > 0:
            params["categorical_features"] = [
                x for x in self.discrete_features if x != self.target_column
            ]

        self.smote = self.smote(**params)
        syn_X, syn_y = self.smote.fit_resample(self.X, self.y_train)

        # only retain fake data not true samples
        syn_X = syn_X[self.X.shape[0] :]
        syn_y = syn_y[self.y_train.shape[0] :]

        # shuffle generated data
        idx = rng.permutation(len(syn_X))
        syn_X = syn_X.iloc[idx]
        syn_y = syn_y.iloc[idx]

        syn_X, syn_y = syn_X.reset_index(drop=True), syn_y.reset_index(drop=True)

        if self.is_classification:
            # append y to data if target is discrete; for regression there was no real target
            syn_y = pd.Series(syn_y, name=self.target_column)
            syn_X = pd.concat([syn_X, syn_y], axis=1)

        syn_X[self.discrete_features] = syn_X[self.discrete_features].astype(int)

        syn_X[self.discrete_features] = self.ordinal_encoder.inverse_transform(
            syn_X[self.discrete_features]
        )

        return syn_X

    def save(self, path):
        state = {
            "target_column": self.target_column,
            "k_neighbors": self.k_neighbors,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "is_classification": self.is_classification,
            "X": self.X,
            "ordinal_encoder": self.ordinal_encoder,
            "y_train": self.y_train,
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
        }
        return save_generator_state(path, state)

    @classmethod
    def load(cls, path):
        return restore_generator(cls, load_generator_state(path))
