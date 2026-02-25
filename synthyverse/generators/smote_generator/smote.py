import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN


from ..base import TabularBaseGenerator


class SMOTEGenerator(TabularBaseGenerator):
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
        **kwargs: Additional arguments passed to `TabularBaseGenerator`.

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
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        k_neighbors: int = 5,
        n_jobs: int = -1,
        random_state: int = 0,
        **kwargs,
    ):
        self.target_column = target_column
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        super().__init__(random_state=random_state, **kwargs)

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):

        self.is_classification = self.target_column in discrete_features

        self.X = X.copy()

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

    def _generate_data(self, n: int):
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
            c = np.random.choice(list(sampling_strategy.keys()), 1).item()
            sampling_strategy[c] += n - diff
            assert sum(sampling_strategy.values()) - self.y_train.shape[0] == n

        # compute per-class counts
        class_counts = pd.Series(self.y_train).value_counts()
        min_count = int(class_counts.min())

        # SMOTE needs at least 2 samples in any class it will oversample
        if min_count < 2:
            raise ValueError(
                f"SMOTE cannot run: smallest class has {min_count} sample(s). "
                "Need at least 2. Consider RandomOverSampler or duplication for tiny classes."
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
        idx = np.random.permutation(len(syn_X))
        syn_X = syn_X.iloc[idx]
        syn_y = syn_y.iloc[idx]

        syn_X, syn_y = syn_X.reset_index(drop=True), syn_y.reset_index(drop=True)

        if self.is_classification:
            # append y to data if target is discrete; for regression there was no real target
            syn_y = pd.Series(syn_y, name=self.target_column)
            syn_X = pd.concat([syn_X, syn_y], axis=1)

        syn_X[self.discrete_features] = syn_X[self.discrete_features].astype(int)

        return syn_X
