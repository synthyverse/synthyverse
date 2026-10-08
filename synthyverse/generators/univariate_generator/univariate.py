import numpy as np
import pandas as pd

from ..base import BaseGenerator
from sklearn.preprocessing import KBinsDiscretizer


class UnivariateGenerator(BaseGenerator):
    """Univariate baseline generator for tabular synthetic data.

    Generates each feature independently. Categorical features are
    sampled from the observed empirical distribution. Numerical features are sampled
    uniformly from the range of the real data.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import UnivariateGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = UnivariateGenerator()
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "univariate"

    def __init__(
        self,
        n_bins: int = 1000,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.n_bins = n_bins

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.columns = X.columns.tolist()
        self.categorical_features = [
            col for col in self.columns if col in discrete_features
        ]
        self.numerical_features = [
            col for col in self.columns if col not in self.categorical_features
        ]
        self.category_values = {}
        self.category_probabilities = {}

        for col in self.categorical_features:
            frequencies = X[col].value_counts(normalize=True, dropna=False)
            self.category_values[col] = frequencies.index.to_numpy()
            self.category_probabilities[col] = frequencies.to_numpy()

        self.num_enc = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile")
        x_num_enc = self.num_enc.fit_transform(X[self.numerical_features])

        self.numeric_bins = {}
        self.numeric_probabilities = {}

        for col in self.numerical_features:
            col_idx = self.columns.index(col)
            bin_idx, cnt = np.unique(x_num_enc[:,col_idx], return_counts=True)
            p = cnt / cnt.sum()
            self.numeric_bins[col_idx] = bin_idx
            self.numeric_probabilities[col_idx] = p

        return self

    def _generate(self, n: int):
        rng = np.random.default_rng(self.random_state)

        syn = pd.DataFrame(index=range(n))
        for col in self.columns:
            if col in self.categorical_features:
                sampled_indices = rng.choice(
                    len(self.category_values[col]),
                    size=n,
                    replace=True,
                    p=self.category_probabilities[col],
                )
                syn[col] = self.category_values[col][sampled_indices]
            else:
                col_idx = self.columns.index(col)
                bin_idx = rng.choice(self.numeric_bins[col_idx], p=self.numeric_probabilities[col_idx], size=n).astype(int)
                edges = self.num_enc.bin_edges_[col_idx]
                u = rng.random(n)
                syn[col] = edges[bin_idx] + u * (edges[bin_idx + 1] - edges[bin_idx])

        return syn[self.columns]

    def _state(self):
        return {
            "columns": self.columns,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "category_values": self.category_values,
            "category_probabilities": self.category_probabilities,
            "numeric_bins": self.numeric_bins,
            "numerica_probabilities": self.numeric_probabilities,
        }
