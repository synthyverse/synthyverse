import numpy as np
import pandas as pd

from ..base import BaseGenerator


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
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)

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
        self.numeric_ranges = {}

        for col in self.categorical_features:
            frequencies = X[col].value_counts(normalize=True, dropna=False)
            self.category_values[col] = frequencies.index.to_numpy()
            self.category_probabilities[col] = frequencies.to_numpy()

        for col in self.numerical_features:
            self.numeric_ranges[col] = (X[col].min(), X[col].max())

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
                continue

            low, high = self.numeric_ranges[col]
            syn[col] = rng.uniform(low, high, size=n)

        return syn[self.columns]

    def _state(self):
        return {
            "columns": self.columns,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "category_values": self.category_values,
            "category_probabilities": self.category_probabilities,
            "numeric_ranges": self.numeric_ranges,
        }
