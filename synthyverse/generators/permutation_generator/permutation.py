import pandas as pd
import numpy as np

from ..base import BaseGenerator
from ..persistence import load_generator_state, restore_generator, save_generator_state


class PermutationGenerator(BaseGenerator):
    """Permutation-based generator for tabular synthetic data.

    Generates synthetic data by randomly permuting a fraction of values in each
    column of the training data. This is a simple baseline generator, mostly used for testing purposes.

    Args:
        random_state (int): Random seed for reproducibility. Default: 0.
        permutation_rate (float): Fraction of values to permute in each column (0.0 to 1.0). Default: 0.5.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import PermutationGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = PermutationGenerator(
        ...     permutation_rate=0.5,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "permutation"

    def __init__(self, random_state: int = 0, permutation_rate: float = 0.5):
        self.permutation_rate = permutation_rate
        self.random_state = random_state

    def _fit(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        self.X = X.copy()

        return self

    def _generate(self, n: int):
        self.rng = np.random.default_rng(self.random_state)

        syn = pd.DataFrame(index=range(n))
        for col in self.X.columns:
            syn[col] = (
                self.X[col].sample(n, replace=True, random_state=self.rng).to_numpy()
            )

        return syn

    def save(self, path):
        state = {
            "X": self.X,
            "permutation_rate": self.permutation_rate,
            "random_state": self.random_state,
        }
        return save_generator_state(path, state)

    @classmethod
    def load(cls, path):
        return restore_generator(cls, load_generator_state(path))
