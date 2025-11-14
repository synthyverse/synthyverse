from ..base import TabularBaseGenerator
import pandas as pd
import numpy as np


class PermutationGenerator(TabularBaseGenerator):
    """Permutation-based generator for tabular synthetic data.

    Generates synthetic data by randomly permuting a fraction of values in each
    column of the training data. This is a simple baseline generator, mostly used for testing purposes.

    Args:
        random_state (int): Random seed for reproducibility. Default: 0.
        permutation_rate (float): Fraction of values to permute in each column (0.0 to 1.0). Default: 0.5.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

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

    def __init__(self, random_state: int = 0, permutation_rate: float = 0.5, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.permutation_rate = permutation_rate
        self.random_state = random_state

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        self.X = X.copy()

    def _generate_data(self, n: int):
        if n <= len(self.X):
            syn = self.X[:n]
        else:
            syn = self.X
            remaining_rows = n - len(self.X)
            while remaining_rows > 0:
                syn = pd.concat([syn, self.X])
                remaining_rows -= len(self.X)
            syn = syn[:n]

        syn = syn.reset_index(drop=True)

        for col in self.X.columns:
            mask = np.random.rand(n) < self.permutation_rate
            permuted = (
                self.X[col]
                .sample(n, replace=True, random_state=self.random_state)
                .values
            )
            syn.loc[mask, col] = permuted[mask]

        return syn
