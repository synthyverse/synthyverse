import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from ..base import BaseGenerator
from ..persistence import load_generator_state, restore_generator, save_generator_state


class UnivariateGenerator(BaseGenerator):
    """Univariate baseline generator for tabular synthetic data.

    Generates each feature independently. Categorical features are sampled from
    their empirical category frequencies. Numerical features are fitted with
    :class:`sklearn.preprocessing.QuantileTransformer` and sampled by drawing
    uniform values followed by inverse transformation.

    Args:
        random_state (int): Random seed for reproducibility. Default: 0.
        n_quantiles (int): Maximum number of quantiles used by each numerical
            ``QuantileTransformer``. The effective value is capped at the number
            of non-missing observations per feature. Default: 1000.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import UnivariateGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = UnivariateGenerator(random_state=42)
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "univariate"

    def __init__(self, random_state: int = 0, n_quantiles: int = 1000):
        if n_quantiles < 1:
            raise ValueError("n_quantiles must be at least 1.")
        self.random_state = random_state
        self.n_quantiles = n_quantiles

    def _fit(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        self.columns = X.columns.tolist()
        self.categorical_features = [
            col for col in self.columns if col in discrete_features
        ]
        self.numerical_features = [
            col for col in self.columns if col not in self.categorical_features
        ]
        self.category_values = {}
        self.category_probabilities = {}
        self.quantile_transformers = {}
        self.numeric_missing_rates = {}

        for col in self.categorical_features:
            frequencies = X[col].value_counts(normalize=True, dropna=False)
            self.category_values[col] = frequencies.index.to_numpy()
            self.category_probabilities[col] = frequencies.to_numpy()

        for col in self.numerical_features:
            values = X[col].dropna().to_numpy().reshape(-1, 1)
            if len(values) == 0:
                raise ValueError(
                    f"Column {col} has only missing values and cannot be fitted."
                )
            n_quantiles = min(self.n_quantiles, len(values))
            transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="uniform",
                random_state=self.random_state,
            )
            transformer.fit(values)
            self.quantile_transformers[col] = transformer
            self.numeric_missing_rates[col] = float(X[col].isna().mean())

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

            uniform_samples = rng.random((n, 1))
            sampled_values = (
                self.quantile_transformers[col]
                .inverse_transform(uniform_samples)
                .ravel()
            )
            missing_rate = self.numeric_missing_rates[col]
            if missing_rate > 0:
                missing_mask = rng.random(n) < missing_rate
                sampled_values = sampled_values.astype(float, copy=False)
                sampled_values[missing_mask] = np.nan
            syn[col] = sampled_values

        return syn[self.columns]

    def save(self, path):
        state = {
            "random_state": self.random_state,
            "n_quantiles": self.n_quantiles,
            "columns": self.columns,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "category_values": self.category_values,
            "category_probabilities": self.category_probabilities,
            "quantile_transformers": self.quantile_transformers,
            "numeric_missing_rates": self.numeric_missing_rates,
        }
        return save_generator_state(path, state)

    @classmethod
    def load(cls, path):
        return restore_generator(cls, load_generator_state(path))
