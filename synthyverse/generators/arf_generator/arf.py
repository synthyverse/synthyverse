# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
import pandas as pd
from .model import arf
import numpy as np

from ..base import BaseGenerator


class ARFGenerator(BaseGenerator):
    """Adversarial Random Forest (ARF).

    ARF leverages random forests in alternating rounds of generation/discrimination to estimate densities and generate synthetic data.

    Uses the implementation from the arfpy package (https://github.com/bips-hb/arfpy/) with some minor modifications to ensure robustness for resampling from leafs with 1 unique value.

    Paper: "Adversarial random forests for density estimation and generative modeling" by Watson et al. (2023).

    Args:
        num_trees (int): Number of trees in the random forests. Default: 20.
        delta (float): Tolerance parameter for convergence. Default: 0.0.
        max_iters (int): Maximum number of adversarial iterations. Default: 10.
        early_stop (bool): Whether to use early stopping. Default: True.
        verbose (bool): Whether to print training progress. Default: True.
        min_node_size (int): Minimum leaf node samples in trees. Default: 5.
        retain_value_ranges (bool): Whether to clip numerical features to training
            ranges after generation. Default: False.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import ARFGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = ARFGenerator(
        ...     num_trees=50,
        ...     max_iters=10,
        ...     early_stop=True
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "arf"

    def __init__(
        self,
        num_trees: int = 20,
        delta: float = 0.0,
        max_iters: int = 10,
        early_stop: bool = True,
        verbose: bool = True,
        min_node_size: int = 5,
        retain_value_ranges: bool = False,  # whether to retain numerical feature ranges
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size
        self.retain_value_ranges = retain_value_ranges

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        xx = X.copy()
        self.discrete_features = list(discrete_features)
        self.numerical_features = [
            col for col in xx.columns if col not in self.discrete_features
        ]
        if self.discrete_features:
            xx[self.discrete_features] = xx[self.discrete_features].astype("category")

        if self.retain_value_ranges:
            self.value_ranges = {}
            for col in self.numerical_features:
                self.value_ranges[col] = {
                    "min": xx[col].min(),
                    "max": xx[col].max(),
                }

        self.model = arf(
            xx,
            num_trees=self.num_trees,
            delta=self.delta,
            max_iters=self.max_iters,
            early_stop=self.early_stop,
            verbose=self.verbose,
            min_node_size=self.min_node_size,
            random_state=self.random_state,
        )
        self.model.forde()

        return self

    def _generate(self, n: int):
        syn = self.model.forge(n)

        if self.retain_value_ranges:
            for col in self.value_ranges.keys():
                syn[col] = np.clip(
                    syn[col],
                    self.value_ranges[col]["min"],
                    self.value_ranges[col]["max"],
                )

        return syn

    def _state(self):
        return {
            "model": self.model,
            "num_trees": self.num_trees,
            "delta": self.delta,
            "max_iters": self.max_iters,
            "early_stop": self.early_stop,
            "verbose": self.verbose,
            "min_node_size": self.min_node_size,
            "retain_value_ranges": self.retain_value_ranges,
            "value_ranges": getattr(self, "value_ranges", None),
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
            "ordinal_encoder": self.ordinal_encoder,
        }
