# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
import pandas as pd
from .model import arf
import numpy as np

from ..base import BaseGenerator
from ..persistence import load_generator_state, restore_generator, save_generator_state


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
        random_state (int): Random seed for reproducibility. Default: 0.

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
        ...     early_stop=True,
        ...     random_state=42
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
    ):
        self.random_state = random_state
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size
        self.retain_value_ranges = retain_value_ranges

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        xx = X.copy()
        xx[discrete_features] = xx[discrete_features].astype(str)
        self.numerical_features = [
            col for col in xx.columns if col not in discrete_features
        ]
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

    def save(self, path):
        state = {
            "model": self.model,
            "retain_value_ranges": self.retain_value_ranges,
            "value_ranges": getattr(self, "value_ranges", None),
        }
        return save_generator_state(path, state)

    @classmethod
    def load(cls, path):
        return restore_generator(cls, load_generator_state(path))
