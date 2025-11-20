import pandas as pd
from arfpy.arf import arf
import numpy as np
from ..base import TabularBaseGenerator


class ARFGenerator(TabularBaseGenerator):
    """Adversarial Random Forest (ARF).

    ARF leverages random forests in alternating rounds of generation/discrimination to estimate densities and generate synthetic data.

    Uses the arfpy package implementation.

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
        **kwargs: Additional arguments passed to TabularBaseGenerator.

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
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.retain_value_ranges = retain_value_ranges
        self.model_params = {
            "num_trees": num_trees,
            "delta": delta,
            "max_iters": max_iters,
            "early_stop": early_stop,
            "verbose": verbose,
            "min_node_size": min_node_size,
            "random_state": random_state,
        }

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
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

        self.model = arf(xx, **self.model_params)
        self.model.forde()

    def _generate_data(self, n: int):
        syn = self.model.forge(n)
        if self.retain_value_ranges:
            for col in self.numerical_features:
                syn[col] = np.clip(
                    syn[col],
                    self.value_ranges[col]["min"],
                    self.value_ranges[col]["max"],
                )

        return syn
