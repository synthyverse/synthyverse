from ..base import TabularBaseGenerator
import pandas as pd
from typing import Optional
from nrgboost import Dataset, NRGBooster


class NRGBoostGenerator(TabularBaseGenerator):
    """ENeRgy-based Generative Boosting (NRGBoost).

    Turns gradient-boosted decision trees into energy-based generative models.

    Uses the nrgboost pypi package implementation.

    Paper: "NRGBoost: Energy-Based Generative Boosted Trees" by J. Bravo (2024).

    Args:
        num_trees (int): Number of trees in the boosted ensemble. Default: 200.
        shrinkage (float): Shrinkage parameter for boosting. Default: 0.15.
        line_search (bool): Whether to use line search for step size optimization. Default: True.
        max_leaves (int): Maximum number of leaves per tree. Default: 256.
        max_ratio_in_leaf (float): Maximum ratio of data / model data per leaf. Default: 2.
        min_data_in_leaf (float): Minimum data points per leaf. Default: 0.
        initial_uniform_mixture (float): Mixture coeficient for the starting point of boosting: 0 means starting from the product of training marginals, 1 means starting from a uniform distribution. Default: 0.1.
        categorical_split_one_vs_all (bool): Whether to use one-vs-all splitting for categorical features. Default: False.
        feature_frac (float): Fraction of features to randomly consider for splitting each node. Default: 1.
        splitter (str): Determines how trees are grown. "best" is best first and "depth" is breadth first. Default: "best".
        num_steps (int): Number of Gibbs sampling steps. Default: 100.
        num_sampling_rounds (Optional[int]): Include only first n trees when sampling. Default: None.
        temperature (float): Temperature parameter for sampling. Default: 1.0.
        num_sampling_threads (int): Number of threads for parallel sampling (0 for openmp default). Default: 0.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import NRGBoostGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = NRGBoostGenerator(
        ...     num_trees=200,
        ...     shrinkage=0.15,
        ...     num_steps=100,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "nrgboost"

    def __init__(
        self,
        num_trees: int = 200,
        shrinkage: float = 0.15,
        line_search: bool = True,
        max_leaves: int = 256,
        max_ratio_in_leaf: float = 2,
        min_data_in_leaf: float = 0,
        initial_uniform_mixture: float = 0.1,
        categorical_split_one_vs_all: bool = False,
        feature_frac: float = 1,
        splitter: str = "best",
        num_steps: int = 100,
        num_sampling_rounds: Optional[int] = None,
        temperature: float = 1.0,
        num_sampling_threads: int = 0,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.training_params = {
            "num_trees": num_trees,
            "shrinkage": shrinkage,
            "line_search": line_search,
            "max_leaves": max_leaves,
            "max_ratio_in_leaf": max_ratio_in_leaf,
            "min_data_in_leaf": min_data_in_leaf,
            "initial_uniform_mixture": initial_uniform_mixture,
        }

        self.sampling_params = {
            "seed": random_state,
            "num_steps": num_steps,
            "num_rounds": num_sampling_rounds,
            "temperature": temperature,
            "num_threads": num_sampling_threads,
        }

        self.random_state = random_state

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        xx = X.copy()
        xx[discrete_features] = xx[discrete_features].astype("category")

        self.model = NRGBooster.fit(
            Dataset(xx), params=self.training_params, seed=self.random_state
        )

    def _generate_data(self, n: int):
        syn = self.model.sample(n, **self.sampling_params)
        return syn
