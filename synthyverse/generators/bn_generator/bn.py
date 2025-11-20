import pandas as pd
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins

from ..base import TabularBaseGenerator


class BNGenerator(TabularBaseGenerator):
    """Bayesian Network (BN).

    Uses Bayesian networks to model dependencies between variables and generate
    synthetic data by sampling from the learned joint distribution.

    Uses the implementation from Synthcity (https://github.com/vanderschaarlab/synthcity/).

    Args:
        struct_learning_n_iter (int): Number of iterations for DAG learning. Default: 1000.
        struct_learning_search_method (str): Search method for DAG learning.
            Options: "hillclimb", "pc", "tree_search", "mmhc", "exhaustive". Default: "tree_search".
        struct_learning_score (str): Scoring function for DAG learning.
            Options: "k2", "bdeu", "bic", "bds". Default: "k2".
        struct_max_indegree (int): Maximum number of parents for each node. Decrease to reduce computational overhead. Default: 4.
        encoder_max_clusters (int): Maximum clusters for encoding continuous variables. Default: 10.
        encoder_noise_scale (float): Noise scale for encoding. Default: 0.1.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import BNGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = BNGenerator(
        ...     struct_learning_search_method="tree_search",
        ...     struct_learning_score="k2",
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "bn"

    def __init__(
        self,
        struct_learning_n_iter: int = 1000,
        struct_learning_search_method: str = "tree_search",  # hillclimb, pc, tree_search, mmhc, exhaustive
        struct_learning_score: str = "k2",  # k2, bdeu, bic, bds
        struct_max_indegree: int = 4,
        encoder_max_clusters: int = 10,
        encoder_noise_scale: float = 0.1,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.model = Plugins().get(
            "bayesian_network",
            struct_learning_n_iter=struct_learning_n_iter,
            struct_learning_search_method=struct_learning_search_method,
            struct_learning_score=struct_learning_score,
            struct_max_indegree=struct_max_indegree,
            encoder_max_clusters=encoder_max_clusters,
            encoder_noise_scale=encoder_noise_scale,
            random_state=random_state,
        )

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        loader = GenericDataLoader(
            X,
            target_column=self.target_column,
            train_size=1,
            random_state=self.random_state,
        )
        self.model.fit(loader)

    def _generate_data(self, n: int):
        syn = self.model.generate(n)
        return syn.dataframe()
