import pandas as pd
from arfpy.arf import arf

from ..base import BaseGenerator


class ARFGenerator(BaseGenerator):
    name = "arf"

    def __init__(
        self,
        num_trees: int = 20,
        delta: float = 0.0,
        max_iters: int = 10,
        early_stop: bool = True,
        verbose: bool = True,
        min_node_size: int = 5,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.model_params = {
            "num_trees": num_trees,
            "delta": delta,
            "max_iters": max_iters,
            "early_stop": early_stop,
            "verbose": verbose,
            "min_node_size": min_node_size,
            "random_state": random_state,
        }

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        X[discrete_features] = X[discrete_features].astype(str)
        self.model = arf(X, **self.model_params)
        self.model.forde()

    def _generate_data(self, n: int):
        return self.model.forge(n)
