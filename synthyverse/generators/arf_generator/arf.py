import pandas as pd
from arfpy.arf import arf

from ..base import BaseGenerator


class ARFGenerator(BaseGenerator):
    name = "arf"

    def __init__(self, num_trees: int = 20, random_state: int = 0):
        super().__init__(random_state=random_state)
        self.num_trees = num_trees
        self.random_state = random_state

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        X[discrete_features] = X[discrete_features].astype(str)
        self.model = arf(X, num_trees=self.num_trees, random_state=self.random_state)
        self.model.forde()

    def _generate_data(self, n: int):
        return self.model.forge(n)
