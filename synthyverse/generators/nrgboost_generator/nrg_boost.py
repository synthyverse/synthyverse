from ..base import TabularBaseGenerator
import pandas as pd

from nrgboost import Dataset, NRGBooster


class NRGBoostGenerator(TabularBaseGenerator):
    name = "nrgboost"

    def __init__(
        self,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)

        self.random_state = random_state

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):

        xx = X.copy()
        xx[discrete_features] = xx[discrete_features].astype("category")

        self.model = NRGBooster.fit(Dataset(xx), seed=self.random_state)

    def _generate_data(self, n: int):
        syn = self.model.sample(n)
        return syn
