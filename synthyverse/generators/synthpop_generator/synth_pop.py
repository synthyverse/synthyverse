import pandas as pd
import numpy as np
from synthpop import DataProcessor, MissingDataHandler

from .synthpop_dir.cart import CARTMethod

from ..base import TabularBaseGenerator


class SynthpopGenerator(TabularBaseGenerator):
    name = "synthpop"

    def __init__(
        self,
        smoothing: bool = False,
        proper: bool = False,
        minibucket: int = 5,
        tree_params: dict = {},
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state
        self.tree_params = tree_params

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):

        metadata = MissingDataHandler().get_column_dtypes(X)
        self.processor = DataProcessor(metadata)
        x_pr = self.processor.preprocess(X)

        self.model = CARTMethod(
            metadata,
            smoothing=self.smoothing,
            proper=self.proper,
            minibucket=self.minibucket,
            tree_params=self.tree_params,
            random_state=self.random_state,
        )
        self.model.fit(x_pr)

    def _generate_data(self, n: int):
        syn = self.model.sample(n)
        syn = self.processor.postprocess(syn)
        return syn
