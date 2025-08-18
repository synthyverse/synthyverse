from ctgan import TVAE
from ..base import BaseGenerator
import pandas as pd


class TVAEGenerator(BaseGenerator):
    name = "tvae"

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=True,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.model = TVAE(
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            batch_size=batch_size,
            verbose=verbose,
            epochs=epochs,
            cuda=cuda,
        )

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        self.model.fit(X, discrete_features)

    def _generate_data(self, n: int):
        return self.model.sample(n)
