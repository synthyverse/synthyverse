from ctgan import CTGAN
from ..base import BaseGenerator
import pandas as pd


class CTGANGenerator(BaseGenerator):
    name = "ctgan"

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        epochs=300,
        pac=10,
        cuda=True,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.model = CTGAN(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
        )

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        self.model.fit(X, discrete_features)

    def _generate_data(self, n: int):
        return self.model.sample(n)
