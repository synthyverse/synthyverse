from ctgan import CTGAN
from ctgan.synthesizers.ctgan import Discriminator
from ..base import TabularBaseGenerator
import pandas as pd

from ...utils.utils import get_total_trainable_params


class CTGANGenerator(TabularBaseGenerator):
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
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.epochs = epochs
        self.batch_size = batch_size

        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.pac = pac
        self.cuda = cuda

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        # round batch_size to be divisible by pac
        self.batch_size = self.batch_size // self.pac * self.pac

        # TBD: force batch_size to be divisible by pac
        self.model = CTGAN(
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            epochs=self.epochs,
            pac=self.pac,
            cuda=self.cuda,
        )

        self.model.fit(X, discrete_features)

        n_generator_params = get_total_trainable_params(self.model._generator)
        n_discriminator_params = get_total_trainable_params(
            Discriminator(
                self.model._transformer.output_dimensions
                + self.model._data_sampler.dim_cond_vec(),
                self.model._discriminator_dim,
                pac=self.model.pac,
            )
        )
        print(f"Number of generator params: {n_generator_params}")
        print(f"Number of discriminator params: {n_discriminator_params}")
        print(f"Total number of params: {n_generator_params + n_discriminator_params}")

    def _generate_data(self, n: int):
        return self.model.sample(n)
