from ctgan import CTGAN
from ctgan.synthesizers.ctgan import Discriminator
from ..base import TabularBaseGenerator
import pandas as pd

from ...utils.utils import get_total_trainable_params


class CTGANGenerator(TabularBaseGenerator):
    """Conditional Tabular GAN (CTGAN).

    Conditions on discrete columns, and uses mode-specific normalization for numerical columns.

    Uses the implementation from the ctgan package, which is also used in the Synthetic Data Vault.

    Paper: "Modeling tabular data using conditional gan" by Xu et al. (2019).

    Args:
        embedding_dim (int): Dimension of the embedding layer. Default: 128.
        generator_dim (tuple): Tuple of dimensions for generator layers. Default: (256, 256).
        discriminator_dim (tuple): Tuple of dimensions for discriminator layers. Default: (256, 256).
        generator_lr (float): Learning rate for generator optimizer. Default: 2e-4.
        generator_decay (float): Weight decay for generator optimizer. Default: 1e-6.
        discriminator_lr (float): Learning rate for discriminator optimizer. Default: 2e-4.
        discriminator_decay (float): Weight decay for discriminator optimizer. Default: 1e-6.
        batch_size (int): Batch size for training. Default: 500.
        discriminator_steps (int): Number of discriminator steps per generator step. Default: 1.
        log_frequency (bool): Whether to log training frequency. Default: True.
        verbose (bool): Whether to print training progress. Default: True.
        epochs (int): Number of training epochs. Default: 300.
        pac (int): Number of samples per class for PAC discriminator. Default: 10.
        cuda (bool): Whether to use CUDA if available. Default: True.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import CTGANGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = CTGANGenerator(
        ...     epochs=300,
        ...     batch_size=500,
        ...     cuda=True,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

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
