from ctgan import TVAE
from ctgan.synthesizers.tvae import Encoder
from ..base import TabularBaseGenerator
import pandas as pd

from ...utils.utils import get_total_trainable_params


class TVAEGenerator(TabularBaseGenerator):
    """Tabular Variational Autoencoder (TVAE).

    Similar to CTGAN.
    Conditions on discrete columns, and uses mode-specific normalization for numerical columns.
    Uses the implementation from the ctgan package, which is also used in the Synthetic Data Vault.
    Paper: "Modeling tabular data using conditional gan" by Xu et al. (2019).

    Args:
        embedding_dim (int): Dimension of the embedding layer. Default: 128.
        compress_dims (tuple): Tuple of dimensions for encoder layers. Default: (128, 128).
        decompress_dims (tuple): Tuple of dimensions for decoder layers. Default: (128, 128).
        l2scale (float): L2 regularization scale. Default: 1e-5.
        batch_size (int): Batch size for training. Default: 500.
        epochs (int): Number of training epochs. Default: 300.
        loss_factor (int): Loss factor for Beta-VAE. Default: 2.
        cuda (bool): Whether to use CUDA if available. Default: True.
        verbose (bool): Whether to print training progress. Default: True.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TVAEGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = TVAEGenerator(
        ...     embedding_dim=128,
        ...     epochs=300,
        ...     cuda=True,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

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
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.cuda = cuda
        self.verbose = verbose

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        self.model = TVAE(
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            verbose=self.verbose,
            epochs=self.epochs,
            cuda=self.cuda,
        )

        self.model.fit(X, discrete_features)

        decoder_params = get_total_trainable_params(self.model.decoder)
        encoder_params = get_total_trainable_params(
            Encoder(
                self.model.transformer.output_dimensions,
                self.model.compress_dims,
                self.model.embedding_dim,
            )
        )

        print(f"Number of decoder params: {decoder_params}")
        print(f"Number of encoder params: {encoder_params}")
        print(f"Total number of params: {decoder_params + encoder_params}")

    def _generate_data(self, n: int):
        return self.model.sample(n)
