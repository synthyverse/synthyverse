# Third-party notice: this wrapper requires the BSL-licensed ctgan package.
# See THIRD_PARTY_NOTICES.md and LICENSES/CTGAN-BSL-1.1.txt.
from typing import Optional

from ...utils.utils import resolve_epochs_from_training_steps

import pandas as pd

from .._optional import require_ctgan
from ..base import BaseGenerator
from ..persistence import load_generator_state, save_generator_state


class TVAEGenerator(BaseGenerator):
    """Tabular Variational Autoencoder (TVAE).

    Similar to CTGAN; uses mode-specific normalization for numerical columns.

    Uses the implementation from the ctgan package, which is also used in the Synthetic Data Vault.

    Paper: "Modeling tabular data using conditional gan" by Xu et al. (2019).

    Args:
        embedding_dim (int): Dimension of the embedding layer. Default: 128.
        compress_dims (tuple): Tuple of dimensions for encoder layers. Default: (128, 128).
        decompress_dims (tuple): Tuple of dimensions for decoder layers. Default: (128, 128).
        l2scale (float): L2 regularization scale. Default: 1e-5.
        batch_size (int): Batch size for training. Default: 500.
        epochs (int): Number of training epochs. Default: 300.
        training_steps (int, optional): Total number of training steps. When
            provided, this overrides ``epochs`` by deriving the epoch count from
            the training sample size and batch size. Default: None.
        loss_factor (int): Loss factor for Beta-VAE. Default: 2.
        cuda (bool): Whether to use CUDA if available. Default: True.
        verbose (bool): Whether to print training progress. Default: True.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        log_steps (int): Steps between timeout checks. Default: 100.
        random_state (int): Random seed for reproducibility. Default: 0.

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
        training_steps=None,
        loss_factor=2,
        cuda=True,
        verbose=True,
        cap_train_time: Optional[float] = None,
        log_steps: int = 100,
        random_state: int = 0,
    ):
        require_ctgan()
        self.random_state = random_state
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_steps = training_steps
        self.loss_factor = loss_factor
        self.cuda = cuda
        self.verbose = verbose
        self.cap_train_time = cap_train_time
        self.log_steps = log_steps

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        from .synthesizer import TVAE

        epochs = resolve_epochs_from_training_steps(
            self.epochs,
            self.training_steps,
            len(X),
            self.batch_size,
        )

        self.model = TVAE(
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            verbose=self.verbose,
            epochs=epochs,
            cuda=self.cuda,
            loss_factor=self.loss_factor,
            cap_train_time=self.cap_train_time,
            log_steps=self.log_steps,
        )

        self.model.fit(X, discrete_features)

        return self

    def _generate(self, n: int):
        return self.model.sample(n)

    def save(self, path):
        return save_generator_state(
            path,
            {
                "model": self.model,
            },
        )

    @classmethod
    def load(cls, path):
        require_ctgan()
        state = load_generator_state(path)
        generator = cls.__new__(cls)
        if isinstance(state, dict):
            generator.__dict__.update(state)
        else:
            generator.model = state
        return generator
