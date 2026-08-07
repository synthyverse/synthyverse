# Third-party notice: this wrapper requires the BSL-licensed ctgan package.
# See THIRD_PARTY_NOTICES.md and LICENSES/CTGAN-BSL-1.1.txt.
from typing import Optional

import pandas as pd

from ...utils.utils import resolve_epochs_from_training_steps
from ..dgm_utils import clone_state_dict, split_validation, validate_c2st
from .._optional import require_ctgan
from ..base import BaseGenerator


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
        loss_factor (int): Multiplier applied to the reconstruction loss term.
            Default: 2.
        cuda (bool): Whether to use CUDA if available. Default: True.
        verbose (bool): Whether to print training progress. Default: True.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        val_size (float): Fraction of training rows reserved for validation set early stopping. Default: 0.0.
        val_steps (int): Epochs between validation, or training steps when ``training_steps`` is provided. Default: 50.
        target_column (str): Name of the target column, potentially used for stratified validation splitting. Default: None.

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
        ...     cuda=True
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
        val_size: float = 0.0,
        val_steps: int = 50,
        target_column: Optional[str] = None,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        require_ctgan()
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
        self.val_size = val_size
        self.val_steps = val_steps
        self.target_column = target_column

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        from .synthesizer import TVAE

        self.discrete_features = list(discrete_features)
        X, X_val = split_validation(
            X,
            self.val_size,
            self.target_column,
            self.discrete_features,
            self.random_state,
        )

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
        )

        best_val_score = float("inf")
        best_val_model = None

        def validate():
            nonlocal best_val_score, best_val_model
            self.model.decoder.eval()
            score = validate_c2st(self, X_val, random_state=self.random_state)
            self.model.decoder.train()
            if score < best_val_score:
                best_val_score = score
                best_val_model = clone_state_dict(self.model.decoder)
                return False
            return True

        def validate_callback(step, epoch, epoch_end):
            if X_val is None:
                return False
            if self.training_steps is not None:
                return (
                    not epoch_end
                    and step > 0
                    and step % self.val_steps == 0
                    and validate()
                )
            return epoch_end and epoch % self.val_steps == 0 and validate()

        self.model.fit(
            X,
            discrete_features,
            validate_callback=validate_callback,
        )

        if best_val_model is not None:
            self.model.decoder.load_state_dict(
                {k: v.to(self.model._device) for k, v in best_val_model.items()}
            )
        self.model.decoder.eval()

        return self

    def _generate(self, n: int):
        return self.model.sample(n)

    def _state(self):
        return {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "compress_dims": self.compress_dims,
            "decompress_dims": self.decompress_dims,
            "l2scale": self.l2scale,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "training_steps": self.training_steps,
            "loss_factor": self.loss_factor,
            "cuda": self.cuda,
            "verbose": self.verbose,
            "cap_train_time": self.cap_train_time,
            "val_size": self.val_size,
            "val_steps": self.val_steps,
            "target_column": self.target_column,
            "discrete_features": getattr(self, "discrete_features", None),
        }

    @classmethod
    def _restore_state(cls, state):
        require_ctgan()
        generator = cls.__new__(cls)
        if isinstance(state, dict):
            generator.__dict__.update(state)
        else:
            generator.model = state
        return generator
