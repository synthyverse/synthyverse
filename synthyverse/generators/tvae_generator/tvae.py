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
        lr (float): Learning rate. Default: 1e-3.
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
        target_column (str, optional): Column used for stratified validation
            splitting. Default: None.
        val_size (float): Fraction of rows reserved for C2ST validation. Set
            to <=0 to disable C2ST early stopping. Default: -1.
        val_steps (int): Epochs between validation C2ST checks, or training
            steps when ``training_steps`` is provided. Set to <=0 to disable
            validation. Default: -1.
        patience (int): Number of consecutive non-improving C2ST validation
            checks before early stopping. Default: 3.
        max_validation_rows (int): Maximum number of rows reserved for C2ST
            validation. Extra rows remain in the training set. Default: 30000.

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
        lr=1e-3,
        batch_size=500,
        epochs=300,
        training_steps=None,
        loss_factor=2,
        cuda=True,
        verbose=True,
        cap_train_time: Optional[float] = None,
        target_column: Optional[str] = None,
        val_size: float = -1,
        val_steps: int = -1,
        patience: int = 3,
        max_validation_rows: int = 30_000,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        require_ctgan()
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_steps = training_steps
        self.loss_factor = loss_factor
        self.cuda = cuda
        self.verbose = verbose
        self.cap_train_time = cap_train_time
        self.target_column = target_column
        self.val_size = val_size
        self.val_steps = val_steps
        self.patience = patience
        self.max_validation_rows = max_validation_rows

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        from .synthesizer import TVAE

        self.discrete_features = list(discrete_features)
        X = X.copy()
        if self.val_size >= 1:
            raise ValueError("TVAE requires val_size to be less than 1.")
        if self.patience < 1:
            raise ValueError("TVAE requires patience to be at least 1.")
        X_val = None
        if self.val_size > 0 and self.val_steps > 0:
            X, X_val = split_validation(
                X,
                self.val_size,
                self.target_column,
                random_state=self.random_state,
                max_validation_rows=self.max_validation_rows,
            )
            X = X.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True) if X_val is not None else None

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
            lr=self.lr,
            batch_size=self.batch_size,
            verbose=self.verbose,
            epochs=epochs,
            cuda=self.cuda,
            loss_factor=self.loss_factor,
            cap_train_time=self.cap_train_time,
        )

        best_val_score = float("inf")
        best_val_model = None
        bad_val_steps = 0
        use_validation = X_val is not None

        def validate():
            nonlocal best_val_score, best_val_model, bad_val_steps
            self.model.decoder.eval()
            score = validate_c2st(self, X_val, random_state=self.random_state)
            self.model.decoder.train()
            if score < best_val_score:
                best_val_score = score
                best_val_model = clone_state_dict(self.model.decoder)
                bad_val_steps = 0
            else:
                bad_val_steps += 1
            return bad_val_steps >= self.patience

        def validate_callback(step, epoch, epoch_end):
            if not use_validation:
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
        if getattr(self.model, "timed_out_", False) and use_validation:
            validate()

        if best_val_model is not None:
            self.model.decoder.load_state_dict(
                {k: v.to(self.model._device) for k, v in best_val_model.items()}
            )
        self.model.decoder.eval()

        return self

    def _generate(self, n: int):
        return self.model.sample(n)

    def get_trainable_params(self):
        return self.model.trainable_params_

    def get_trained_steps_epochs(self):
        if self.training_steps is not None:
            return {"trained_steps": self.model.trained_steps_}
        return {"trained_epochs": self.model.trained_epochs_}

    def _state(self):
        return {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "compress_dims": self.compress_dims,
            "decompress_dims": self.decompress_dims,
            "l2scale": self.l2scale,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "training_steps": self.training_steps,
            "loss_factor": self.loss_factor,
            "cuda": self.cuda,
            "verbose": self.verbose,
            "cap_train_time": self.cap_train_time,
            "target_column": self.target_column,
            "val_size": self.val_size,
            "val_steps": self.val_steps,
            "patience": self.patience,
            "max_validation_rows": self.max_validation_rows,
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
