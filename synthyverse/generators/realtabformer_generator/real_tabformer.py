from ..base import TabularBaseGenerator
import pandas as pd
from realtabformer import REaLTabFormer


class RealTabFormerGenerator(TabularBaseGenerator):
    """Realistic Relational and Tabular Data using Transformers.

    Fine-tunes GPT-2 for tabular synthetic data generation.
    Uses the realtabformer pypi package implementation.
    Paper: "Realtabformer: Generating realistic relational and tabular data using transformers" by Solatorio et al. (2023).

    Args:
        workspace (str): Directory for storing checkpoints and samples.
        epochs (int): Number of training epochs. Default: 1000.
        batch_size (int): Batch size for training. Default: 8.
        mask_rate (float): Masking rate for training. Default: 0.
        early_stopping_patience (int): Patience for early stopping. Default: 5.
        early_stopping_threshold (float): Threshold for early stopping. Default: 0.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import RealTabFormerGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator (requires workspace)
        >>> generator = RealTabFormerGenerator(
        ...     workspace="./realtabformer_workspace",
        ...     epochs=1000,
        ...     batch_size=8,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "realtabformer"
    needs_workspace = True

    def __init__(
        self,
        workspace: str,
        epochs: int = 1000,
        batch_size: int = 8,
        mask_rate: float = 0,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)

        self.model = REaLTabFormer(
            model_type="tabular",
            checkpoints_dir=workspace,
            samples_save_dir=workspace,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            mask_rate=mask_rate,
            random_state=random_state,
        )

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        self.model.fit(X, discrete_features)

    def _generate_data(self, n: int):
        syn = self.model.sample(n, save_samples=False)

        return syn
