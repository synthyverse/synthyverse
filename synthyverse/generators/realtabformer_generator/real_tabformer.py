from ..base import BaseGenerator
import pandas as pd
from realtabformer import REaLTabFormer


class RealTabFormerGenerator(BaseGenerator):
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
    ):
        super().__init__(random_state=random_state)

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

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        self.model.fit(X, discrete_features)

    def _generate_data(self, n: int):
        syn = self.model.sample(n, save_samples=False)

        return syn
