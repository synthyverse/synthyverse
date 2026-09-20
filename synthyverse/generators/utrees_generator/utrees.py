import numpy as np
import pandas as pd

from ..base import BaseGenerator
from ._utrees.unmasking_trees import UnmaskingTrees


class UTreesGenerator(BaseGenerator):
    """XGBoost-based any-order autoregressive model.

    Based on the utrees python package: https://github.com/calvinmccarter/unmasking-trees/.

    Paper: "Unmasking Trees for Tabular Data" by McCarter et al. (2024).

    Args:
        depth: Balanced tree depth for numerical columns. Default: 4.
        duplicate_K: Masking orders per training row. Default: 50.
        clf_kwargs: Arguments passed to XGBoost classifiers. Default: {}.
        strategy: Numerical quantization strategy. Default: "kdiquantile".
        softmax_temp: Sampling temperature. Default: 1.0.
        cast_float32: Cast estimator inputs to float32. Default: True.
        random_state: Training seed. Default: 0.
        full_determinism: Request deterministic backend behavior. Default: False.
    """

    name = "utrees"

    def __init__(
        self,
        depth: int = 4,
        duplicate_K: int = 50,
        clf_kwargs: dict = {},
        strategy: str = "kdiquantile",
        softmax_temp: float = 1.0,
        cast_float32: bool = True,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.depth = depth
        self.duplicate_K = duplicate_K
        self.clf_kwargs = clf_kwargs
        self.strategy = strategy
        self.softmax_temp = softmax_temp
        self.cast_float32 = cast_float32

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.columns = X.columns
        quantize_cols = [
            (
                "categorical"
                if col in discrete_features
                else (
                    "integer" if pd.api.types.is_integer_dtype(X[col]) else "continuous"
                )
            )
            for col in X.columns
        ]
        self.model = UnmaskingTrees(
            depth=self.depth,
            duplicate_K=self.duplicate_K,
            clf_kwargs=self.clf_kwargs,
            strategy=self.strategy,
            softmax_temp=self.softmax_temp,
            cast_float32=self.cast_float32,
            random_state=self.random_state,
        ).fit(X.to_numpy(dtype=np.float64), quantize_cols=quantize_cols)

    def _generate(self, n: int):
        self.model.random_state_.seed(self.random_state)
        return pd.DataFrame(self.model.generate(n), columns=self.columns)
