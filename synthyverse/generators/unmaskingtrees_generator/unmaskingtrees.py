from ..base import TabularBaseGenerator
import pandas as pd

from utrees import UnmaskingTrees


class UnmaskingTreesGenerator(TabularBaseGenerator):
    name = "unmaskingtrees"

    def __init__(
        self,
        depth: int = 4,
        duplicate_K: int = 50,
        clf_kwargs: dict = {},
        strategy: str = "kdiquantile",  # quantile, uniform, kmeans
        softmax_temp: float = 1,
        cast_float32: bool = True,
        tabpfn: bool = False,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.model_params = {
            "depth": depth,
            "duplicate_K": duplicate_K,
            "clf_kwargs": clf_kwargs,
            "softmax_temp": softmax_temp,
            "cast_float32": cast_float32,
            "tabpfn": tabpfn,
            "strategy": strategy,
            "random_state": random_state,
        }

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        self.ori_cols = X.columns
        quantize_cols = []
        for col in X.columns:
            if col in discrete_features:
                quantize_cols.append("categorical")
            else:
                quantize_cols.append("continuous")
        self.model = UnmaskingTrees(**self.model_params)
        self.model.fit(X.to_numpy(), quantize_cols)

    def _generate_data(self, n: int):
        syn = self.model.generate(n)
        syn = pd.DataFrame(syn, columns=self.ori_cols)
        return syn
