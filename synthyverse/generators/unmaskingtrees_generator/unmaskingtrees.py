from ..base import TabularBaseGenerator
import pandas as pd

from utrees import UnmaskingTrees


class UnmaskingTreesGenerator(TabularBaseGenerator):
    """Unmasking Trees.

    Unmasking Trees is an autoregressive model which hierarchically partitions features into binary bins,
    to then recursively train XGBoost classifiers along the meta-tree hierarchy.

    We use the implementation from the utrees pypi package. Can be costly for large datasets.

    Paper: "Unmasking trees for tabular data" by C. McCarter (2024).

    Args:
        depth (int): Depth of the meta-tree. Default: 4.
        duplicate_K (int): Number of duplications for each sample. Default: 50.
        xgboost_kwargs (dict): Dictionary of additional XGBoost parameters. Default: {}.
        strategy (str): Strategy for quantization. Options: "quantile", "uniform",
            "kmeans", "kdiquantile". Default: "kdiquantile".
        softmax_temp (float): Temperature for softmax. Default: 1.
        cast_float32 (bool): Whether to cast to float32. Default: True.
        tabpfn (bool): Whether to use TabPFN. Default: False.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import UnmaskingTreesGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = UnmaskingTreesGenerator(
        ...     depth=4,
        ...     duplicate_K=50,
        ...     strategy="kdiquantile",
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "unmaskingtrees"

    def __init__(
        self,
        depth: int = 4,
        duplicate_K: int = 50,
        xgboost_kwargs: dict = {},
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
            "xgboost_kwargs": xgboost_kwargs,
            "softmax_temp": softmax_temp,
            "cast_float32": cast_float32,
            "tabpfn": tabpfn,
            "strategy": strategy,
            "random_state": random_state,
        }

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
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
