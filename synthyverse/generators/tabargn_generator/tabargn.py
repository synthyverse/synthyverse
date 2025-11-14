from ..base import TabularBaseGenerator

import pandas as pd
from mostlyai import engine


class TabARGNGenerator(TabularBaseGenerator):
    """Tabular AutoRegressive Generative Network (TabARGN).

    TabARGN uses masked transformers for tabular data generation.
    We use the implementation from the MostlyAI engine.
    Paper: "TabularARGN: A Flexible and Efficient Auto-Regressive Framework for Generating High-Fidelity Synthetic Data" by Tiwald et al. (2025).

    Args:
        workspace (str): Directory for storing intermediate files.
        max_epochs (int): Maximum number of training epochs. Default: 100.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabARGNGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator (requires workspace)
        >>> generator = TabARGNGenerator(
        ...     workspace="./tabargn_workspace",
        ...     max_epochs=100,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabargn"
    needs_workspace = True

    def __init__(
        self,
        workspace: str,
        max_epochs: int = 100,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.workspace = workspace
        self.max_epochs = max_epochs

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        # set up workspace and default logging
        engine.init_logging()

        # execute the engine steps
        engine.split(  # split data as PQT files for `trn` + `val` to `{ws}/OriginalData/tgt-data`
            workspace_dir=self.workspace,
            tgt_data=X,
            model_type="TABULAR",
        )
        engine.analyze(workspace_dir=self.workspace)
        engine.encode(workspace_dir=self.workspace)
        engine.train(workspace_dir=self.workspace, max_epochs=self.max_epochs)

    def _generate_data(self, n: int):
        engine.generate(workspace_dir=self.workspace, sample_size=n)
        syn = pd.read_parquet(f"{self.workspace}/SyntheticData")
        return syn
