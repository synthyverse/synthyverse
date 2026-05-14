import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from mostlyai import engine

from ..base import BaseGenerator
from ..persistence import load_generator_state, restore_generator, save_generator_state


SAVED_WORKSPACE_DIR = "workspace"


class TabARGNGenerator(BaseGenerator):
    """Tabular AutoRegressive Generative Network (TabARGN).

    Uses the implementation from the MostlyAI engine.

    Paper: "TabularARGN: A Flexible and Efficient Auto-Regressive Framework for Generating High-Fidelity Synthetic Data" by Tiwald et al. (2025).

    Args:
        workspace (str, optional): Directory for storing intermediate files.
            If omitted, an internal temporary workspace is created.
        max_epochs (int): Maximum number of training epochs. Default: 100.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabARGNGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = TabARGNGenerator(
        ...     max_epochs=100,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabargn"
    needs_workspace = False

    def __init__(
        self,
        workspace: Optional[str] = None,
        max_epochs: int = 100,
        random_state: int = 0,
    ):
        self.random_state = random_state
        self.workspace = workspace or tempfile.mkdtemp(prefix="synthyverse_tabargn_")
        self.max_epochs = max_epochs

    def _fit(
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

        return self

    def _generate(self, n: int):
        engine.generate(workspace_dir=self.workspace, sample_size=n)
        syn = pd.read_parquet(f"{self.workspace}/SyntheticData")
        return syn

    def save(self, path):
        path = Path(path)
        workspace_dir = path / SAVED_WORKSPACE_DIR
        self._copy_workspace(Path(self.workspace), workspace_dir)
        return save_generator_state(
            path,
            {
                "workspace": SAVED_WORKSPACE_DIR,
                "max_epochs": getattr(self, "max_epochs", 100),
                "random_state": getattr(self, "random_state", 0),
            },
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        state = load_generator_state(path)
        workspace = state.get("workspace")
        if workspace is not None and not Path(workspace).is_absolute():
            state["workspace"] = str(path / workspace)
        state.setdefault("max_epochs", 100)
        state.setdefault("random_state", 0)
        return restore_generator(cls, state)

    @staticmethod
    def _copy_workspace(source: Path, destination: Path):
        source = source.resolve()
        destination = destination.resolve()

        if not source.exists():
            raise FileNotFoundError(
                f"TabARGN workspace does not exist and cannot be saved: {source}"
            )
        if source == destination:
            return

        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
