from ..base import BaseGenerator

import pandas as pd
from mostlyai import engine


class TabARGNGenerator(BaseGenerator):
    name = "tabargn"
    needs_workspace = True

    def __init__(
        self,
        workspace: str,
        max_epochs: int = 100,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.workspace = workspace
        self.max_epochs = max_epochs

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):

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
