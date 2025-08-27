from ..base import BaseGenerator
import pandas as pd
import torch
import os
import shutil

from .tabddpm_dir.plugin import TabDDPMPlugin
from synthcity.plugins.core.dataloader import GenericDataLoader


class TabDDPMGenerator(BaseGenerator):
    name = "tabddpm"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        n_iter: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        log_interval: int = 100,
        model_type: str = "mlp",
        model_params: dict = {},
        dim_embed: int = 128,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.model_type = model_type
        self.model_params = model_params
        self.dim_embed = dim_embed
        self.target_column = target_column
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        workspace = "tabddpm_workspace"

        if self.target_column in discrete_features:
            is_classification = True
        else:
            is_classification = False

        loader = GenericDataLoader(data=X, target_column=self.target_column)

        self.model = TabDDPMPlugin(
            is_classification=is_classification,
            n_iter=self.n_iter,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
            gaussian_loss_type=self.gaussian_loss_type,
            scheduler=self.scheduler,
            device=self.device,
            log_interval=self.log_interval,
            model_type=self.model_type,
            model_params=self.model_params,
            dim_embed=self.dim_embed,
            random_state=self.random_state,
            workspace="workspace",
        )
        kwargs = {
            "discrete_columns": [
                x for x in discrete_features if x != self.target_column
            ],
        }
        self.model.fit(loader, **kwargs)
        # delete workspace
        shutil.rmtree(workspace)

    def _generate_data(self, n: int):
        return self.model.generate(n)
