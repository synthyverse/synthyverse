from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from tqdm import trange

from ..base import BaseGenerator
from ..dgm_utils import FastTensorDataLoader
from ..persistence import load_generator_state, restore_generator, save_generator_state
from ...utils.utils import resolve_epochs_from_training_steps
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


def _quantile_transformer(n: int, seed: int) -> QuantileTransformer:
    return QuantileTransformer(
        n_quantiles=max(min(n // 30, 1000), 10),
        output_distribution="normal",
        subsample=int(1e9),
        random_state=seed,
    )


class TabDDPMGenerator(BaseGenerator):
    """Tabular Denoising Diffusion Probabilistic Model (TabDDPM) generator.

    TabDDPM combines continuous diffusion for numerical features with multinomial diffusion for categorical features.

    We use the implementation from SynthCity, with some modifications to allow manual specification of discrete features.

    Paper: "Tabddpm: Modelling tabular data with diffusion models" by Kotelnikov et al. (2023).

    Args:
        target_column (str): Name of the target column.
        epochs (int): Number of training epochs. Default: 1000.
        training_steps (int, optional): Total number of training steps. When
            provided, this overrides ``epochs`` by deriving the epoch count from
            the training sample size and batch size. Default: None.
        lr (float): Learning rate. Default: 0.002.
        weight_decay (float): Weight decay for optimization. Default: 1e-4.
        batch_size (int): Batch size for training. Default: 1024.
        num_timesteps (int): Number of diffusion timesteps. Default: 1000.
        gaussian_loss_type (str): Type of Gaussian loss. Options: "mse", "kl". Default: "mse".
        scheduler (str): Learning rate scheduler type. Options: "cosine", "linear". Default: "cosine".
        log_interval (int): Steps between logging. Default: 100.
        model_params (dict): Dictionary of model parameters. Default: {"n_layers_hidden": 3, "n_units_hidden": 256, "dropout": 0.0}.
        embedding_dim (int): Embedding dimension. Default: 128.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabDDPMGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator (requires target column)
        >>> generator = TabDDPMGenerator(
        ...     target_column="target",
        ...     epochs=1000,
        ...     scheduler="cosine",
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabddpm"

    def __init__(
        self,
        target_column: str,
        epochs: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        log_interval: int = 100,
        model_params: dict = {},
        embedding_dim: int = 128,
        random_state: int = 0,
        training_steps: int = None,
    ):
        self.epochs = epochs
        self.training_steps = training_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.model_params = model_params
        self.embedding_dim = embedding_dim
        self.target_column = target_column
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_columns = X.columns
        self.is_classification = self.target_column in discrete_features

        cond = None
        train = X.copy()
        discrete_columns = [x for x in discrete_features if x != self.target_column]
        if self.is_classification:
            train = train.drop(columns=[self.target_column])
            cond = X[self.target_column]
            self.target_name = cond.name
            self._labels, counts = np.unique(cond, return_counts=True)
            self._cond_dist = counts / counts.sum()

        train = self._fit_transform(train, discrete_columns)
        self._fit_diffusion(train, cond, discrete_columns)
        return self

    def _fit_transform(self, X: pd.DataFrame, discrete_columns: list) -> pd.DataFrame:
        self.discrete_columns = list(discrete_columns)
        self.column_dtypes = X.infer_objects().dtypes
        self.feature_names = X.columns
        self.quantile_transformers = {}

        out = X.copy()
        for col in out.columns:
            if col in self.discrete_columns:
                continue
            transformer = _quantile_transformer(len(out), self.random_state)
            out[col] = transformer.fit_transform(out[[col]]).reshape(-1)
            self.quantile_transformers[col] = transformer
        return out

    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, transformer in self.quantile_transformers.items():
            out[col] = transformer.inverse_transform(out[[col]]).reshape(-1)
        return out.astype(self.column_dtypes)

    def _fit_diffusion(
        self, X: pd.DataFrame, cond: Optional[pd.Series], discrete_columns: list
    ) -> None:
        cat_info = [(col, vals.nunique()) for col, vals in X[discrete_columns].items()]
        if cat_info:
            cat_cols, cat_counts = zip(*cat_info)
            num_cols = X.columns.difference(cat_cols)
            X = X[list(num_cols) + list(cat_cols)]
            self.feature_names_out = X.columns
        else:
            cat_cols, cat_counts = [], [0]
            self.feature_names_out = self.feature_names

        if self.is_classification and cond is not None:
            self.n_classes = cond.nunique()
        else:
            self.n_classes = 0

        y = (
            torch.tensor([torch.nan] * len(X), dtype=torch.float32, device=self.device)
            if cond is None
            else torch.tensor(
                cond.values,
                dtype=torch.long if self.is_classification else torch.float32,
                device=self.device,
            )
        )
        self.dataloader = FastTensorDataLoader(
            torch.tensor(X.values, dtype=torch.float32, device=self.device),
            y,
            batch_size=self.batch_size,
        )

        self.diffusion = GaussianMultinomialDiffusion(
            model_params=self.model_params.copy(),
            num_categorical_features=cat_counts,
            num_numerical_features=X.shape[1] - len(cat_cols),
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            num_classes=self.n_classes,
            conditional=cond is not None,
            dim_emb=self.embedding_dim,
            scheduler=self.scheduler,
            device=self.device,
        ).to(self.device)

        self.ema_model = deepcopy(self.diffusion.denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_history = []

        steps = curr_count = 0
        curr_loss_multi = curr_loss_gauss = 0.0
        epochs = resolve_epochs_from_training_steps(
            self.epochs,
            self.training_steps,
            len(X),
            self.batch_size,
        )
        pbar = trange(epochs, desc="Epoch", leave=True)

        for epoch in pbar:
            self.diffusion.train()
            for x, y in self.dataloader:
                self.optimizer.zero_grad()
                args = (x,) if cond is None else (x, y)
                loss_multi, loss_gauss = self.diffusion.mixed_loss(*args)
                loss = loss_multi + loss_gauss
                loss.backward()
                self.optimizer.step()
                self._anneal_lr(epoch + 1, epochs)

                curr_count += len(x)
                curr_loss_multi += loss_multi.item() * len(x)
                curr_loss_gauss += loss_gauss.item() * len(x)
                steps += 1

                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                loss_value = mloss + gloss
                self._update_ema(
                    self.ema_model.parameters(), self.diffusion.parameters()
                )

                if steps % self.log_interval == 0:
                    self.loss_history.append([steps, mloss, gloss, loss_value])
                    curr_count = 0
                    curr_loss_multi = curr_loss_gauss = 0.0

            self.diffusion.eval()
            pbar.set_postfix(loss=loss_value)

        self.loss_history = pd.DataFrame(
            self.loss_history, columns=["step", "mloss", "gloss", "loss"]
        ).set_index("step")

    def _anneal_lr(self, epoch: int, epochs: int) -> None:
        lr = self.lr * (1 - epoch / epochs)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_ema(
        self, target_params: Iterator, source_params: Iterator, rate: float = 0.999
    ) -> None:
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

    def _generate(self, n: int):
        cond = None
        if self.is_classification:
            cond = np.random.choice(self._labels, size=n, p=self._cond_dist)
            cond_tensor = torch.tensor(cond, dtype=torch.long, device=self.device)
        else:
            cond_tensor = None

        self.diffusion.eval()
        sample = self.diffusion.sample_all(n, cond_tensor).detach().cpu().numpy()
        df = pd.DataFrame(sample, columns=self.feature_names_out)
        df = self._inverse_transform(df[self.feature_names])
        if self.is_classification:
            df = df.join(pd.Series(cond, name=self.target_name))
        return df[self.output_columns]

    def save(self, path):
        return save_generator_state(Path(path), self.__dict__)

    @classmethod
    def load(cls, path):
        generator = restore_generator(cls, load_generator_state(path))
        generator.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator.diffusion.to(generator.device)
        generator.diffusion.eval()
        return generator
