# Third-party notice: based on Apache-2.0-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution, NOTICE, and modification details.
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Optional
import time
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from ..base import BaseGenerator
from ..dgm_utils import (
    FastTensorDataLoader,
    QuantileStandardScaler,
    clone_state_dict,
    split_validation,
    validate_c2st,
)
from ...utils.utils import resolve_epochs_from_training_steps
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


class TabDDPMGenerator(BaseGenerator):
    """Tabular Denoising Diffusion Probabilistic Model (TabDDPM) generator.

    TabDDPM combines continuous diffusion for numerical features with multinomial diffusion for categorical features.

    Based on the implementation from the synthcity Python library: https://github.com/vanderschaarlab/synthcity/.

    Paper: "Tabddpm: Modelling tabular data with diffusion models" by Kotelnikov et al. (2023).

    Args:
        target_column (str, optional): Name of the target column. Required when
            ``conditional_generation`` is True. Also used for stratified
            validation splitting when the column is discrete.
        conditional_generation (bool): Whether to condition generation on a
            discrete ``target_column``. Continuous targets are ignored for
            conditioning. Default: False.
        epochs (int): Number of training epochs. Default: 1000.
        training_steps (int, optional): Total number of training steps. When
            provided, this overrides ``epochs`` by deriving the epoch count from
            the training sample size and batch size. Default: None.
        lr (float): Learning rate. Default: 0.002.
        weight_decay (float): Weight decay for optimization. Default: 1e-4.
        batch_size (int): Batch size for training. Default: 1024.
        num_timesteps (int): Number of diffusion timesteps. Default: 1000.
        gaussian_loss_type (str): Type of Gaussian loss. Options: "mse", "kl". Default: "mse".
        scheduler (str): Beta scheduler type. Options: "cosine", "linear". Default: "cosine".
        log_interval (int): Steps between logging. Default: 100.
        model_params (dict): Dictionary of model parameters. When empty, defaults
            to ``{"n_layers_hidden": 3, "n_units_hidden": 256, "dropout": 0.0}``.
            Default: ``{}``.
        embedding_dim (int): Embedding dimension. Default: 128.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        val_size (float): Fraction of training rows reserved for validation set early stopping. Default: 0.0.
        val_steps (int): Epochs between validation, or training steps when ``training_steps`` is provided. Default: 5000.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabDDPMGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = TabDDPMGenerator(
        ...     epochs=1000,
        ...     scheduler="cosine"
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabddpm"

    def __init__(
        self,
        target_column: Optional[str] = None,
        conditional_generation: bool = False,
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
        cap_train_time: Optional[float] = None,
        val_size: float = 0.0,
        val_steps: int = 5000,
        random_state: int = 0,
        full_determinism: bool = False,
        training_steps: int = None,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
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
        self.conditional_generation = conditional_generation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cap_train_time = cap_train_time
        self.val_size = val_size
        self.val_steps = val_steps

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_columns = X.columns
        self.discrete_features = list(discrete_features)
        if self.conditional_generation and self.target_column is None:
            raise ValueError(
                "target_column must be passed when conditional_generation is True."
            )
        if self.conditional_generation and self.target_column not in X.columns:
            raise ValueError(f"target_column '{self.target_column}' is not in X.")
        X, X_val = split_validation(
            X,
            self.val_size,
            self.target_column,
            self.discrete_features,
            self.random_state,
        )
        self.is_conditional = (
            self.conditional_generation and self.target_column in self.discrete_features
        )
        self.is_classification = (
            self.is_conditional and self.target_column in discrete_features
        )

        cond = None
        train = X.copy()
        discrete_columns = list(discrete_features)
        if self.is_conditional:
            train = train.drop(columns=[self.target_column])
            cond = X[self.target_column]
            self.target_name = cond.name
            discrete_columns = [x for x in discrete_features if x != self.target_column]
            if self.is_classification:
                self._labels, cond = np.unique(cond, return_inverse=True)
                counts = np.bincount(cond)
                cond = pd.Series(cond, index=X.index, name=self.target_name)
                self._cond_dist = counts / counts.sum()

        train = self._fit_transform(train, discrete_columns)
        self._fit_diffusion(train, cond, discrete_columns, X_val)
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
            transformer = QuantileStandardScaler(len(out), self.random_state)
            out[col] = transformer.fit_transform(out[[col]]).reshape(-1)
            self.quantile_transformers[col] = transformer
        return out

    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, transformer in self.quantile_transformers.items():
            out[col] = transformer.inverse_transform(out[[col]]).reshape(-1)
        if self.discrete_columns:
            out[self.discrete_columns] = out[self.discrete_columns].round().astype(int)
        return out.astype(self.column_dtypes)

    def _fit_diffusion(
        self,
        X: pd.DataFrame,
        cond: Optional[pd.Series],
        discrete_columns: list,
        X_val: pd.DataFrame = None,
    ) -> None:
        cat_info = [
            (col, self._categorical_cardinality(col)) for col in discrete_columns
        ]
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

        best_val_score = float("inf")
        best_val_model = None
        start_time = time.monotonic()
        timed_out = False
        stop_training = False
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

                if (
                    X_val is not None
                    and self.training_steps is not None
                    and steps > 0
                    and steps % self.val_steps == 0
                ):
                    self.diffusion.eval()
                    score = validate_c2st(self, X_val, random_state=self.random_state)
                    if score < best_val_score:
                        best_val_score = score
                        best_val_model = clone_state_dict(self.diffusion)
                    else:
                        stop_training = True
                        break
                    self.diffusion.train()

                if (
                    self.cap_train_time is not None
                    and time.monotonic() - start_time > self.cap_train_time
                ):
                    print(f"Training timed out after {self.cap_train_time} seconds.")
                    timed_out = True
                    break

            pbar.set_postfix(loss=loss_value)
            if (
                X_val is not None
                and self.training_steps is None
                and (epoch + 1) % self.val_steps == 0
            ):
                self.diffusion.eval()
                score = validate_c2st(self, X_val, random_state=self.random_state)
                if score < best_val_score:
                    best_val_score = score
                    best_val_model = clone_state_dict(self.diffusion)
                else:
                    stop_training = True
            if timed_out or stop_training:
                break
        if best_val_model is not None:
            self.diffusion.load_state_dict(
                {k: v.to(self.device) for k, v in best_val_model.items()}
            )
        self.diffusion.eval()
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
        if self.is_conditional:
            if self.is_classification:
                cond_codes = np.random.choice(
                    len(self._labels), size=n, p=self._cond_dist
                )
                cond = self._labels[cond_codes]
                cond_tensor = torch.tensor(
                    cond_codes, dtype=torch.long, device=self.device
                )
        else:
            cond_tensor = None

        self.diffusion.eval()
        sample = self.diffusion.sample_all(n, cond_tensor).detach().cpu().numpy()
        df = pd.DataFrame(sample, columns=self.feature_names_out)
        df = self._inverse_transform(df[self.feature_names])
        if self.is_conditional:
            df = df.join(pd.Series(cond, name=self.target_name))
        return df[self.output_columns]

    def _state(self):
        state = {
            "batch_size": self.batch_size,
            "output_columns": self.output_columns,
            "target_column": self.target_column,
            "conditional_generation": self.conditional_generation,
            "is_conditional": self.is_conditional,
            "is_classification": self.is_classification,
            "discrete_columns": self.discrete_columns,
            "column_dtypes": self.column_dtypes,
            "feature_names": self.feature_names,
            "quantile_transformers": self.quantile_transformers,
            "ordinal_encoder": self.ordinal_encoder,
            "feature_names_out": self.feature_names_out,
            "diffusion": self.diffusion,
        }
        if self.is_conditional:
            state["target_name"] = self.target_name
        if self.is_classification:
            state.update(
                {
                    "_labels": self._labels,
                    "_cond_dist": self._cond_dist,
                }
            )
        return state

    def _load_extra(self, path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion.to(self.device)
        self.diffusion.eval()
