# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ..base import BaseGenerator
from ...utils.utils import (
    get_total_trainable_params,
    resolve_epochs_from_training_steps,
)
from ..dgm_utils import (
    FastTensorDataLoader,
    QuantileStandardScaler,
    clone_state_dict,
    split_validation,
    validate_c2st,
)
from .diffusion import UnifiedCtimeDiffusion
from .modules import Model, UniModMLP

LRScheduler = Literal["reduce_lr_on_plateau", "anneal", "fixed"]
CLossWeightSchedule = Literal["anneal", "fixed"]
NetConditioning = Literal["sigma", "t"]


def update_ema(target_params, source_params, rate):
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


class TabDiffGenerator(BaseGenerator):
    """TabDiff generator for mixed-type tabular data.

    TabDiff uses a joint continuous-time diffusion process for numerical and categorical features, with
    feature-wise learnable noise schedules.

    Based on the implementation from the original paper: https://github.com/MinkaiXu/TabDiff/.

    Paper: "Tabdiff: a mixed-type diffusion model for tabular data generation" by Shi et al. (2025).

    Args:
        epochs (int): Number of training epochs. Default: 8000.
        training_steps (int, optional): Total number of training steps. When
            provided, this overrides ``epochs`` by deriving the epoch count from
            the training sample size and batch size. Default: None.
        lr (float): Learning rate. Default: 1e-3.
        weight_decay (float): Weight decay for AdamW. Default: 0.
        batch_size (int): Batch size for training and sampling. Default: 4096.
        ema_decay (float): Exponential moving average decay. Default: 0.997.
        lr_scheduler (str): Learning rate scheduler. Options:
            "reduce_lr_on_plateau" lowers the learning rate when training loss
            plateaus, using ``reduce_lr_patience`` and ``factor``; "anneal"
            linearly decays the learning rate to 0 over training; "fixed"
            keeps the initial learning rate. Default: "reduce_lr_on_plateau".
        reduce_lr_patience (int): Plateau scheduler patience. Default: 50.
        factor (float): Multiplicative factor for plateau learning rate decay. Default: 0.90.
        closs_weight_schedule (str): Continuous loss weight schedule. Options:
            "anneal" linearly reduces the numerical-feature loss weight to 0
            over training; "fixed" keeps it at ``c_lambda``. Default: "anneal".
        c_lambda (float): Weight for the continuous loss. Default: 1.0.
        d_lambda (float): Weight for the discrete loss. Default: 1.0.
        num_layers (int): Number of backbone layers. Default: 2.
        d_token (int): Token dimension in the backbone. Default: 4.
        n_head (int): Number of attention heads. Default: 1.
        mlp_factor (int): MLP expansion factor. Default: 32.
        bias (bool): Whether to use bias terms in the backbone. Default: True.
        embedding_dim (int): Projection and time embedding dimension. Default: 1024.
        mlp_dim (int): Hidden width of the denoiser MLP. Default: 2048.
        mlp_layers (int): Number of hidden denoiser MLP layers with width
            ``mlp_dim``. Default: 2.
        num_timesteps (int): Number of sampling timesteps. Default: 50.
        learnable_noise_schedules (bool): Whether to learn feature-wise
            numerical and categorical noise schedules. Default: True.
        stochastic_sampler (bool): Whether to use stochastic sampling. Default: True.
        second_order_correction (bool): Whether to use second-order sampler correction.
            Default: True.
        precond (bool): Whether to use EDM-style preconditioning. Default: True.
        sigma_data (float): Data sigma used by preconditioning. Default: 1.0.
        net_conditioning (str): Conditioning type passed to the denoising
            network when ``precond=True``. Options: "sigma" passes the EDM
            log-sigma conditioning value; "t" passes the raw diffusion time.
            Default: "sigma".
        sigma_min (float): Minimum sigma for numerical noise schedules. Default: 0.002.
        sigma_max (float): Maximum sigma for numerical noise schedules. Default: 80.
        rho (float): Rho parameter for numerical noise schedules. Default: 7.
        eps_max (float): Maximum epsilon for categorical noise schedules. Default: 1e-3.
        eps_min (float): Minimum epsilon for categorical noise schedules. Default: 1e-5.
        rho_init (float): Initial rho for learned numerical schedules. Default: 7.0.
        rho_offset (float): Rho offset for learned numerical schedules. Default: 5.0.
        k_init (float): Initial k for learned categorical schedules. Default: -6.0.
        k_offset (float): K offset for learned categorical schedules. Default: 1.0.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        target_column (str, optional): Column used for stratified validation
            splitting. Default: None.
        val_size (float): Fraction of rows reserved for C2ST validation. Set
            to <=0 to disable C2ST early stopping. Default: 0.2.
        val_steps (int): Epochs between validation C2ST checks, or training
            steps when ``training_steps`` is provided. Set to <=0 to disable
            validation. Default: 1000.
        patience (int): Number of consecutive non-improving C2ST validation
            checks before early stopping. Default: 3.
        max_validation_rows (int): Maximum number of rows reserved for C2ST
            validation. Extra rows remain in the training set. Default: 30000.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabDiffGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = TabDiffGenerator(
        ...     epochs=8000,
        ...     batch_size=4096
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabdiff"

    def __init__(
        self,
        epochs: int = 8000,
        training_steps: int = None,
        lr: float = 1e-3,
        weight_decay: float = 0,
        batch_size: int = 4096,
        ema_decay: float = 0.997,
        lr_scheduler: LRScheduler = "reduce_lr_on_plateau",
        reduce_lr_patience: int = 50,
        factor: float = 0.90,
        closs_weight_schedule: CLossWeightSchedule = "anneal",
        c_lambda: float = 1.0,
        d_lambda: float = 1.0,
        num_layers: int = 2,
        d_token: int = 4,
        n_head: int = 1,
        mlp_factor: int = 32,
        bias: bool = True,
        embedding_dim: int = 1024,
        mlp_dim: int = 2048,
        mlp_layers: int = 2,
        num_timesteps: int = 50,
        learnable_noise_schedules: bool = True,
        stochastic_sampler: bool = True,
        second_order_correction: bool = True,
        precond: bool = True,
        sigma_data: float = 1.0,
        net_conditioning: NetConditioning = "sigma",
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        eps_max: float = 1e-3,
        eps_min: float = 1e-5,
        rho_init: float = 7.0,
        rho_offset: float = 5.0,
        k_init: float = -6.0,
        k_offset: float = 1.0,
        cap_train_time: Optional[float] = None,
        target_column: Optional[str] = None,
        val_size: float = 0.2,
        val_steps: int = 1000,
        patience: int = 3,
        max_validation_rows: int = 30_000,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.epochs = epochs
        self.training_steps = training_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.ema_decay = ema_decay
        self.lr_scheduler = lr_scheduler
        self.reduce_lr_patience = reduce_lr_patience
        self.factor = factor
        self.closs_weight_schedule = closs_weight_schedule
        self.c_lambda = c_lambda
        self.d_lambda = d_lambda
        self.num_layers = num_layers
        self.d_token = d_token
        self.n_head = n_head
        self.mlp_factor = mlp_factor
        self.bias = bias
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.num_timesteps = num_timesteps
        self.learnable_noise_schedules = learnable_noise_schedules
        self.stochastic_sampler = stochastic_sampler
        self.second_order_correction = second_order_correction
        self.precond = precond
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.rho_init = rho_init
        self.rho_offset = rho_offset
        self.k_init = k_init
        self.k_offset = k_offset
        self.cap_train_time = cap_train_time
        self.target_column = target_column
        self.val_size = val_size
        self.val_steps = val_steps
        self.patience = patience
        self.max_validation_rows = max_validation_rows
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        X = X.copy()
        if self.val_size >= 1:
            raise ValueError("TabDiff requires val_size to be less than 1.")
        if self.patience < 1:
            raise ValueError("TabDiff requires patience to be at least 1.")
        should_validate = self.val_size > 0 and self.val_steps > 0
        if should_validate and self.max_validation_rows <= 0:
            raise ValueError("TabDiff requires max_validation_rows to be positive.")
        if (
            should_validate
            and self.target_column is not None
            and self.target_column not in X
        ):
            raise ValueError(f"target_column '{self.target_column}' is not in X.")
        X_val = None
        if should_validate and len(X) > 1:
            X, X_val = split_validation(
                X,
                self.val_size,
                self.target_column,
                random_state=self.random_state,
                max_validation_rows=self.max_validation_rows,
            )
            X = X.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True) if X_val is not None else None

        epochs = resolve_epochs_from_training_steps(
            self.epochs,
            self.training_steps,
            len(X),
            self.batch_size,
        )
        self.col_order = X.columns
        self.discrete_features = list(discrete_features)
        self.numerical_features = [
            c for c in X.columns if c not in self.discrete_features
        ]
        x_num, x_cat = self._fit_transform(X)
        x = torch.from_numpy(np.concatenate([x_num, x_cat], axis=1)).float()

        self.d_numerical = x_num.shape[1]
        self.categories = (
            np.array(
                self._categorical_cardinalities(self.discrete_features),
                dtype=np.int64,
            )
            if self.discrete_features
            else np.array([], dtype=np.int64)
        )
        self.diffusion = self._make_diffusion().to(self.device)
        self.trainable_params_ = get_total_trainable_params(self.diffusion)

        train_loader = FastTensorDataLoader(
            None,
            x,
            batch_size=min(self.batch_size, len(x)),
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.factor, patience=self.reduce_lr_patience
        )
        ema_model = deepcopy(self.diffusion._denoise_fn)
        ema_num_schedule = deepcopy(self.diffusion.num_schedule)
        ema_cat_schedule = deepcopy(self.diffusion.cat_schedule)
        for model in [ema_model, ema_num_schedule, ema_cat_schedule]:
            for param in model.parameters():
                param.detach_()

        best_val_score = float("inf")
        best_val_model = None
        bad_val_steps = 0
        use_validation = self.val_steps > 0 and X_val is not None

        def validate():
            nonlocal best_val_score, best_val_model, bad_val_steps
            self.diffusion.eval()
            score = validate_c2st(self, X_val, random_state=self.random_state)
            if score < best_val_score:
                best_val_score = score
                best_val_model = clone_state_dict(self.diffusion)
                bad_val_steps = 0
            else:
                bad_val_steps += 1
            return bad_val_steps >= self.patience

        train_time = 0.0
        timed_out = False
        stop_training = False
        step = 0
        self.trained_steps_ = 0
        self.trained_epochs_ = 0
        for epoch in tqdm(range(epochs)):
            closs_weight = self.c_lambda
            if self.closs_weight_schedule == "anneal":
                closs_weight *= 1 - epoch / epochs
            elif self.closs_weight_schedule != "fixed":
                raise NotImplementedError(self.closs_weight_schedule)

            dloss_sum = closs_sum = n_obs = 0
            self.diffusion.train()
            for _, batch in train_loader:
                step_start_time = time.monotonic()
                batch = batch.float().to(self.device)
                optimizer.zero_grad()
                dloss, closs = self.diffusion.mixed_loss(batch)
                loss = self.d_lambda * dloss + closs_weight * closs
                loss.backward()
                optimizer.step()
                dloss_sum += dloss.item() * len(batch)
                closs_sum += closs.item() * len(batch)
                n_obs += len(batch)
                step += 1
                self.trained_steps_ = step
                train_time += time.monotonic() - step_start_time
                if self.cap_train_time is not None and train_time > self.cap_train_time:
                    print(f"Training timed out after {self.cap_train_time} seconds.")
                    timed_out = True
                    break

                if (
                    use_validation
                    and self.training_steps is not None
                    and step > 0
                    and step % self.val_steps == 0
                ):
                    if validate():
                        stop_training = True
                        break
                    self.diffusion.train()

            if n_obs == 0:
                raise ValueError(
                    "TabDiff training produced no observations in an epoch."
                )
            total_loss = dloss_sum / n_obs + closs_sum / n_obs
            if not np.isfinite(total_loss):
                raise ValueError(
                    "TabDiff training produced a non-finite loss; aborting fit."
                )
            if self.lr_scheduler == "reduce_lr_on_plateau":
                scheduler.step(total_loss)
            elif self.lr_scheduler == "anneal":
                lr = self.lr * (1 - epoch / epochs)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            elif self.lr_scheduler != "fixed":
                raise NotImplementedError(self.lr_scheduler)

            self.trained_epochs_ = epoch + 1
            update_ema(
                ema_model.parameters(),
                self.diffusion._denoise_fn.parameters(),
                self.ema_decay,
            )
            update_ema(
                ema_num_schedule.parameters(),
                self.diffusion.num_schedule.parameters(),
                self.ema_decay,
            )
            update_ema(
                ema_cat_schedule.parameters(),
                self.diffusion.cat_schedule.parameters(),
                self.ema_decay,
            )
            if timed_out or stop_training:
                break

            if (
                use_validation
                and self.training_steps is None
                and (epoch + 1) % self.val_steps == 0
            ):
                if validate():
                    stop_training = True
            if timed_out or stop_training:
                break

        if timed_out and use_validation:
            validate()

        if best_val_model is None:
            self.diffusion._denoise_fn = ema_model
            self.diffusion.num_schedule = ema_num_schedule
            self.diffusion.cat_schedule = ema_cat_schedule
        else:
            self.diffusion.load_state_dict(
                {k: v.to(self.device) for k, v in best_val_model.items()}
            )
        self.diffusion.eval()
        return self

    def _generate(self, n: int):
        self.diffusion.eval()
        with torch.no_grad():
            syn = self.diffusion.sample_all(n, self.batch_size).numpy()

        syn_num = syn[:, : self.d_numerical]
        syn_cat = syn[:, self.d_numerical :].astype(int)
        frames = []
        if self.numerical_features:
            frames.append(
                pd.DataFrame(
                    self.quantile_transformer.inverse_transform(syn_num),
                    columns=self.numerical_features,
                )
            )
        if self.discrete_features:
            frames.append(
                pd.DataFrame(
                    syn_cat,
                    columns=self.discrete_features,
                )
            )
        return pd.concat(frames, axis=1)[self.col_order]

    def get_trainable_params(self):
        if hasattr(self, "trainable_params_"):
            return self.trainable_params_
        return get_total_trainable_params(self.diffusion)

    def get_trained_steps_epochs(self):
        if self.training_steps is not None:
            return {"trained_steps": self.trained_steps_}
        return {"trained_epochs": self.trained_epochs_}

    def _fit_transform(self, X):
        if self.numerical_features:
            self.quantile_transformer = QuantileStandardScaler(
                len(X), self.random_state
            )
            x_num = self.quantile_transformer.fit_transform(
                X[self.numerical_features].to_numpy().astype(float)
            )
        else:
            self.quantile_transformer = None
            x_num = np.empty((len(X), 0), dtype=np.float32)

        if self.discrete_features:
            x_cat = X[self.discrete_features].to_numpy(dtype=np.int64)
        else:
            x_cat = np.empty((len(X), 0), dtype=np.int64)
        return x_num.astype(np.float32), x_cat.astype(np.float32)

    def _make_diffusion(self):
        backbone = UniModMLP(
            d_numerical=self.d_numerical,
            categories=(self.categories + 1).tolist(),
            num_layers=self.num_layers,
            d_token=self.d_token,
            n_head=self.n_head,
            factor=self.mlp_factor,
            bias=self.bias,
            embedding_dim=self.embedding_dim,
            mlp_dim=self.mlp_dim,
            mlp_layers=self.mlp_layers,
        )
        model = Model(
            backbone,
            precond=self.precond,
            sigma_data=self.sigma_data,
            net_conditioning=self.net_conditioning,
        )
        scheduler = (
            "power_mean_per_column" if self.learnable_noise_schedules else "power_mean"
        )
        cat_scheduler = (
            "log_linear_per_column" if self.learnable_noise_schedules else "log_linear"
        )
        return UnifiedCtimeDiffusion(
            num_classes=self.categories,
            num_numerical_features=self.d_numerical,
            denoise_fn=model,
            y_only_model=None,
            num_timesteps=self.num_timesteps,
            scheduler=scheduler,
            cat_scheduler=cat_scheduler,
            noise_dist="uniform",
            edm_params={"sigma_data": self.sigma_data},
            noise_schedule_params={
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "rho": self.rho,
                "eps_max": self.eps_max,
                "eps_min": self.eps_min,
                "rho_init": self.rho_init,
                "rho_offset": self.rho_offset,
                "k_init": self.k_init,
                "k_offset": self.k_offset,
            },
            sampler_params={
                "stochastic_sampler": self.stochastic_sampler,
                "second_order_correction": self.second_order_correction,
            },
            device=self.device,
        )

    def _state(self):
        return {
            "batch_size": self.batch_size,
            "num_layers": self.num_layers,
            "d_token": self.d_token,
            "n_head": self.n_head,
            "mlp_factor": self.mlp_factor,
            "bias": self.bias,
            "embedding_dim": self.embedding_dim,
            "mlp_dim": self.mlp_dim,
            "mlp_layers": self.mlp_layers,
            "num_timesteps": self.num_timesteps,
            "learnable_noise_schedules": self.learnable_noise_schedules,
            "stochastic_sampler": self.stochastic_sampler,
            "second_order_correction": self.second_order_correction,
            "precond": self.precond,
            "sigma_data": self.sigma_data,
            "net_conditioning": self.net_conditioning,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "eps_max": self.eps_max,
            "eps_min": self.eps_min,
            "rho_init": self.rho_init,
            "rho_offset": self.rho_offset,
            "k_init": self.k_init,
            "k_offset": self.k_offset,
            "col_order": self.col_order,
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
            "d_numerical": self.d_numerical,
            "categories": self.categories,
            "quantile_transformer": self.quantile_transformer,
            "ordinal_encoder": self.ordinal_encoder,
            "trained_steps_": self.trained_steps_,
            "trained_epochs_": self.trained_epochs_,
            "trainable_params_": self.trainable_params_,
        }

    def _save_extra(self, path: Path) -> None:
        torch.save(
            {
                "denoise_fn": self.diffusion._denoise_fn.state_dict(),
                "num_schedule": self.diffusion.num_schedule.state_dict(),
                "cat_schedule": self.diffusion.cat_schedule.state_dict(),
            },
            path / "diffusion.pt",
        )

    def _load_extra(self, path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = self._make_diffusion().to(self.device)
        state = torch.load(path / "diffusion.pt", map_location=self.device)
        self.diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
        self.diffusion.num_schedule.load_state_dict(state["num_schedule"])
        self.diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
        self.diffusion.eval()
