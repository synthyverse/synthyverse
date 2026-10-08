from typing import Literal
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_ema import ExponentialMovingAverage

from ..base import BaseGenerator
from .flow_model import ExpVFM
from ..tabdiff_generator.modules import UniModMLP

from ..dgm_utils import (
    FastTensorDataLoader,
    QuantileStandardScaler,
    cpu_state_dict,
    load_state_dict,
    split_validation,
    validate_c2st,
)
from ...utils.utils import get_total_trainable_params, resolve_epochs_from_training_steps

LRScheduler = Literal["reduce_lr_on_plateau", "anneal", "fixed"]
CLossWeightSchedule = Literal["anneal", "fixed"]


class TabbyFlowGenerator(BaseGenerator):
    """TabbyFlow generator for mixed-type tabular data.

    TabbyFlow applies Variational Flow Matching to tabular data.

    Based on the implementation from the original paper: https://github.com/andresguzco/ef-vfm.

    Paper: "Exponential Family Variational Flow Matching for Tabular Data Generation" by Guzman-Cordero et al. (2025).

    Args:
        epochs (int): Number of training epochs. Default: 8000.
        training_steps (int, optional): Total number of training steps. When
            provided, this overrides ``epochs`` by deriving the epoch count from
            the training sample size and batch size. Default: None.
        lr (float): Learning rate. Default: 1e-3.
        weight_decay (float): Weight decay for AdamW. Default: 0.
        batch_size (int): Batch size for training and sampling. Default: 4096.
        num_timesteps (int): Number of Euler sampling steps. Default: 200.
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
        max_grad_nrom (float): Maximum norm for gradient clipping. Disable by setting it to 0. Default: 1.0.
        warmup_epochs (int): Numer of epochs for warming up learning rate linearly. Default: 100.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        target_column (str, optional): Column used for stratified validation
            splitting. Default: None.
        val_size (float): Fraction of rows reserved for C2ST validation. Set
            to <=0 to disable C2ST early stopping. Default: -1.
        val_steps (int): Epochs between validation C2ST checks, or training
            steps when ``training_steps`` is provided. Set to <=0 to disable
            validation. Default: -1.
        patience (int): Number of consecutive non-improving C2ST validation
            checks before early stopping. Default: 3.
        max_validation_rows (int): Maximum number of rows reserved for C2ST
            validation. Negative values remove the cap. Extra rows remain in the training set. Default: 30000.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabbyFlowGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = TabbyFlowGenerator(
        ...     epochs=8000,
        ...     batch_size=4096
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabbyflow"

    def __init__(
        self,
        epochs: int = 8000,
        training_steps: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0,
        batch_size: int = 4096,
        num_timesteps: int = 200,
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
        max_grad_norm: float = 1.0,
        warmup_epochs: int = 100,
        cap_train_time: float | None = None,
        target_column: str | None = None,
        val_size: float = -1,
        val_steps: int = -1,
        patience: int = 3,
        max_validation_rows: int = 30_000,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state, full_determinism)

        self.epochs = epochs
        self.training_steps = training_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
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
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.cap_train_time = cap_train_time
        self.target_column = target_column
        self.val_size = val_size
        self.val_steps = val_steps
        self.patience = patience
        self.max_validation_rows = max_validation_rows
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X, discrete_features):

        if self.val_size >= 1:
            raise ValueError("TabbyFlow requires val_size to be less than 1.")
        if self.patience < 1:
            raise ValueError("TabbyFlow requires patience to be at least 1.")
        X_val = None
        if self.val_size > 0 and self.val_steps > 0:
            X, X_val = split_validation(
                X,
                self.val_size,
                self.target_column,
                discrete_features=discrete_features,
                random_state=self.random_state,
                max_validation_rows=self.max_validation_rows,
            )

        self.discrete_features = discrete_features
        self.numerical_features = [
            col for col in X.columns if col not in discrete_features
        ]
        self.col_order = X.columns
        X_train = X.copy()
        self.quant_encoder = QuantileStandardScaler(X_train.shape[0], self.random_state)
        X_train[self.numerical_features] = self.quant_encoder.fit_transform(
            X_train[self.numerical_features].astype(float)
        )

        X_discrete = torch.tensor(X_train[self.discrete_features].to_numpy())
        X_numerical = torch.tensor(X_train[self.numerical_features].to_numpy())
        X_train = torch.cat((X_numerical, X_discrete), dim=1).float()
        self.d_numerical = X_numerical.shape[1]
        self.categories = (
            np.array(
                self._categorical_cardinalities(self.discrete_features),
                dtype=np.int64,
            )
            if self.discrete_features
            else np.array([], dtype=np.int64)
        )
        train_loader = FastTensorDataLoader(
            X_train, batch_size=self.batch_size, shuffle=True
        )

        self.epochs = resolve_epochs_from_training_steps(
            self.epochs,
            self.training_steps,
            len(X),
            self.batch_size,
        )
        self.flow = self._make_flow()
        self.flow.train()
        ema_model = ExponentialMovingAverage(
            self.flow.parameters(),
            decay=self.ema_decay,  # use_num_updates=False
        )
        best_ema_model = None

        optimizer = torch.optim.AdamW(
            self.flow.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.factor, patience=self.reduce_lr_patience
        )

        def _anneal_lr(step):
            frac_done = step / self.epochs
            lr = self.lr * (1 - frac_done)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        def _run_step(x, closs_weight, dloss_weight):
            x = x.to(self.device)

            self.flow.train()

            optimizer.zero_grad()

            dloss, closs = self.flow.mixed_loss(x)

            loss = dloss_weight * dloss + closs_weight * closs
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.flow.parameters(), self.max_grad_norm
                )
            optimizer.step()

            return dloss, closs

        def compute_loss():
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            data_iter = train_loader
            for batch in data_iter:
                x = batch[0].float().to(self.device)
                self.flow.eval()
                with torch.no_grad():
                    batch_dloss, batch_closs = self.flow.mixed_loss(x)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            return mloss, gloss

        curr_epoch = 0
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_ema_loss = np.inf
        best_val_score = float("inf")
        best_val_model = None
        bad_val_steps = 0
        use_validation = X_val is not None

        def validate():
            nonlocal best_val_score, best_val_model, bad_val_steps
            self.flow.eval()
            try:
                with ema_model.average_parameters():
                    score = validate_c2st(self, X_val, random_state=self.random_state)
                    if score < best_val_score:
                        best_val_score = score
                        best_val_model = cpu_state_dict(self.flow, copy=True)
                        bad_val_steps = 0
                    else:
                        bad_val_steps += 1
            finally:
                self.flow.train()
            return bad_val_steps >= self.patience

        train_time = 0.0
        timed_out = stop_training = False
        self.trained_steps_ = 0
        self.trained_epochs_ = 0

        for epoch in range(curr_epoch, self.epochs):
            curr_epoch = epoch + 1
            # Set up pbar
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

            # Compute the loss weights
            if self.closs_weight_schedule == "fixed":
                pass
            elif self.closs_weight_schedule == "anneal":
                frac_done = epoch / self.epochs
                closs_weight = self.c_lambda * (1 - frac_done)
            else:
                raise NotImplementedError(
                    f"The continuous loss weight schedule {self.closs_weight_schedule} is not implemneted"
                )

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = optimizer.param_groups[0]["lr"]
            for batch in pbar:
                step_start_time = time.monotonic()
                x = batch[0].float().to(self.device)
                batch_dloss, batch_closs = _run_step(x, closs_weight, dloss_weight)
                if self.training_steps is not None:
                    ema_model.update()
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
                pbar.set_postfix(
                    {
                        "lr": curr_lr,
                        "DLoss": np.around(curr_dloss / curr_count, 4),
                        "CLoss": np.around(curr_closs / curr_count, 4),
                        "TotalLoss": np.around(
                            (curr_dloss + curr_closs) / curr_count, 4
                        ),
                        "closs_weight": closs_weight,
                        "dloss_weight": dloss_weight,
                    }
                )
                self.trained_steps_ += 1
                train_time += time.monotonic() - step_start_time
                if self.cap_train_time is not None and train_time > self.cap_train_time:
                    print(f"Training timed out after {self.cap_train_time} seconds.")
                    timed_out = True
                    break
                if (
                    use_validation
                    and self.training_steps is not None
                    and self.trained_steps_ % self.val_steps == 0
                ):
                    if validate():
                        stop_training = True
                        break

            # Log training Loss
            log_dict = {}
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss
            if np.isnan(gloss):
                print("Finding Nan in gaussian loss")
                break
            loss_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/c_loss": gloss,
                "loss/d_loss": mloss,
                "loss/total_loss": total_loss,
            }
            log_dict.update(loss_dict)

            # Adjust learning rate (warmup overrides during early epochs)
            if self.warmup_epochs > 0 and (epoch + 1) <= self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            elif self.lr_scheduler == "reduce_lr_on_plateau":
                scheduler.step(total_loss)
            elif self.lr_scheduler == "anneal":
                _anneal_lr(epoch)
            elif self.lr_scheduler == "fixed":
                pass
            else:
                raise NotImplementedError(
                    f"LR scheduler with name '{self.lr_scheduler}' is not implemented"
                )

            # Update EMA models
            self.trained_epochs_ = curr_epoch
            if self.training_steps is None:
                ema_model.update()
            if timed_out or stop_training:
                break

            if use_validation:
                if self.training_steps is None and curr_epoch % self.val_steps == 0:
                    if validate():
                        break
            else:
                # Select the EMA checkpoint by training loss without validation.
                with ema_model.average_parameters():
                    ema_mloss, ema_gloss = compute_loss()
                    ema_total_loss = ema_mloss + ema_gloss
                    if ema_total_loss < best_ema_loss and curr_epoch > 4000:
                        best_ema_loss = ema_total_loss
                        best_ema_model = cpu_state_dict(self.flow, copy=True)

        if timed_out and use_validation:
            validate()

        if best_val_model is not None:
            self.flow.load_state_dict(best_val_model)
        elif best_ema_model is None:
            ema_model.copy_to()
        else:
            self.flow.load_state_dict(best_ema_model)
        self.flow.eval()
        return self

    def _generate(self, n):
        self.flow.eval()
        with torch.no_grad():
            syn_X = self.flow.sample_all(n, self.batch_size, keep_nan_samples=True)

        syn_X_num, syn_X_cat = (
            syn_X[:, : len(self.numerical_features)],
            syn_X[:, len(self.numerical_features) :],
        )
        syn_X_discrete = syn_X_cat.long().numpy()
        syn_X_numerical = syn_X_num.numpy()
        syn_X_numerical = self.quant_encoder.inverse_transform(syn_X_numerical)

        syn_X = pd.concat(
            (pd.DataFrame(syn_X_discrete), pd.DataFrame(syn_X_numerical)), axis=1
        )
        syn_X.columns = self.discrete_features + self.numerical_features
        syn_X = syn_X[self.col_order]
        return syn_X

    def get_trainable_params(self):
        return get_total_trainable_params(self.flow)

    def get_trained_steps_epochs(self):
        if self.training_steps is not None:
            return {"trained_steps": self.trained_steps_}
        return {"trained_epochs": self.trained_epochs_}

    def _make_flow(self):
        backbone = UniModMLP(
            d_numerical=self.d_numerical,
            categories=(self.categories).tolist(),
            num_layers=self.num_layers,
            d_token=self.d_token,
            n_head=self.n_head,
            factor=self.mlp_factor,
            bias=self.bias,
            embedding_dim=self.embedding_dim,
            mlp_dim=self.mlp_dim,
            mlp_layers=self.mlp_layers,
        )
        flow = ExpVFM(
            num_classes=self.categories,
            num_numerical_features=len(self.numerical_features),
            vf_fn=backbone,
            device=self.device,
            num_timesteps=self.num_timesteps,
        )
        return flow.to(self.device)

    def _state(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"flow", "device"}
        }

    def _save_extra(self, path):
        torch.save(cpu_state_dict(self.flow), path / "flow.pt")

    def _load_extra(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flow = self._make_flow()
        load_state_dict(self.flow, path / "flow.pt")
        self.flow.eval()
