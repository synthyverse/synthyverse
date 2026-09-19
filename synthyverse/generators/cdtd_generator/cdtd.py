# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
from pathlib import Path

import pandas as pd
import torch
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..base import BaseGenerator
from ..dgm_utils import (
    FastTensorDataLoader,
    QuantileStandardScaler,
    clone_state_dict,
    split_validation,
    validate_c2st,
)
from .layers import MLP, MixedTypeDiffusion
from .utils import LinearScheduler, cycle
from ...utils.utils import get_total_trainable_params

from typing import Optional
import time


class CDTDGenerator(BaseGenerator):
    """Continuous Diffusion for mixed-type Tabular Data (CDTD).

    CDTD uses continuous diffusion for mixed-type tabular data. It provides several improvements to homogenize data types in the modelling process.

    Uses the simple wrapper implementation from the original paper's authors (https://github.com/muellermarkus/cdtd_simple)

    Paper: "Continuous Diffusion for Mixed-Type Tabular Data" by Mueller et al. (2023).

    Args:
        cat_emb_dim (int): Embedding dimension for categorical features. Default: 16.
        embedding_dim (int): Embedding dimension for MLP layers. Default: 256.
        mlp_n_layers (int): Number of MLP layers. Default: 5.
        mlp_n_units (int): Number of units per MLP layer. Default: 1024.
        sigma_data_cat (float): Data sigma for categorical features. Default: 1.0.
        sigma_data_cont (float): Data sigma for continuous features. Default: 1.0.
        sigma_min_cat (float): Minimum sigma for categorical features. Default: 0.0.
        sigma_min_cont (float): Minimum sigma for continuous features. Default: 0.0.
        sigma_max_cat (float): Maximum sigma for categorical features. Default: 100.0.
        sigma_max_cont (float): Maximum sigma for continuous features. Default: 80.0.
        cat_emb_init_sigma (float): Initial sigma for categorical embeddings. Default: 0.001.
        timewarp_type (str): Type of time warping. Options: "single", "bytype", "all". Default: "bytype".
        timewarp_weight_low_noise (float): Weight for low noise in time warping. Default: 3.0.
        training_steps (int): Number of training steps (iterations, not epochs). Default: 30000.
        num_steps_warmup (int): Number of warmup steps. Default: 1000.
        num_timesteps (int): Number of sampling timesteps. Default: 200.
        batch_size (int): Batch size for training. Default: 4096.
        lr (float): Learning rate. Default: 1e-3.
        ema_decay (float): Exponential moving average decay. Default: 0.999.
        log_steps (int): Steps between logging. Default: 100.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        target_column (str, optional): Column used for stratified validation
            splitting. Default: None.
        val_size (float): Fraction of rows reserved for C2ST validation. Set
            to <=0 to disable C2ST early stopping. Default: -1.
        val_steps (int): Steps between validation C2ST checks. Set to <=0 to
            disable validation. Default: -1.
        patience (int): Number of consecutive non-improving C2ST validation
            checks before early stopping. Default: 3.
        max_validation_rows (int): Maximum number of rows reserved for C2ST
            validation. Negative values remove the cap. Extra rows remain in the training set. Default: 30000.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import CDTDGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator
        >>> generator = CDTDGenerator(
        ...     timewarp_type="bytype",
        ...     training_steps=30000
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "cdtd"
    supports_purely_numerical = False
    supports_purely_categorical = False

    def __init__(
        self,
        cat_emb_dim: int = 16,
        embedding_dim: int = 256,
        mlp_n_layers: int = 5,
        mlp_n_units: int = 1024,
        sigma_data_cat: float = 1.0,
        sigma_data_cont: float = 1.0,
        sigma_min_cat: float = 0.0,
        sigma_min_cont: float = 0.0,
        sigma_max_cat: float = 100.0,
        sigma_max_cont: float = 80.0,
        cat_emb_init_sigma: float = 0.001,
        timewarp_type: str = "bytype",
        timewarp_weight_low_noise: float = 3.0,
        training_steps: int = 30_000,
        num_steps_warmup: int = 1000,
        num_timesteps: int = 200,
        batch_size: int = 4096,
        lr: float = 1e-3,
        ema_decay: float = 0.999,
        log_steps: int = 100,
        random_state: int = 0,
        full_determinism: bool = False,
        cap_train_time: Optional[float] = None,
        target_column: Optional[str] = None,
        val_size: float = -1,
        val_steps: int = -1,
        patience: int = 3,
        max_validation_rows: int = 30_000,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.cat_emb_dim = cat_emb_dim
        self.embedding_dim = embedding_dim
        self.mlp_n_layers = mlp_n_layers
        self.mlp_n_units = mlp_n_units
        self.sigma_data_cat = sigma_data_cat
        self.sigma_data_cont = sigma_data_cont
        self.sigma_min_cat = sigma_min_cat
        self.sigma_min_cont = sigma_min_cont
        self.sigma_max_cat = sigma_max_cat
        self.sigma_max_cont = sigma_max_cont
        self.cat_emb_init_sigma = cat_emb_init_sigma
        self.timewarp_type = timewarp_type
        self.timewarp_weight_low_noise = timewarp_weight_low_noise
        self.training_steps = training_steps
        self.num_steps_warmup = num_steps_warmup
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.lr = lr
        self.ema_decay = ema_decay
        self.log_steps = log_steps
        self.cap_train_time = cap_train_time
        self.target_column = target_column
        self.val_size = val_size
        self.val_steps = val_steps
        self.patience = patience
        self.max_validation_rows = max_validation_rows

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        X = X.copy()
        if self.val_size >= 1:
            raise ValueError("CDTD requires val_size to be less than 1.")
        if self.patience < 1:
            raise ValueError("CDTD requires patience to be at least 1.")
        X_val = None
        if self.val_size > 0 and self.val_steps > 0:
            X, X_val = split_validation(
                X,
                self.val_size,
                self.target_column,
                random_state=self.random_state,
                max_validation_rows=self.max_validation_rows,
            )
            X = X.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True) if X_val is not None else None

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

        X_discrete = torch.tensor(X_train[self.discrete_features].to_numpy()).long()
        X_numerical = torch.tensor(X_train[self.numerical_features].to_numpy()).float()

        # --- build diffusion model ---
        self.num_cat_features = X_discrete.shape[1]
        self.num_cont_features = X_numerical.shape[1]
        num_features = self.num_cat_features + self.num_cont_features

        categories = self._categorical_cardinalities(self.discrete_features)
        self.categories = categories

        proportions = []
        n_sample = X_discrete.shape[0]
        for i in range(len(categories)):
            counts = torch.bincount(X_discrete[:, i], minlength=categories[i])
            proportions.append(counts / n_sample)
        self.proportions = proportions

        score_model = MLP(
            self.num_cont_features,
            self.cat_emb_dim,
            categories,
            proportions,
            self.embedding_dim,
            self.mlp_n_layers,
            self.mlp_n_units,
        )

        self.diff_model = MixedTypeDiffusion(
            model=score_model,
            dim=self.cat_emb_dim,
            categories=categories,
            num_features=num_features,
            sigma_data_cat=self.sigma_data_cat,
            sigma_data_cont=self.sigma_data_cont,
            sigma_min_cat=self.sigma_min_cat,
            sigma_max_cat=self.sigma_max_cat,
            sigma_min_cont=self.sigma_min_cont,
            sigma_max_cont=self.sigma_max_cont,
            proportions=proportions,
            cat_emb_init_sigma=self.cat_emb_init_sigma,
            timewarp_type=self.timewarp_type,
            timewarp_weight_low_noise=self.timewarp_weight_low_noise,
        )

        # --- train ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_batch_size = min(self.batch_size, X_discrete.shape[0])
        train_loader = FastTensorDataLoader(
            X_discrete,
            X_numerical,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_iter = cycle(train_loader)

        self.diff_model = self.diff_model.to(self.device)
        self.diff_model.train()

        ema_diff_model = ExponentialMovingAverage(
            self.diff_model.parameters(), decay=self.ema_decay
        )

        optimizer = torch.optim.AdamW(
            self.diff_model.parameters(), lr=self.lr, weight_decay=0
        )
        scheduler = LinearScheduler(
            self.training_steps,
            base_lr=self.lr,
            final_lr=1e-6,
            warmup_steps=self.num_steps_warmup,
            warmup_begin_lr=1e-6,
            anneal_lr=True,
        )

        current_step = 0
        n_obs = sum_loss = 0

        best_val_score = float("inf")
        best_val_model = None
        bad_val_steps = 0
        use_validation = X_val is not None

        def validate():
            nonlocal best_val_score, best_val_model, bad_val_steps
            self.diff_model.eval()
            ema_diff_model.store()
            ema_diff_model.copy_to()
            score = validate_c2st(self, X_val, random_state=self.random_state)

            if score < best_val_score:
                best_val_score = score
                best_val_model = clone_state_dict(self.diff_model)
                bad_val_steps = 0
            else:
                bad_val_steps += 1

            ema_diff_model.restore()
            self.diff_model.train()
            return bad_val_steps >= self.patience

        train_time = 0.0
        timed_out = False
        self.trained_steps_ = 0
        with tqdm(initial=current_step, total=self.training_steps) as pbar:
            while current_step < self.training_steps:
                step_start_time = time.monotonic()
                optimizer.zero_grad()

                inputs = next(train_iter)
                x_cat, x_cont = (
                    inp.to(self.device) if inp is not None else None for inp in inputs
                )

                losses, _ = self.diff_model.loss_fn(x_cat, x_cont, None)
                losses["train_loss"].backward()

                optimizer.step()
                self.diff_model.timewarp_cdf.update_ema()
                ema_diff_model.update()

                sum_loss += losses["train_loss"].detach().mean().item() * x_cat.shape[0]
                n_obs += x_cat.shape[0]
                current_step += 1
                self.trained_steps_ = current_step
                pbar.update(1)

                if current_step % self.log_steps == 0:
                    # check loss and early stopping at log step
                    pbar.set_description(
                        f"Loss (last {self.log_steps} steps): {(sum_loss / n_obs):.3f}"
                    )
                    n_obs = sum_loss = 0

                for param_group in optimizer.param_groups:
                    param_group["lr"] = scheduler(current_step)

                train_time += time.monotonic() - step_start_time
                if self.cap_train_time is not None and train_time > self.cap_train_time:
                    print(f"Training timed out after {self.cap_train_time} seconds.")
                    timed_out = True
                    break

                if (
                    use_validation
                    and current_step % self.val_steps == 0
                    and current_step > 0
                ):
                    if validate():
                        break

        if timed_out and use_validation:
            validate()

        if best_val_model is None:
            ema_diff_model.copy_to()
        else:
            self.diff_model.load_state_dict(
                {k: v.to(self.device) for k, v in best_val_model.items()}
            )
        self.diff_model.eval()
        return self

    def _generate(self, n: int):
        self.diff_model.eval()

        n_batches, remainder = divmod(n, self.batch_size)
        sample_sizes = (
            n_batches * [self.batch_size] + [remainder]
            if remainder != 0
            else n_batches * [self.batch_size]
        )

        x_cat_list = []
        x_cont_list = []

        for num_samples in tqdm(sample_sizes):
            cat_latents = torch.randn(
                (num_samples, self.num_cat_features, self.cat_emb_dim),
                device=self.device,
            )
            cont_latents = torch.randn(
                (num_samples, self.num_cont_features), device=self.device
            )
            x_cat_gen, x_cont_gen = self.diff_model.sampler(
                cat_latents, cont_latents, self.num_timesteps
            )
            x_cat_list.append(x_cat_gen)
            x_cont_list.append(x_cont_gen)

        x_cat = torch.cat(x_cat_list).cpu()
        x_cont = torch.cat(x_cont_list).cpu()

        syn_X_discrete = x_cat.long().numpy()
        syn_X_numerical = x_cont.numpy()

        syn_X_numerical = self.quant_encoder.inverse_transform(syn_X_numerical)

        syn_X = pd.concat(
            (pd.DataFrame(syn_X_discrete), pd.DataFrame(syn_X_numerical)), axis=1
        )
        syn_X.columns = self.discrete_features + self.numerical_features
        syn_X = syn_X[self.col_order]

        return syn_X

    def get_trainable_params(self):
        return get_total_trainable_params(self.diff_model)

    def get_trained_steps_epochs(self):
        return {"trained_steps": self.trained_steps_}

    def _state(self):
        return {
            "cat_emb_dim": self.cat_emb_dim,
            "embedding_dim": self.embedding_dim,
            "mlp_n_layers": self.mlp_n_layers,
            "mlp_n_units": self.mlp_n_units,
            "sigma_data_cat": self.sigma_data_cat,
            "sigma_data_cont": self.sigma_data_cont,
            "sigma_min_cat": self.sigma_min_cat,
            "sigma_min_cont": self.sigma_min_cont,
            "sigma_max_cat": self.sigma_max_cat,
            "sigma_max_cont": self.sigma_max_cont,
            "cat_emb_init_sigma": self.cat_emb_init_sigma,
            "timewarp_type": self.timewarp_type,
            "timewarp_weight_low_noise": self.timewarp_weight_low_noise,
            "num_timesteps": self.num_timesteps,
            "batch_size": self.batch_size,
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
            "col_order": self.col_order,
            "ordinal_encoder": self.ordinal_encoder,
            "quant_encoder": self.quant_encoder,
            "num_cat_features": self.num_cat_features,
            "num_cont_features": self.num_cont_features,
            "categories": self.categories,
            "proportions": self.proportions,
            "target_column": self.target_column,
            "val_size": self.val_size,
            "val_steps": self.val_steps,
            "patience": self.patience,
            "max_validation_rows": self.max_validation_rows,
            "trained_steps_": self.trained_steps_,
        }

    def _save_extra(self, path: Path) -> None:
        torch.save(self.diff_model.state_dict(), path / "diff_model.pt")

    def _load_extra(self, path: Path) -> None:
        self.num_timesteps = getattr(self, "num_timesteps", 200)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        score_model = MLP(
            self.num_cont_features,
            self.cat_emb_dim,
            self.categories,
            self.proportions,
            self.embedding_dim,
            self.mlp_n_layers,
            self.mlp_n_units,
        )
        self.diff_model = MixedTypeDiffusion(
            model=score_model,
            dim=self.cat_emb_dim,
            categories=self.categories,
            num_features=self.num_cat_features + self.num_cont_features,
            sigma_data_cat=self.sigma_data_cat,
            sigma_data_cont=self.sigma_data_cont,
            sigma_min_cat=self.sigma_min_cat,
            sigma_max_cat=self.sigma_max_cat,
            sigma_min_cont=self.sigma_min_cont,
            sigma_max_cont=self.sigma_max_cont,
            proportions=self.proportions,
            cat_emb_init_sigma=self.cat_emb_init_sigma,
            timewarp_type=self.timewarp_type,
            timewarp_weight_low_noise=self.timewarp_weight_low_noise,
        ).to(self.device)
        state_dict = torch.load(path / "diff_model.pt", map_location=self.device)
        self.diff_model.load_state_dict(state_dict)
        self.diff_model.eval()
