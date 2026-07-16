# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OrdinalEncoder
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..base import BaseGenerator
from ..dgm_utils import FastTensorDataLoader
from .layers import MLP, MixedTypeDiffusion
from .utils import LinearScheduler, cycle, set_seeds
from ..persistence import load_generator_state, restore_generator, save_generator_state
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
        random_state (int): Random seed for reproducibility. Default: 0.

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
        ...     training_steps=30000,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "cdtd"

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
        cap_train_time: Optional[float] = None,
    ):
        self.random_state = random_state
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

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.discrete_features = discrete_features
        self.numerical_features = [
            col for col in X.columns if col not in discrete_features
        ]
        self.col_order = X.columns

        X_discrete = X[self.discrete_features].to_numpy()
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        X_discrete = self.ordinal_encoder.fit_transform(X_discrete)

        X_numerical = X[self.numerical_features].to_numpy().astype(float)
        self.quant_encoder = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X_numerical.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=self.random_state,
        )
        X_numerical = self.quant_encoder.fit_transform(X_numerical)
        self.scaler = StandardScaler()
        X_numerical = self.scaler.fit_transform(X_numerical)

        X_discrete = torch.tensor(X_discrete).long()
        X_numerical = torch.tensor(X_numerical).float()

        # --- build diffusion model ---
        self.num_cat_features = X_discrete.shape[1]
        self.num_cont_features = X_numerical.shape[1]
        num_features = self.num_cat_features + self.num_cont_features

        categories = []
        for i in range(self.num_cat_features):
            categories.append(int(X_discrete[:, i].unique().numel()))
        self.categories = categories

        proportions = []
        n_sample = X_discrete.shape[0]
        for i in range(len(categories)):
            _, counts = X_discrete[:, i].unique(return_counts=True)
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

        print(
            f"Total trainable parameters: {get_total_trainable_params(self.diff_model):,}"
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

        set_seeds(self.random_state, cuda_deterministic=True)
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

        start_time = time.monotonic()
        with tqdm(initial=current_step, total=self.training_steps) as pbar:
            while current_step < self.training_steps:
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
                pbar.update(1)

                if current_step % self.log_steps == 0:
                    # check loss and early stopping at log step
                    pbar.set_description(
                        f"Loss (last {self.log_steps} steps): {(sum_loss / n_obs):.3f}"
                    )
                    n_obs = sum_loss = 0

                for param_group in optimizer.param_groups:
                    param_group["lr"] = scheduler(current_step)

                # check if training timed out
                if (self.cap_train_time is not None) and (
                    current_step % self.log_steps == 0
                ):
                    if (time.monotonic() - start_time) > self.cap_train_time:
                        print(
                            f"Training timed out after {self.cap_train_time} seconds."
                        )
                        break

        ema_diff_model.copy_to()
        self.diff_model.eval()

        return self

    def _generate(self, n: int):
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

        syn_X_numerical = self.scaler.inverse_transform(syn_X_numerical)
        syn_X_numerical = self.quant_encoder.inverse_transform(syn_X_numerical)
        syn_X_discrete = self.ordinal_encoder.inverse_transform(syn_X_discrete)

        syn_X = pd.concat(
            (pd.DataFrame(syn_X_discrete), pd.DataFrame(syn_X_numerical)), axis=1
        )
        syn_X.columns = self.discrete_features + self.numerical_features
        syn_X = syn_X[self.col_order]

        return syn_X

    def save(self, path):
        path = Path(path)
        state = {
            "random_state": self.random_state,
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
            "training_steps": self.training_steps,
            "num_timesteps": self.num_timesteps,
            "batch_size": self.batch_size,
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
            "col_order": self.col_order,
            "ordinal_encoder": self.ordinal_encoder,
            "quant_encoder": self.quant_encoder,
            "scaler": self.scaler,
            "num_cat_features": self.num_cat_features,
            "num_cont_features": self.num_cont_features,
            "categories": self.categories,
            "proportions": self.proportions,
        }
        save_generator_state(path, state)
        torch.save(self.diff_model.state_dict(), path / "diff_model.pt")
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        generator = restore_generator(cls, load_generator_state(path))
        generator.num_timesteps = getattr(generator, "num_timesteps", 200)
        generator.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        score_model = MLP(
            generator.num_cont_features,
            generator.cat_emb_dim,
            generator.categories,
            generator.proportions,
            generator.embedding_dim,
            generator.mlp_n_layers,
            generator.mlp_n_units,
        )
        generator.diff_model = MixedTypeDiffusion(
            model=score_model,
            dim=generator.cat_emb_dim,
            categories=generator.categories,
            num_features=generator.num_cat_features + generator.num_cont_features,
            sigma_data_cat=generator.sigma_data_cat,
            sigma_data_cont=generator.sigma_data_cont,
            sigma_min_cat=generator.sigma_min_cat,
            sigma_max_cat=generator.sigma_max_cat,
            sigma_min_cont=generator.sigma_min_cont,
            sigma_max_cont=generator.sigma_max_cont,
            proportions=generator.proportions,
            cat_emb_init_sigma=generator.cat_emb_init_sigma,
            timewarp_type=generator.timewarp_type,
            timewarp_weight_low_noise=generator.timewarp_weight_low_noise,
        ).to(generator.device)
        state_dict = torch.load(path / "diff_model.pt", map_location=generator.device)
        generator.diff_model.load_state_dict(state_dict)
        generator.diff_model.eval()
        return generator
