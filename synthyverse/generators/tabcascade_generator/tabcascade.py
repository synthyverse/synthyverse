# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from typing import Optional
import time

from ..base import BaseGenerator
from ..dgm_utils import (
    FastTensorDataLoader,
    QuantileStandardScaler,
    clone_state_dict,
    validate_c2st,
)
from ...utils.utils import get_total_trainable_params
from .encoder import Discretizer
from .highres import HighResFlowModel
from .lowres import CatCDTD, LowResMLP, cycle


class Config(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def config(**kwargs):
    return Config(
        {
            key: config(**value) if isinstance(value, dict) else value
            for key, value in kwargs.items()
        }
    )


class TabCascadeGenerator(BaseGenerator):
    """TabCascade generator for mixed-type tabular data.

    TabCascade is a cascaded diffusion model, using categorical diffusion to model low-resolution information
    and flow matching to model high-resolution information. Numerical features are discretized by distributional
    trees as a low-resolution encoding.

    Based on the implementation from the original paper's authors: https://github.com/muellermarkus/tabcascade.

    Paper: "Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features" by Mueller et al. (2026).

    Args:
        epochs (int): Number of training epochs used when ``training_steps`` is
            None. Default: 100.
        training_steps (int): Number of training steps. When provided, this
            overrides ``epochs``. Default: 30000.
        batch_size (int): Batch size for training and generation. Default: 4096.
        embedding_dim (int): Embedding dimension for the low- and high-resolution
            MLPs. Default: 256.
        num_timesteps (int): Number of sampling timesteps in both cascade
            stages. Default: 200.
        encoder (str): Numerical discretizer. Options: "dt" and "gmm".
            Default: "dt".
        max_depth (int): Maximum depth for the distributional tree encoder.
            Default: 8.
        k_max (int): Maximum number of mixture components for the GMM encoder.
            Default: 10.
        lowres_cat_emb_dim (int): Low-resolution categorical embedding
            dimension. Default: 16.
        highres_cat_emb_dim (int): High-resolution categorical embedding
            dimension. Default: 8.
        lowres_mlp_n_layers (int): Number of low-resolution MLP layers.
            Default: 5.
        lowres_mlp_n_units (int): Number of low-resolution MLP units.
            Default: 664.
        highres_mlp_n_layers (int): Number of high-resolution MLP layers.
            Default: 5.
        highres_mlp_n_units (int): Number of high-resolution MLP units.
            Default: 394.
        gamma_input_dim (int): Input dimension for the high-resolution
            conditional noise schedule. Default: 16.
        lowres_sigma_min (float): Minimum noise level for the low-resolution
            model. Default: 0.
        lowres_sigma_max (float): Maximum noise level for the low-resolution
            model. Default: 100.
        lowres_sigma_data (float): Data noise scale used by the
            low-resolution model. Default: 1.0.
        lowres_timewarp_weight_low_noise (float): Weight assigned to
            low-noise samples by the low-resolution timewarp. Default: 3.0.
        lowres_timewarp_variant (str): Low-resolution timewarp type.
            Default: "logistic".
        lowres_cat_emb_init_sigma (float): Standard deviation used to
            initialize low-resolution categorical embeddings. Default: 0.001.
        lowres_normalize_by_entropy (bool): Whether to normalize
            low-resolution categorical losses by feature entropy. Default:
            True.
        lowres_mlp_act (str): Low-resolution MLP activation function.
            Default: "relu".
        lr (float): Low-resolution model learning rate. Default: 1e-3.
        highres_lr (float): High-resolution model learning rate. Default: 2e-3.
        ema_decay (float): Exponential moving average decay. Default: 0.999.
        weight_decay (float): Weight decay for AdamW. Default: 0.
        betas (tuple): AdamW beta coefficients for both optimizers. Default:
            (0.9, 0.999).
        num_steps_warmup (int): Low-resolution warmup steps. Default: 1000.
        highres_num_steps_warmup (int): High-resolution warmup steps. Default:
            -1.
        clip_grad (bool): Whether to clip gradients for both cascade stages.
            Default: False.
        log_steps (int): Steps between progress logging. Default: 100.
        cap_train_time (float): Time limit in seconds for training. Default: None.
        val_steps (int): Epochs between training-set C2ST validation, or training steps when ``training_steps`` is provided. Set to <=0 to disable validation. Default: 5000.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabCascadeGenerator
        >>>
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> generator = TabCascadeGenerator(
        ...     training_steps=30000,
        ...     batch_size=4096,
        ... )
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabcascade"
    supports_purely_categorical = False

    def __init__(
        self,
        epochs: int = 100,
        training_steps: int = 30_000,
        batch_size: int = 4096,
        embedding_dim: int = 256,
        num_timesteps: int = 200,
        encoder: str = "dt",
        adjust_means: bool = False,
        max_depth: int = 8,
        k_max: int = 10,
        lowres_cat_emb_dim: int = 16,
        highres_cat_emb_dim: int = 8,
        lowres_mlp_n_layers: int = 5,
        lowres_mlp_n_units: int = 664,
        highres_mlp_n_layers: int = 5,
        highres_mlp_n_units: int = 394,
        gamma_input_dim: int = 16,
        lowres_sigma_min: float = 0,
        lowres_sigma_max: float = 100,
        lowres_sigma_data: float = 1.0,
        lowres_timewarp_weight_low_noise: float = 3.0,
        lowres_timewarp_variant: str = "logistic",
        lowres_cat_emb_init_sigma: float = 0.001,
        lowres_normalize_by_entropy: bool = True,
        lowres_mlp_act: str = "relu",
        lr: float = 1e-3,
        highres_lr: float = 2e-3,
        ema_decay: float = 0.999,
        weight_decay: float = 0,
        betas: tuple = (0.9, 0.999),
        num_steps_warmup: int = 1000,
        highres_num_steps_warmup: int = -1,
        clip_grad: bool = False,
        log_steps: int = 100,
        random_state: int = 0,
        full_determinism: bool = False,
        cap_train_time: Optional[float] = None,
        val_steps: int = 5000,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.__dict__.update(locals())
        self.betas = tuple(betas)
        del self.__dict__["self"]
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cap_train_time = cap_train_time

    def encode_into_z(self, x_num):
        self.z_encoder = Discretizer(
            x_num,
            variant=self.config.data.encoder,
            seed=self.random_state,
            k_max=self.config.data.k_max,
            adjust_means=self.config.data.adjust_means,
            max_depth=self.config.data.max_depth,
        )
        groups, mask, infl_groups, has_miss = self.z_encoder.encode(x_num)
        self.z_means = self.z_encoder.means
        self.z_stds = self.z_encoder.stds
        return groups, mask, infl_groups, has_miss

    def get_masks(self, groups):
        infl_mask = []
        for i in range(groups.shape[1]):
            z_infl_groups = torch.tensor(self.z_infl_groups[i])
            infl_groups = z_infl_groups + 1 if self.z_has_miss[i] else z_infl_groups
            infl_mask.append(torch.isin(groups[:, i], infl_groups))
        infl_mask = torch.column_stack(infl_mask)

        miss_mask = []
        for i in range(groups.shape[1]):
            if self.z_has_miss[i]:
                miss_mask.append(groups[:, i] == 0)
            else:
                miss_mask.append(torch.zeros_like(groups[:, i]).bool())
        miss_mask = torch.column_stack(miss_mask) if self.z_has_miss.any() else None
        return infl_mask, miss_mask

    def get_classes_and_proportions(self, x_cat, groups):
        n_classes_cat = []
        proportions_cat = []
        n_sample = x_cat.shape[0]
        cat_cardinalities = self._categorical_cardinalities(self.discrete_features)
        for i in range(x_cat.shape[1]):
            n_classes_cat.append(cat_cardinalities[i])
            counts = torch.bincount(x_cat[:, i], minlength=cat_cardinalities[i])
            proportions_cat.append(counts / n_sample)

        n_classes_num = []
        proportions_num = []
        for i in range(groups.shape[1]):
            n_classes = int(groups[:, i].max()) + 1
            counts = torch.bincount(groups[:, i], minlength=n_classes)
            n_classes_num.append(n_classes)
            proportions_num.append(counts / n_sample)

        return n_classes_cat + n_classes_num, proportions_cat + proportions_num

    def get_train_loader(self, x_cat, x_num, z_groups, z_mask):
        x_means = torch.nanmean(x_num, dim=0)
        for i in range(x_num.shape[1]):
            x_num[:, i] = torch.nan_to_num(x_num[:, i], nan=x_means[i])

        return FastTensorDataLoader(
            x_cat,
            x_num,
            z_groups,
            z_mask,
            batch_size=min(self.config.data.batch_size, x_num.shape[0]),
            shuffle=True,
            drop_last=True,
        )

    def get_lowres_model(self):
        cfg = self.config.lowres.model
        predictor = LowResMLP(
            self.n_classes,
            cfg.cat_emb_dim,
            cfg.mlp_emb_dim,
            cfg.mlp_n_layers,
            cfg.mlp_n_units,
            self.proportions,
            cfg.mlp_act,
        )
        return CatCDTD(
            predictor,
            self.n_classes,
            self.proportions,
            cfg.cat_emb_dim,
            cfg.sigma_min,
            cfg.sigma_max,
            cfg.sigma_data,
            cfg.normalize_by_entropy,
            cfg.timewarp_weight_low_noise,
            cfg.timewarp_variant,
            cfg.cat_emb_init_sigma,
        )

    def get_highres_model(self):
        cfg = self.config.highres.model
        return HighResFlowModel(
            self.z_means,
            self.z_stds,
            self.n_classes,
            cfg.mlp_emb_dim,
            cfg.mlp_n_layers,
            cfg.mlp_n_units,
            cfg.gamma_input_dim,
            cfg.cat_emb_dim,
        )

    def _train_tabcascade(self, x_cat, x_num, X_train):
        self.n_cat_cols = x_cat.shape[1]
        z_groups, z_mask, self.z_infl_groups, self.z_has_miss = self.encode_into_z(
            x_num
        )
        self.n_classes, self.proportions = self.get_classes_and_proportions(
            x_cat, z_groups
        )
        self.train_loader = self.get_train_loader(x_cat, x_num, z_groups, z_mask)
        self.lowres = self.get_lowres_model().to(self.device)
        self.highres = self.get_highres_model().to(self.device)

        print(
            f"Total trainable parameters: "
            f"{get_total_trainable_params(self.lowres) + get_total_trainable_params(self.highres):,}"
        )

        ema_lowres = ExponentialMovingAverage(
            self.lowres.parameters(),
            decay=self.config.lowres.training.ema_decay,
        )
        ema_highres = ExponentialMovingAverage(
            self.highres.parameters(),
            decay=self.config.highres.training.ema_decay,
        )
        opt_lowres = torch.optim.AdamW(
            self.lowres.parameters(),
            lr=self.config.lowres.training.lr,
            weight_decay=self.config.lowres.training.weight_decay,
            betas=self.config.lowres.training.betas,
        )
        opt_highres = torch.optim.AdamW(
            self.highres.parameters(),
            lr=self.config.highres.training.lr,
            weight_decay=self.config.highres.training.weight_decay,
            betas=self.config.highres.training.betas,
        )
        scheduler_highres = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_highres,
            mode="min",
            factor=0.9,
            patience=3,
            min_lr=1e-6,
        )

        train_loader = cycle(self.train_loader)
        step = n_inputs = 0
        lowres_loss_trn = highres_loss_trn = 0
        pbar = tqdm(total=self.config.lowres.training.num_steps_train)

        train_time = 0.0
        best_val_score = float("inf")
        best_val_model = None
        stop_training = False
        while step < self.config.lowres.training.num_steps_train:
            step_start_time = time.monotonic()
            if step < self.config.lowres.training.num_steps_warmup:
                lr = (
                    self.config.lowres.training.lr
                    * (step + 1)
                    / self.config.lowres.training.num_steps_warmup
                )
                for param_group in opt_lowres.param_groups:
                    param_group["lr"] = lr

            if (self.config.lowres.model.variant == "cdtd") and (
                step > self.config.lowres.training.num_steps_warmup
            ):
                aux_step = step - self.config.lowres.training.num_steps_warmup
                rate = 1 - (
                    aux_step
                    / (
                        self.config.lowres.training.num_steps_train
                        - self.config.lowres.training.num_steps_warmup
                    )
                )
                lr = self.config.lowres.training.lr * rate + 1e-6 * (1 - rate)
                for param_group in opt_lowres.param_groups:
                    param_group["lr"] = lr

            if step < self.config.highres.training.num_steps_warmup:
                lr = (
                    self.config.highres.training.lr
                    * (step + 1)
                    / self.config.highres.training.num_steps_warmup
                )
                for param_group in opt_highres.param_groups:
                    param_group["lr"] = lr

            if self.config.highres.model.get("variant", "flow") == "cdtd" and (
                step > self.config.highres.training.num_steps_warmup
            ):
                aux_step = step - self.config.highres.training.num_steps_warmup
                rate = 1 - (
                    aux_step
                    / (
                        self.config.lowres.training.num_steps_train
                        - self.config.highres.training.num_steps_warmup
                    )
                )
                lr = self.config.highres.training.lr * rate + 1e-6 * (1 - rate)
                for param_group in opt_highres.param_groups:
                    param_group["lr"] = lr

            opt_lowres.zero_grad(set_to_none=True)
            opt_highres.zero_grad(set_to_none=True)

            batch = next(train_loader)
            x_cat, x_num, z_num, mask = (x.to(self.device) for x in batch)
            batch_size = len(x_cat)
            n_inputs += batch_size

            lowres_input = torch.column_stack((x_cat, z_num))
            train_loss_lowres = self.lowres.loss_fn(lowres_input)["train_loss"]
            train_loss_lowres.backward()
            if self.config.lowres.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.lowres.parameters(), max_norm=1.0)
            opt_lowres.step()
            ema_lowres.update()
            lowres_loss_trn += train_loss_lowres.detach().item() * batch_size

            train_loss_highres = self.highres.loss_fn(x_num, x_cat, z_num, mask)
            train_loss_highres.backward()
            if self.config.highres.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.highres.parameters(), max_norm=1.0)
            opt_highres.step()
            ema_highres.update()
            highres_loss_trn += train_loss_highres.detach().item() * batch_size

            if step % self.config.lowres.training.log_steps == 0:
                lowres_loss_trn = lowres_loss_trn / n_inputs
                highres_loss_trn = highres_loss_trn / n_inputs
                pbar.set_postfix(
                    {
                        "loss (lowres)": f"{lowres_loss_trn:.4f}",
                        "loss (highres)": f"{highres_loss_trn:.4f}",
                    },
                )

                scheduler_highres.step(highres_loss_trn)
                lowres_loss_trn = highres_loss_trn = n_inputs = 0

            step += 1
            pbar.update(1)

            train_time += time.monotonic() - step_start_time
            if (
                self.cap_train_time is not None
                and train_time > self.cap_train_time
            ):
                print(f"Training timed out after {self.cap_train_time} seconds.")
                break

            if (
                self._val_steps_train > 0
                and step > 0
                and step % self._val_steps_train == 0
            ):
                self.lowres.eval()
                self.highres.eval()
                ema_lowres.store()
                ema_highres.store()
                ema_lowres.copy_to()
                ema_highres.copy_to()
                score = validate_c2st(self, X_train, random_state=self.random_state)
                if score < best_val_score:
                    best_val_score = score
                    best_val_model = {
                        "lowres": clone_state_dict(self.lowres),
                        "highres": clone_state_dict(self.highres),
                    }
                else:
                    stop_training = True
                ema_lowres.restore()
                ema_highres.restore()
                self.lowres.train()
                self.highres.train()
                if stop_training:
                    break
        pbar.close()

        if best_val_model is None:
            ema_lowres.copy_to()
            ema_highres.copy_to()
        else:
            self.lowres.load_state_dict(
                {k: v.to(self.device) for k, v in best_val_model["lowres"].items()}
            )
            self.highres.load_state_dict(
                {k: v.to(self.device) for k, v in best_val_model["highres"].items()}
            )
        self.lowres.eval()
        self.highres.eval()

    def _sample_tabcascade(self, num_samples):
        x_low_gen = self.lowres.sample_data(
            num_samples,
            num_steps=self.config.lowres.model.generation_steps,
            batch_size=self.config.lowres.model.generation_batch_size,
            verbose=False,
        )
        x_cat_gen = x_low_gen[:, : self.n_cat_cols]
        z_num_gen = x_low_gen[:, self.n_cat_cols :]
        x_num_gen = self.highres.sample_data(
            x_cat_gen,
            z_num_gen,
            num_steps=self.config.highres.model.generation_steps,
            batch_size=self.config.highres.model.generation_batch_size,
            verbose=False,
        )
        infl_mask, miss_mask = self.get_masks(z_num_gen)
        z_num_gen_means = (
            self.highres.get_group_means(
                z_num_gen.to(self.device) + self.highres.group_offset
            )
            .squeeze(-1)
            .cpu()
        )
        x_num_gen = torch.where(infl_mask, z_num_gen_means, x_num_gen)
        if miss_mask is not None:
            x_num_gen = torch.masked_fill(x_num_gen, miss_mask, torch.nan)
        return x_cat_gen.numpy(), x_num_gen.numpy()

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        X = X.copy()
        self.col_order = X.columns
        self.discrete_features = list(discrete_features)
        self.numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]
        if not self.numerical_features:
            raise ValueError("TabCascade requires at least one numerical feature.")

        if self.discrete_features:
            x_cat = torch.tensor(X[self.discrete_features].to_numpy()).long()
        else:
            x_cat = torch.empty((len(X), 0), dtype=torch.long)

        if self.numerical_features:
            self.quantile_encoder = QuantileStandardScaler(
                len(X), self.random_state
            )

        x_num = X[self.numerical_features].to_numpy().astype(float)
        x_num = self.quantile_encoder.fit_transform(x_num)
        x_num = torch.tensor(x_num).float()

        steps_per_epoch = max(len(X) // min(self.batch_size, len(X)), 1)
        num_steps_train = self.training_steps or self.epochs * steps_per_epoch
        self._val_steps_train = 0
        if self.val_steps > 0:
            self._val_steps_train = (
                self.val_steps
                if self.training_steps is not None
                else self.val_steps * steps_per_epoch
            )
        self.config = self._make_config(num_steps_train)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._train_tabcascade(x_cat, x_num, X)
        return self

    def _generate(self, n: int):
        x_cat, x_num = self._sample_tabcascade(n)
        x_num = self.quantile_encoder.inverse_transform(x_num)
        frames = []
        if self.discrete_features:
            frames.append(
                pd.DataFrame(
                    x_cat.astype(int),
                    columns=self.discrete_features,
                )
            )
        frames.append(pd.DataFrame(x_num, columns=self.numerical_features))
        return pd.concat(frames, axis=1)[self.col_order]

    def _make_config(self, num_steps_train):
        return config(
            data={
                "encoder": self.encoder,
                "adjust_means": self.adjust_means,
                "max_depth": self.max_depth,
                "k_max": self.k_max,
                "batch_size": self.batch_size,
            },
            highres={
                "model": {
                    "mlp_n_layers": self.highres_mlp_n_layers,
                    "mlp_n_units": self.highres_mlp_n_units,
                    "mlp_emb_dim": self.embedding_dim,
                    "cat_emb_dim": self.highres_cat_emb_dim,
                    "gamma_input_dim": self.gamma_input_dim,
                    "generation_steps": self.num_timesteps,
                    "generation_batch_size": self.batch_size,
                },
                "training": {
                    "num_steps_warmup": self.highres_num_steps_warmup,
                    "ema_decay": self.ema_decay,
                    "lr": self.highres_lr,
                    "weight_decay": self.weight_decay,
                    "betas": self.betas,
                    "clip_grad": self.clip_grad,
                },
            },
            lowres={
                "model": {
                    "variant": "cdtd",
                    "mlp_act": self.lowres_mlp_act,
                    "mlp_n_layers": self.lowres_mlp_n_layers,
                    "mlp_n_units": self.lowres_mlp_n_units,
                    "mlp_emb_dim": self.embedding_dim,
                    "cat_emb_dim": self.lowres_cat_emb_dim,
                    "cat_emb_init_sigma": self.lowres_cat_emb_init_sigma,
                    "normalize_by_entropy": self.lowres_normalize_by_entropy,
                    "timewarp_variant": self.lowres_timewarp_variant,
                    "timewarp_weight_low_noise": self.lowres_timewarp_weight_low_noise,
                    "sigma_min": self.lowres_sigma_min,
                    "sigma_max": self.lowres_sigma_max,
                    "sigma_data": self.lowres_sigma_data,
                    "generation_steps": self.num_timesteps,
                    "generation_batch_size": self.batch_size,
                },
                "training": {
                    "num_steps_train": num_steps_train,
                    "log_steps": self.log_steps,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "betas": self.betas,
                    "ema_decay": self.ema_decay,
                    "clip_grad": self.clip_grad,
                    "scheduler": True,
                    "num_steps_warmup": self.num_steps_warmup,
                },
            },
        )

    def _state(self):
        return {
            "config": config(
                data={"encoder": self.config.data.encoder},
                lowres={"model": self.config.lowres.model},
                highres={"model": self.config.highres.model},
            ),
            "col_order": self.col_order,
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
            "ordinal_encoder": self.ordinal_encoder,
            "quantile_encoder": self.quantile_encoder,
        }

    def _save_extra(self, path: Path) -> None:
        torch.save(
            {
                "n_cat_cols": self.n_cat_cols,
                "n_classes": self.n_classes,
                "proportions": self.proportions,
                "z_means": self.z_means,
                "z_stds": self.z_stds,
                "z_infl_groups": self.z_infl_groups,
                "z_has_miss": self.z_has_miss,
                "lowres": self.lowres.state_dict(),
                "highres": self.highres.state_dict(),
            },
            path / "tabcascade.pt",
        )

    def _load_extra(self, path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(
            path / "tabcascade.pt", map_location=self.device, weights_only=False
        )
        self.n_cat_cols = state["n_cat_cols"]
        self.n_classes = state["n_classes"]
        self.proportions = state["proportions"]
        self.z_means = state["z_means"]
        self.z_stds = state["z_stds"]
        self.z_infl_groups = state["z_infl_groups"]
        self.z_has_miss = state["z_has_miss"]
        self.lowres = self.get_lowres_model().to(self.device)
        self.highres = self.get_highres_model().to(self.device)
        self.lowres.load_state_dict(state["lowres"])
        self.highres.load_state_dict(state["highres"])
        self.lowres.eval()
        self.highres.eval()
