# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..base import BaseGenerator
from ..dgm_utils import FastTensorDataLoader
from ..persistence import load_generator_state, restore_generator, save_generator_state
from ...utils.utils import get_total_trainable_params
from .encoder import Discretizer
from .highres import HighResFlowModel
from .lowres import CatCDTD, LowResMLP, cycle, set_seeds


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
        random_state (int): Random seed for reproducibility. Default: 0.

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
        ...     random_state=42,
        ... )
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabcascade"

    def __init__(
        self,
        epochs: int = 100,
        training_steps: int = 30_000,
        batch_size: int = 4096,
        embedding_dim: int = 256,
        num_timesteps: int = 200,
        encoder: str = "dt",
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
    ):
        self.__dict__.update(locals())
        self.betas = tuple(betas)
        del self.__dict__["self"]
        self.seed = self.random_state
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode_into_z(self, x_num):
        self.z_encoder = Discretizer(
            x_num,
            variant=self.config.data.encoder,
            seed=self.seed,
            k_max=self.config.data.k_max,
            max_depth=self.config.data.max_depth,
        )
        groups, mask, infl_groups, has_miss = self.z_encoder.encode(x_num)

        if self.config.data.encoder == "gmm":
            for i in range(groups.shape[1]):
                vals = groups[:, i].unique()
                self.z_encoder.means[i] = self.z_encoder.means[i][vals]
            self.gmm_ord_enc = OrdinalEncoder()
            groups = self.gmm_ord_enc.fit_transform(groups.numpy())
            groups = torch.from_numpy(groups).long()

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
        for i in range(x_cat.shape[1]):
            val, counts = x_cat[:, i].unique(return_counts=True)
            n_classes_cat.append(len(val))
            proportions_cat.append(counts / n_sample)

        n_classes_num = []
        proportions_num = []
        for i in range(groups.shape[1]):
            val, counts = groups[:, i].unique(return_counts=True)
            n_classes_num.append(len(val))
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

    def train(self, x_cat, x_num):
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

        while step < self.config.lowres.training.num_steps_train:
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
                lowres_loss_trn = highres_loss_trn = n_inputs = 0
                scheduler_highres.step(highres_loss_trn)
            step += 1
            pbar.update(1)
        pbar.close()

        ema_lowres.copy_to()
        self.lowres.eval()
        ema_highres.copy_to()
        self.highres.eval()

    def sample(self, num_samples):
        x_low_gen = self.lowres.sample_data(
            num_samples,
            num_steps=self.config.lowres.model.generation_steps,
            batch_size=self.config.lowres.model.generation_batch_size,
            seed=self.seed,
            verbose=False,
        )
        x_cat_gen = x_low_gen[:, : self.n_cat_cols]
        z_num_gen = x_low_gen[:, self.n_cat_cols :]
        x_num_gen = self.highres.sample_data(
            x_cat_gen,
            z_num_gen,
            num_steps=self.config.highres.model.generation_steps,
            batch_size=self.config.highres.model.generation_batch_size,
            seed=self.seed,
            verbose=False,
        )

        if self.config.data.encoder == "gmm":
            z_num_gen_enc = self.gmm_ord_enc.inverse_transform(z_num_gen)
            z_num_gen_enc = torch.from_numpy(z_num_gen_enc).long()
        else:
            z_num_gen_enc = z_num_gen
        infl_mask, miss_mask = self.get_masks(z_num_gen_enc)
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
        set_seeds(self.random_state, cuda_deterministic=True)
        self.col_order = X.columns
        self.discrete_features = list(discrete_features)
        self.numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]
        if not self.numerical_features:
            raise ValueError("TabCascade requires at least one numerical feature.")

        if self.discrete_features:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            x_cat = torch.tensor(
                self.ordinal_encoder.fit_transform(X[self.discrete_features])
            ).long()
        else:
            self.ordinal_encoder = None
            x_cat = torch.empty((len(X), 0), dtype=torch.long)

        if self.numerical_features:
            self.quantile_encoder = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(min(len(X) // 30, 1000), 10),
                subsample=int(1e9),
                random_state=self.random_state,
            )
            self.st_scaler = StandardScaler()

        x_num = X[self.numerical_features].to_numpy().astype(float)
        x_num = self.quantile_encoder.fit_transform(x_num)
        x_num = self.st_scaler.fit_transform(x_num)
        x_num = torch.tensor(x_num).float()

        steps_per_epoch = max(len(X) // min(self.batch_size, len(X)), 1)
        num_steps_train = self.training_steps or self.epochs * steps_per_epoch
        self.config = self._make_config(num_steps_train)
        self.seed = self.random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train(x_cat, x_num)
        return self

    def _generate(self, n: int):
        x_cat, x_num = self.sample(n)
        x_num = self.st_scaler.inverse_transform(x_num)
        x_num = self.quantile_encoder.inverse_transform(x_num)
        frames = []
        if self.discrete_features:
            frames.append(
                pd.DataFrame(
                    self.ordinal_encoder.inverse_transform(x_cat.astype(int)),
                    columns=self.discrete_features,
                )
            )
        frames.append(pd.DataFrame(x_num, columns=self.numerical_features))
        return pd.concat(frames, axis=1)[self.col_order]

    def _make_config(self, num_steps_train):
        return config(
            data={
                "encoder": self.encoder,
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

    def save(self, path):
        path = Path(path)
        state = self.__dict__.copy()
        for key in [
            "config",
            "device",
            "seed",
            "z_encoder",
            "train_loader",
            "n_cat_cols",
            "n_classes",
            "proportions",
            "z_means",
            "z_stds",
            "z_infl_groups",
            "z_has_miss",
            "lowres",
            "highres",
            "gmm_ord_enc",
        ]:
            state.pop(key, None)
        save_generator_state(path, state)
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
                "gmm_ord_enc": getattr(self, "gmm_ord_enc", None),
            },
            path / "tabcascade.pt",
        )
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        generator = restore_generator(cls, load_generator_state(path))
        generator.seed = generator.random_state
        generator.config = generator._make_config(generator.training_steps)
        generator.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(
            path / "tabcascade.pt", map_location=generator.device, weights_only=False
        )
        generator.n_cat_cols = state["n_cat_cols"]
        generator.n_classes = state["n_classes"]
        generator.proportions = state["proportions"]
        generator.z_means = state["z_means"]
        generator.z_stds = state["z_stds"]
        generator.z_infl_groups = state["z_infl_groups"]
        generator.z_has_miss = state["z_has_miss"]
        if state["gmm_ord_enc"] is not None:
            generator.gmm_ord_enc = state["gmm_ord_enc"]
        generator.lowres = generator.get_lowres_model().to(generator.device)
        generator.highres = generator.get_highres_model().to(generator.device)
        generator.lowres.load_state_dict(state["lowres"])
        generator.highres.load_state_dict(state["highres"])
        generator.lowres.eval()
        generator.highres.eval()
        return generator
