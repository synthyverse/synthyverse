# Third-party notice: based on Apache-2.0-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution, NOTICE, and modification details.
from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .vae import Model_VAE, Encoder_model, Decoder_model
from .diffusion import Model, sample
from ..base import BaseGenerator
from ..dgm_utils import FastTensorDataLoader, MLPDiffusion
from ..persistence import load_generator_state, restore_generator, save_generator_state
from ...utils.utils import resolve_epochs_from_training_steps


CE_LOSS_FN = nn.CrossEntropyLoss()


class TabSynGenerator(BaseGenerator):
    """TabSyn: a latent diffusion model for tabular data.

    Trains a VAE to learn a latent representation of tabular data, then fits a diffusion model in that latent space.

    Based on the paper's original implementation: https://github.com/amazon-science/tabsyn/.

    Paper: "Mixed-type tabular data synthesis with score-based diffusion in latent space" by Zhang et al. (2023).

    Args:
        target_column (str): Name of the target column used for stratified validation splitting.
        val_size (float): Fraction of training rows reserved for validation.
            Default: 0.1.
        batch_size (int): Batch size applied to both VAE and diffusion training. Default: 4096.
        epochs (int): Maximum number of diffusion training epochs. Default: 10001.
        training_steps (int, optional): Total diffusion training steps. When
            provided, this overrides ``epochs`` by deriving the epoch count from
            the latent training sample size and batch size.
        lr (float): Learning rate for diffusion training. Default: 1e-3.
        embedding_dim (int): Time embedding dimension used by the diffusion denoiser. Default: 1024.
        mlp_dim (int): Hidden width of the diffusion denoiser MLP. Default: 2048.
        mlp_layers (int): Number of hidden diffusion denoiser MLP layers with width
            ``mlp_dim``. Default: 2.
        num_timesteps (int): Number of reverse diffusion steps used for generation. Default: 50.
        vae_lr (float): Learning rate for VAE training. Default: 1e-3.
        vae_wd (float): Weight decay used by the VAE optimizer. Default: 0.
        vae_d_token (int): Token embedding dimension used by the VAE. Default: 4.
        vae_n_head (int): Number of attention heads in the VAE transformer blocks. Default: 1.
        vae_factor (int): Expansion factor used in VAE feed-forward layers. Default: 32.
        vae_num_layers (int): Number of VAE encoder/decoder layers. Default: 2.
        vae_num_epochs (int): Maximum number of VAE epochs. Default: 4000.
        vae_training_steps (int, optional): Total VAE training steps. When
            provided, this overrides ``vae_num_epochs`` by deriving the epoch
            count from the training sample size and batch size.
        vae_min_beta (float): Minimum KL coefficient for VAE KL annealing. Default: 1e-5.
        vae_max_beta (float): Initial/maximum KL coefficient for VAE KL annealing. Default: 1e-2.
        vae_lambda (float): Multiplicative decay factor applied to beta when validation plateaus.
            Default: 0.7.
        diffusion_wd (float): Weight decay used by the diffusion optimizer. Default: 0.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import TabSynGenerator
        >>>
        >>> # Load data and define categorical columns
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator (requires target column)
        >>> generator = TabSynGenerator(
        ...     target_column="target",
        ...     vae_num_epochs=100,
        ...     epochs=500,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabsyn"
    needs_validation_set = True

    def __init__(
        self,
        target_column: str,
        val_size: float = 0.1,
        vae_lr: float = 1e-3,
        vae_wd: float = 0,
        vae_d_token: int = 4,
        vae_n_head: int = 1,
        vae_factor: int = 32,
        vae_num_layers: int = 2,
        vae_num_epochs: int = 4000,
        vae_training_steps: int = None,
        vae_min_beta: float = 1e-5,
        vae_max_beta: float = 1e-2,
        vae_lambda: float = 0.7,
        diffusion_wd: float = 0,
        random_state: int = 0,
        batch_size: int = 4096,
        epochs: int = 10_000 + 1,
        training_steps: int = None,
        lr: float = 1e-3,
        embedding_dim: int = 1024,
        mlp_dim: int = 2048,
        mlp_layers: int = 2,
        num_timesteps: int = 50,
    ):
        self.random_state = random_state
        self.target_column = target_column
        self.val_size = val_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_steps = training_steps
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.num_timesteps = num_timesteps
        self.vae_lr = vae_lr
        self.vae_wd = vae_wd
        self.vae_d_token = vae_d_token
        self.vae_n_head = vae_n_head
        self.vae_factor = vae_factor
        self.vae_num_layers = vae_num_layers
        self.vae_num_epochs = vae_num_epochs
        self.vae_training_steps = vae_training_steps
        self.vae_min_beta = vae_min_beta
        self.vae_max_beta = vae_max_beta
        self.vae_lambda = vae_lambda
        self.diffusion_wd = diffusion_wd

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        x = X.copy()
        self.ori_columns = X.columns
        self.discrete_features = discrete_features
        if not 0 < self.val_size < 1:
            raise ValueError("TabSyn requires val_size to be between 0 and 1.")

        stratify = (
            x[self.target_column] if self.target_column in discrete_features else None
        )
        x, x_val = train_test_split(
            x,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        x[self.discrete_features] = self.ordinal_encoder.fit_transform(
            x[self.discrete_features]
        )
        x_val[self.discrete_features] = self.ordinal_encoder.transform(
            x_val[self.discrete_features]
        )

        self.scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(len(x) // 30, 1000), 10),
            subsample=int(1e9),
            random_state=self.random_state,
        )
        self.numerical_features = [
            col for col in x.columns if col not in self.discrete_features
        ]
        self.scaler.fit(x[self.numerical_features])
        x[self.numerical_features] = self.scaler.transform(x[self.numerical_features])
        x_val[self.numerical_features] = self.scaler.transform(
            x_val[self.numerical_features]
        )

        # filter out validation obs with cats not seen in training data
        drop_mask = []
        for col_idx in range(x[discrete_features].shape[1]):
            drop_mask.append(
                x_val[discrete_features].values[:, col_idx]
                > x[discrete_features].values[:, col_idx].max()
            )
        drop_mask = np.column_stack(drop_mask)
        drop_mask = np.sum(drop_mask, axis=1) > 0
        x_val = x_val[~drop_mask]
        print(
            f"Number of observations dropped in validation set due to unseen categories: {sum(drop_mask)}"
        )

        # train the VAE on train data and use validation only for model selection
        vae = self._train_vae(x, x_val)
        train_z = self._encode_vae(vae, pd.concat([x, x_val], ignore_index=True))

        # no longer need original data
        del x, x_val, vae

        # train the diffusion model
        self.model = self._train_diffusion(train_z)

        return self

    def _generate(self, n: int):
        self.model.eval()
        with torch.no_grad():
            x = sample(
                self.model.denoise_fn_D,
                n,
                self.sample_dim,
                self.device,
                num_steps=self.num_timesteps,
            )
            x = x * 2 + self.train_z_mean.to(self.device)

            # x = x.float().cpu().numpy()

            x = x.reshape(x.shape[0], -1, self.token_dim)
            norm_input = self.pre_decoder(x.float())
            x_hat_num, x_hat_cat = norm_input

            x_cat = []
            for pred in x_hat_cat:
                x_cat.append(pred.argmax(dim=-1))

        syn_num = x_hat_num.detach().cpu().numpy()
        syn_cat = torch.stack(x_cat).t().cpu().numpy()
        syn_cat = np.int64(syn_cat)

        syn = np.concatenate([syn_num, syn_cat], axis=1)
        cols = self.numerical_features + self.discrete_features
        syn = pd.DataFrame(syn, columns=cols)
        syn = syn[self.ori_columns]
        syn[self.numerical_features] = self.scaler.inverse_transform(
            syn[self.numerical_features]
        )
        syn[self.discrete_features] = self.ordinal_encoder.inverse_transform(
            syn[self.discrete_features]
        )

        return syn

    def _train_vae(self, x, x_val):
        d_numerical = len(x.columns.tolist()) - len(self.discrete_features)
        categories = []
        for col in self.discrete_features:
            categories.append(x[col].nunique())
        self.d_numerical = d_numerical
        self.categories = categories

        vae = Model_VAE(
            self.vae_num_layers,
            d_numerical,
            categories,
            self.vae_d_token,
            n_head=self.vae_n_head,
            factor=self.vae_factor,
            bias=True,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=self.vae_lr,
            weight_decay=self.vae_wd,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=10)

        current_lr = optimizer.param_groups[0]["lr"]
        patience = 0
        batch_size = min(self.batch_size, len(x))

        beta = self.vae_max_beta

        x_tr_num = x[
            [col for col in x.columns if col not in self.discrete_features]
        ].values
        x_tr_num = torch.from_numpy(x_tr_num).float()
        x_tr_cat = x[self.discrete_features].values
        x_tr_cat = torch.from_numpy(x_tr_cat).long()

        train_loader = FastTensorDataLoader(
            x_tr_num,
            x_tr_cat,
            batch_size=batch_size,
            shuffle=True,
        )

        x_val_num = x_val[
            [col for col in x.columns if col not in self.discrete_features]
        ].values
        x_val_num = torch.from_numpy(x_val_num).float()
        x_val_cat = x_val[self.discrete_features].values
        x_val_cat = torch.from_numpy(x_val_cat).long()

        x_val_num = x_val_num.to(self.device)
        x_val_cat = x_val_cat.to(self.device)

        best_val_loss = float("inf")

        epochs = resolve_epochs_from_training_steps(
            self.vae_num_epochs,
            self.vae_training_steps,
            len(x),
            batch_size,
        )
        pbar = tqdm(range(epochs))

        for _ in pbar:
            vae.train()
            for batch_num, batch_cat in train_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)

                optimizer.zero_grad()

                Recon_X_num, Recon_X_cat, mu_z, std_z = vae(batch_num, batch_cat)

                loss_mse, loss_ce, loss_kld, train_acc = self._compute_loss(
                    batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )

                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                optimizer.step()

            vae.eval()
            with torch.no_grad():
                Recon_X_num, Recon_X_cat, mu_z, std_z = vae(x_val_num, x_val_cat)

                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = self._compute_loss(
                    x_val_num, x_val_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]["lr"]

                if new_lr != current_lr:
                    current_lr = new_lr

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_vae = deepcopy(vae.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience == 10:
                        if beta > self.vae_min_beta:
                            beta = beta * self.vae_lambda

        vae.load_state_dict(best_vae)
        vae.eval()
        return vae

    def _encode_vae(self, vae, x):
        pre_encoder = Encoder_model(
            self.vae_num_layers,
            self.d_numerical,
            self.categories,
            self.vae_d_token,
            self.vae_n_head,
            self.vae_factor,
            bias=True,
        ).to(self.device)
        self.pre_decoder = Decoder_model(
            self.vae_num_layers,
            self.d_numerical,
            self.categories,
            self.vae_d_token,
            self.vae_n_head,
            self.vae_factor,
            bias=True,
        ).to(self.device)
        with torch.no_grad():
            pre_encoder.load_weights(vae)
            self.pre_decoder.load_weights(vae)
            pre_encoder.to(self.device), self.pre_decoder.to(self.device)
            pre_encoder.eval(), self.pre_decoder.eval()

            train_z = []
            x_num = torch.from_numpy(x[self.numerical_features].values).float()
            x_cat = torch.from_numpy(x[self.discrete_features].values).long()
            train_loader = FastTensorDataLoader(
                x_num,
                x_cat,
                batch_size=min(self.batch_size, len(x)),
                shuffle=False,
            )
            for batch_num, batch_cat in train_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                train_z.append(pre_encoder(batch_num, batch_cat).detach().cpu().numpy())
            train_z = np.concatenate(train_z)
        return train_z

    def _train_diffusion(self, train_z):
        train_z = torch.from_numpy(train_z).float()
        train_z = train_z[:, 1:, :]
        B, num_tokens, self.token_dim = train_z.shape
        in_dim = num_tokens * self.token_dim
        train_z = train_z.view(B, in_dim)
        self.sample_dim = train_z.shape[1]
        self.train_z_mean = train_z.mean(0)

        in_dim = train_z.shape[1]
        mean, std = train_z.mean(0), train_z.std(0)
        train_data = (train_z - mean) / 2

        train_loader = FastTensorDataLoader(
            None,
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

        denoise_fn = MLPDiffusion(
            in_dim,
            self.embedding_dim,
            mlp_dim=self.mlp_dim,
            mlp_layers=self.mlp_layers,
        ).to(self.device)

        model = Model(denoise_fn=denoise_fn, hid_dim=train_data.shape[1]).to(
            self.device
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.diffusion_wd,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20)
        model.train()

        epochs = resolve_epochs_from_training_steps(
            self.epochs,
            self.training_steps,
            B,
            self.batch_size,
        )
        for _ in tqdm(range(epochs)):
            batch_loss = 0.0
            len_input = 0
            for _, inputs in train_loader:
                inputs = inputs.to(self.device)
                loss = model(inputs)
                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

        return model

    def _compute_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += CE_LOSS_FN(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
            acc += (x_hat == X_cat[:, idx]).float().sum()
            total_num += x_hat.shape[0]

        ce_loss /= idx + 1
        acc /= total_num
        # loss = mse_loss + ce_loss

        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        return mse_loss, ce_loss, loss_kld, acc

    def save(self, path):
        path = Path(path)
        state = {
            "random_state": self.random_state,
            "target_column": self.target_column,
            "val_size": self.val_size,
            "vae_num_layers": self.vae_num_layers,
            "vae_training_steps": self.vae_training_steps,
            "vae_d_token": self.vae_d_token,
            "vae_n_head": self.vae_n_head,
            "vae_factor": self.vae_factor,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "training_steps": self.training_steps,
            "lr": self.lr,
            "embedding_dim": self.embedding_dim,
            "mlp_dim": self.mlp_dim,
            "mlp_layers": self.mlp_layers,
            "num_timesteps": self.num_timesteps,
            "ori_columns": self.ori_columns,
            "discrete_features": self.discrete_features,
            "numerical_features": self.numerical_features,
            "ordinal_encoder": self.ordinal_encoder,
            "scaler": self.scaler,
            "d_numerical": self.d_numerical,
            "categories": self.categories,
            "sample_dim": self.sample_dim,
            "token_dim": self.token_dim,
            "train_z_mean": self.train_z_mean,
        }
        save_generator_state(path, state)
        torch.save(self.model.state_dict(), path / "diffusion_model.pt")
        torch.save(self.pre_decoder.state_dict(), path / "pre_decoder.pt")
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        generator = restore_generator(cls, load_generator_state(path))
        generator.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(generator, "embedding_dim"):
            generator.embedding_dim = generator.diffusion_dim_t
        generator.mlp_dim = getattr(generator, "mlp_dim", generator.embedding_dim * 2)
        generator.mlp_layers = getattr(generator, "mlp_layers", 2)

        generator.pre_decoder = Decoder_model(
            generator.vae_num_layers,
            generator.d_numerical,
            generator.categories,
            generator.vae_d_token,
            generator.vae_n_head,
            generator.vae_factor,
            bias=True,
        ).to(generator.device)
        generator.pre_decoder.load_state_dict(
            torch.load(path / "pre_decoder.pt", map_location=generator.device)
        )
        generator.pre_decoder.eval()

        denoise_fn = MLPDiffusion(
            generator.sample_dim,
            generator.embedding_dim,
            mlp_dim=generator.mlp_dim,
            mlp_layers=generator.mlp_layers,
        ).to(generator.device)
        generator.model = Model(denoise_fn=denoise_fn, hid_dim=generator.sample_dim).to(
            generator.device
        )
        generator.model.load_state_dict(
            torch.load(path / "diffusion_model.pt", map_location=generator.device)
        )
        generator.model.eval()
        return generator
