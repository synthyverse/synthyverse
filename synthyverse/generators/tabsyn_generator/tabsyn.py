from ..base import TabularBaseGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from .vae import Model_VAE, Encoder_model, Decoder_model
from .diffusion import MLPDiffusion, Model, sample


class TabSynGenerator(TabularBaseGenerator):
    """TabSyn: a latent diffusion model for tabular data.

    Trains a VAE to learn a latent representation of tabular data, then fits a diffusion model in that latent space.

    Paper: "Mixed-type tabular data synthesis with score-based diffusion in latent space" by Zhang et al. (2023).
    Based on the paper's original implementation: https://github.com/amazon-science/tabsyn/

    Args:
        target_column (str): Name of the target column used for stratified validation splitting.
        vae_lr (float): Learning rate for VAE training. Default: 1e-3.
        vae_wd (float): Weight decay used by the VAE optimizer. Default: 0.
        vae_d_token (int): Token embedding dimension used by the VAE. Default: 4.
        vae_n_head (int): Number of attention heads in the VAE transformer blocks. Default: 1.
        vae_factor (int): Expansion factor used in VAE feed-forward layers. Default: 32.
        vae_num_layers (int): Number of VAE encoder/decoder layers. Default: 2.
        vae_batch_size (int): Batch size used to train the VAE. Default: 4096.
        vae_num_epochs (int): Maximum number of VAE epochs. Default: 4000.
        vae_min_beta (float): Minimum KL coefficient for VAE KL annealing. Default: 1e-5.
        vae_max_beta (float): Initial/maximum KL coefficient for VAE KL annealing. Default: 1e-2.
        vae_lambda (float): Multiplicative decay factor applied to beta when validation plateaus.
            Default: 0.7.
        diffusion_batch_size (int): Batch size used to train the latent diffusion model. Default: 4096.
        diffusion_num_epochs (int): Maximum number of diffusion training epochs. Default: 10001.
        diffusion_dim_t (int): Time embedding dimension used by the diffusion denoiser. Default: 1024.
        diffusion_lr (float): Learning rate for diffusion training. Default: 1e-3.
        diffusion_wd (float): Weight decay used by the diffusion optimizer. Default: 0.
        diffusion_sampling_steps (int): Number of reverse diffusion steps used for generation. Default: 50.
        diffusion_patience (int): Patience in diffusion training for early stopping. Default: 500.
        num_workers (int): Number of workers for PyTorch data loaders. Increase to speed up training - but may cause issues when locally training on Windows OS. Default: 0.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to `TabularBaseGenerator`.

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
        ...     diffusion_num_epochs=500,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "tabsyn"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        vae_lr: float = 1e-3,
        vae_wd: float = 0,
        vae_d_token: int = 4,
        vae_n_head: int = 1,
        vae_factor: int = 32,
        vae_num_layers: int = 2,
        vae_batch_size: int = 4096,
        vae_num_epochs: int = 4000,
        vae_min_beta: float = 1e-5,
        vae_max_beta: float = 1e-2,
        vae_lambda: float = 0.7,
        diffusion_batch_size: int = 4096,
        diffusion_num_epochs: int = 10_000 + 1,
        diffusion_dim_t: int = 1024,
        diffusion_lr: float = 1e-3,
        diffusion_wd: float = 0,
        diffusion_sampling_steps: int = 50,
        diffusion_patience: int = 500,
        num_workers: int = 0,  # number of workers in pytorch dataloader (>0 can give issues on windows)
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.random_state = random_state
        self.target_column = target_column
        self.num_workers = num_workers

        self.vae_params = {
            "LR": vae_lr,
            "WD": vae_wd,
            "D_TOKEN": vae_d_token,
            "N_HEAD": vae_n_head,
            "FACTOR": vae_factor,
            "NUM_LAYERS": vae_num_layers,
            "BATCH_SIZE": vae_batch_size,
            "NUM_EPOCHS": vae_num_epochs,
            "MIN_BETA": vae_min_beta,
            "MAX_BETA": vae_max_beta,
            "LAMBDA": vae_lambda,
        }

        self.diffusion_params = {
            "BATCH_SIZE": diffusion_batch_size,
            "NUM_EPOCHS": diffusion_num_epochs,
            "DIM_T": diffusion_dim_t,
            "LR": diffusion_lr,
            "WD": diffusion_wd,
            "PATIENCE": diffusion_patience,
            "SAMPLING_STEPS": diffusion_sampling_steps,
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_model(
        self, X: pd.DataFrame, X_val: pd.DataFrame = None, discrete_features: list = []
    ):
        x = X.copy()
        self.ori_columns = X.columns
        self.discrete_features = discrete_features

        if X_val is None:
            stratify = (
                x[self.target_column]
                if self.target_column in discrete_features
                else None
            )
            x, x_val = train_test_split(
                x, test_size=0.1, random_state=self.random_state, stratify=stratify
            )
        else:
            x_val = X_val.copy()

        self.scaler = StandardScaler()
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

        # train the VAE
        train_z = self._train_vae(x, x_val)

        # no longer need original data
        del x, x_val

        # train the diffusion model
        self.model = self._train_diffusion(train_z)

    def _generate_data(self, n: int):
        self.model.eval()
        with torch.no_grad():
            x = sample(
                self.model.denoise_fn_D,
                n,
                self.sample_dim,
                self.device,
                num_steps=self.diffusion_params["SAMPLING_STEPS"],
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

        return syn

    def _train_vae(self, x, x_val):
        d_numerical = len(x.columns.tolist()) - len(self.discrete_features)
        categories = []
        for col in self.discrete_features:
            categories.append(x[col].nunique())

        vae = Model_VAE(
            self.vae_params["NUM_LAYERS"],
            d_numerical,
            categories,
            self.vae_params["D_TOKEN"],
            n_head=self.vae_params["N_HEAD"],
            factor=self.vae_params["FACTOR"],
            bias=True,
        ).to(self.device)
        pre_encoder = Encoder_model(
            self.vae_params["NUM_LAYERS"],
            d_numerical,
            categories,
            self.vae_params["D_TOKEN"],
            self.vae_params["N_HEAD"],
            self.vae_params["FACTOR"],
            bias=True,
        ).to(self.device)
        self.pre_decoder = Decoder_model(
            self.vae_params["NUM_LAYERS"],
            d_numerical,
            categories,
            self.vae_params["D_TOKEN"],
            self.vae_params["N_HEAD"],
            self.vae_params["FACTOR"],
            bias=True,
        ).to(self.device)
        pre_encoder.eval()
        self.pre_decoder.eval()

        optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=self.vae_params["LR"],
            weight_decay=self.vae_params["WD"],
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.95, patience=10, verbose=True
        )

        current_lr = optimizer.param_groups[0]["lr"]
        patience = 0
        batch_size = min(self.vae_params["BATCH_SIZE"], len(x))

        beta = self.vae_params["MAX_BETA"]

        x_tr_num = x[
            [col for col in x.columns if col not in self.discrete_features]
        ].values
        x_tr_num = torch.from_numpy(x_tr_num).float()
        x_tr_cat = x[self.discrete_features].values
        x_tr_cat = torch.from_numpy(x_tr_cat).long()

        train_loader = DataLoader(
            TensorDataset(x_tr_num, x_tr_cat),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # 4
        )

        x_val_num = x_val[
            [col for col in x.columns if col not in self.discrete_features]
        ].values
        x_val_num = torch.from_numpy(x_val_num).float()
        x_val_cat = x_val[self.discrete_features].values
        x_val_cat = torch.from_numpy(x_val_cat).long()

        best_val_loss = float("inf")

        pbar = tqdm(range(self.vae_params["NUM_EPOCHS"]))

        for _ in pbar:

            for batch_num, batch_cat in train_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                vae.train()
                optimizer.zero_grad()

                Recon_X_num, Recon_X_cat, mu_z, std_z = vae(batch_num, batch_cat)

                loss_mse, loss_ce, loss_kld, train_acc = self._compute_loss(
                    batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )

                loss = loss_mse + loss_ce + beta * loss_kld
                pbar.set_postfix(
                    {
                        "Train MSE": loss_mse.item(),
                        "Train CE": loss_ce.item(),
                        "Train KLD": loss_kld.item(),
                    }
                )
                loss.backward()
                optimizer.step()

            vae.eval()
            with torch.no_grad():
                x_val_num = x_val_num.to(self.device)
                x_val_cat = x_val_cat.to(self.device)
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
                    best_vae = vae.state_dict()
                    patience = 0
                else:
                    patience += 1
                    if patience == 10:
                        if beta > self.vae_params["MIN_BETA"]:
                            beta = beta * self.vae_params["LAMBDA"]

        vae.load_state_dict(best_vae)

        # getting latent embeddings
        with torch.no_grad():
            pre_encoder.load_weights(vae)
            self.pre_decoder.load_weights(vae)

            train_z = []
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

        train_loader = DataLoader(
            train_data,
            batch_size=self.diffusion_params["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.num_workers,
        )

        denoise_fn = MLPDiffusion(in_dim, self.diffusion_params["DIM_T"]).to(
            self.device
        )

        model = Model(denoise_fn=denoise_fn, hid_dim=train_data.shape[1]).to(
            self.device
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.diffusion_params["LR"],
            weight_decay=self.diffusion_params["WD"],
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=20, verbose=True
        )
        model.train()

        best_loss = float("inf")
        patience = 0

        for epoch in tqdm(range(self.diffusion_params["NUM_EPOCHS"])):
            batch_loss = 0.0
            len_input = 0
            for inputs in train_loader:
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

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                best_model = model.state_dict()
            else:
                patience += 1
                if patience == 500:
                    print("Early stopping")
                    break

        model.load_state_dict(best_model)
        return model

    def _compute_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
            acc += (x_hat == X_cat[:, idx]).float().sum()
            total_num += x_hat.shape[0]

        ce_loss /= idx + 1
        acc /= total_num
        # loss = mse_loss + ce_loss

        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        return mse_loss, ce_loss, loss_kld, acc
