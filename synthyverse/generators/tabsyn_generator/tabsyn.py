from ..base import BaseGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


from .tabsyn_dir.process_dataset import process_data
from .tabsyn_dir.utils_train import preprocess
from .tabsyn_dir.vae.main import train_vae
from .tabsyn_dir.main import train_tabsyn
from .tabsyn_dir.sample import sample_tabsyn


class TabSynGenerator(BaseGenerator):
    name = "tabsyn"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        val_size: float = 0.1,
        random_state: int = 0,
        vae_lr: float = 1e-3,
        vae_wd: float = 0,
        vae_d_token: int = 4,
        vae_token_bias: bool = True,
        vae_n_head: int = 1,
        vae_factor: int = 32,
        vae_num_layers: int = 2,
        vae_batch_size: int = 4096,
        vae_num_epochs: int = 4000,
        vae_min_beta: float = 1e-5,
        vae_max_beta: float = 1e-2,
        vae_lambda: float = 0.7,
        diffusion_batch_size: int = 4096,
        diffusion_num_epochs: int = 10000 + 1,
        diffusion_dim_t: int = 1024,
        diffusion_lr: float = 1e-3,
        diffusion_wd: float = 0,
        diffusion_patience: int = 500,
    ):
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.target_column = target_column

        self.val_size = val_size

        self.vae_params = {
            "LR": vae_lr,
            "WD": vae_wd,
            "D_TOKEN": vae_d_token,
            "TOKEN_BIAS": vae_token_bias,
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
        }

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        stratify = (
            X[self.target_column] if self.target_column in discrete_features else None
        )
        # create validation set for early stopping
        self.X, self.val_X = train_test_split(
            X,
            test_size=self.val_size,
            stratify=stratify,
            random_state=self.random_state,
        )

        task_type = (
            "binclass" if self.target_column in discrete_features else "regression"
        )
        if (
            self.target_column in discrete_features
            and X[self.target_column].nunique() > 2
        ):
            task_type = "multiclass"

        # create metadata
        self.metadata = {
            "task_type": task_type,
            "column_names": X.columns.tolist(),
            "num_col_idx": [
                X.columns.get_loc(x)
                for x in X.columns
                if x not in discrete_features and x != self.target_column
            ],  # list of indices of numerical columns
            "cat_col_idx": [
                X.columns.get_loc(x)
                for x in X.columns
                if x in discrete_features and x != self.target_column
            ],  # list of indices of categorical columns
            "target_col_idx": [
                X.columns.get_loc(self.target_column)
            ],  # list of indices of the target columns
        }

        # split data and update metadata
        (
            self.metadata,
            X_num_train,
            X_cat_train,
            y_train,
            X_num_test,
            X_cat_test,
            y_test,
        ) = process_data(self.X, self.val_X, self.metadata)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, _, _, _, self.num_inverse, self.cat_inverse = preprocess(
            X_num_train,
            X_cat_train,
            y_train,
            X_num_test,
            X_cat_test,
            y_test,
            self.metadata,
            task_type=self.metadata["task_type"],
            inverse=True,
        )

        train_z, self.pre_decoder = train_vae(
            X_num_train,
            X_cat_train,
            y_train,
            X_num_test,
            X_cat_test,
            y_test,
            self.metadata,
            self.device,
            self.vae_params,
        )
        self.diffusion_model, self.train_z_shape, self.train_z_mean, self.token_dim = (
            train_tabsyn(train_z, self.diffusion_params, self.device)
        )
        self.metadata["token_dim"] = self.token_dim

    def _generate_data(self, n: int):

        syn = sample_tabsyn(
            n=n,
            num_inverse=self.num_inverse,
            cat_inverse=self.cat_inverse,
            info=self.metadata,
            diffusion_model=self.diffusion_model,
            pre_decoder=self.pre_decoder,
            train_z_shape=self.train_z_shape,
            train_z_mean=self.train_z_mean,
            device=self.device,
        )

        return syn
