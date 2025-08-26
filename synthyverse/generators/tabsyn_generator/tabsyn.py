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

    # TBD:
    # - check whether multiclass is handled properly
    # - add all params as arguments in generator constructor
    # - test results
    # - add more documentation on parameters

    def __init__(
        self,
        target_column: str = "target",
        vae_epochs: int = 4000,
        diffusion_epochs: int = 10000 + 1,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.target_column = target_column

        self.max_beta = 1e-2
        self.min_beta = 1e-5
        self.lambd = 0.7
        self.val_size = 0.1
        self.vae_epochs = vae_epochs
        self.diffusion_epochs = diffusion_epochs

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
        if X[self.target_column].nunique() > 2:
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
            self.max_beta,
            self.min_beta,
            self.lambd,
            self.device,
            num_epochs=self.vae_epochs,
        )
        self.diffusion_model, self.train_z_shape, self.train_z_mean, self.token_dim = (
            train_tabsyn(train_z, self.diffusion_epochs, self.device)
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
