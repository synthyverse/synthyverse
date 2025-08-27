import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
import torch

from ..base import BaseGenerator
from .cdtd_dir import CDTD


class CDTDGenerator(BaseGenerator):
    name = "cdtd"

    def __init__(
        self,
        cat_emb_dim: int = 16,
        mlp_emb_dim: int = 256,
        mlp_n_layers: int = 5,
        mlp_n_units: int = 1024,
        sigma_data_cat: float = 1.0,
        sigma_data_cont: float = 1.0,
        sigma_min_cat: float = 0.0,
        sigma_min_cont: float = 0.0,
        sigma_max_cat: float = 100.0,
        sigma_max_cont: float = 80.0,
        cat_emb_init_sigma: float = 0.001,
        timewarp_type: str = "bytype",  # 'single', 'bytype', or 'all'
        timewarp_weight_low_noise: float = 1.0,
        num_steps_train: int = 30_000,
        num_steps_warmup: int = 1000,
        batch_size: int = 4096,
        lr: float = 1e-3,
        ema_decay: float = 0.999,
        log_steps: int = 100,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.cdtd_params = {
            "cat_emb_dim": cat_emb_dim,
            "mlp_emb_dim": mlp_emb_dim,
            "mlp_n_layers": mlp_n_layers,
            "mlp_n_units": mlp_n_units,
            "sigma_data_cat": sigma_data_cat,
            "sigma_data_cont": sigma_data_cont,
            "sigma_min_cat": sigma_min_cat,
            "sigma_min_cont": sigma_min_cont,
            "sigma_max_cat": sigma_max_cat,
            "sigma_max_cont": sigma_max_cont,
            "cat_emb_init_sigma": cat_emb_init_sigma,
            "timewarp_type": timewarp_type,  # 'single', 'bytype', or 'all'
            "timewarp_weight_low_noise": timewarp_weight_low_noise,
        }
        self.training_params = {
            "num_steps_train": num_steps_train,
            "num_steps_warmup": num_steps_warmup,
            "batch_size": batch_size,
            "lr": lr,
            "ema_decay": ema_decay,
            "log_steps": log_steps,
            "seed": self.random_state,
        }

        self.sample_params = {
            "num_steps": 200,
            "batch_size": batch_size,
            "seed": self.random_state,
        }

        # GPUs can use fast float32 operations
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        numerical_features = [col for col in X.columns if col not in discrete_features]
        # retain original column order to output correct dataframe format after generation
        self.col_order = X.columns

        # ordinally encode discrete columns
        X_discrete = X[discrete_features].to_numpy().astype(str)
        self.ord_encoder = OrdinalEncoder()
        X_discrete = self.ord_encoder.fit_transform(X_discrete)

        # quantile transform and standard scale numericals (tries to put 30 samples per bin, but caps range inside [10,1000])
        X_numerical = X[numerical_features].to_numpy().astype(float)
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

        self.cdtd = CDTD(
            X_cat_train=X_discrete, X_cont_train=X_numerical, **self.cdtd_params
        )

        self.cdtd.fit(
            X_cat_train=X_discrete, X_cont_train=X_numerical, **self.training_params
        )

    def _generate_data(self, n: int):

        # synthesize
        syn_X_discrete, syn_X_numerical = self.cdtd.sample(
            num_samples=n, **self.sample_params
        )

        # postprocess to format expected by basegenerator
        syn_X_discrete = self.ord_encoder.inverse_transform(syn_X_discrete)
        syn_X_numerical = self.scaler.inverse_transform(syn_X_numerical)
        syn_X_numerical = self.quant_encoder.inverse_transform(syn_X_numerical)

        # combine to synthetic dataset
        syn_X = pd.concat(
            (pd.DataFrame(syn_X_discrete), pd.DataFrame(syn_X_numerical)), axis=1
        )
        syn_X.columns = self.discrete_features + self.numerical_features
        # rearrange columns to correct order
        syn_X = syn_X[self.col_order]

        return syn_X
