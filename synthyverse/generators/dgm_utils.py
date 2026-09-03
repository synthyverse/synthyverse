import random
from contextlib import contextmanager
from math import ceil
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Optional
from ..evaluation.fidelity import ClassifierTwoSampleTest


@contextmanager
def preserve_rng_state(generator):
    generator_random_state = generator.random_state
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        generator.random_state = generator_random_state
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def validate_c2st(
    self,
    X: pd.DataFrame,
    nfolds: int = 5,
    random_state: int = 0,
    n_repeats: int = 3,
):
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    with preserve_rng_state(self):
        result = 0
        for i in range(n_repeats):
            state = random_state + i + 1
            syn = self.generate(len(X), random_state=state)
            score = ClassifierTwoSampleTest(nfold=nfolds, random_state=state).evaluate(
                X.reset_index(drop=True), syn, discrete_features=self.discrete_features
            )["c2st.auc"]
            result += abs(score - 0.5)
        return result / n_repeats


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def split_validation(
    X: pd.DataFrame,
    val_size: float,
    target_column: Optional[str] = None,
    discrete_features: list[str] = [],
    random_state: int = 42,
    max_validation_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or len(X) <= 1:
        return X.copy(), None
    if target_column is not None and target_column not in X:
        raise ValueError(f"target_column '{target_column}' is not in X.")
    n_val = min(int(ceil(len(X) * val_size)), len(X) - 1)
    if max_validation_rows is not None:
        if max_validation_rows <= 0:
            raise ValueError("max_validation_rows must be positive.")
        n_val = min(n_val, max_validation_rows)
    if n_val <= 0:
        return X.copy(), None
    if target_column is not None:
        return train_test_split(
            X, test_size=n_val, random_state=random_state, stratify=X[target_column]
        )
    else:
        return train_test_split(X, test_size=n_val, random_state=random_state)


class QuantileStandardScaler:
    def __init__(self, n_samples: int, random_state: int):
        self.quantile_transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(n_samples // 30, 1000), 10),
            subsample=int(1e9),
            random_state=random_state,
        )
        self.standard_scaler = StandardScaler()

    def fit(self, X, y=None):
        self.standard_scaler.fit(self.quantile_transformer.fit_transform(X))
        return self

    def transform(self, X):
        return self.standard_scaler.transform(self.quantile_transformer.transform(X))

    def fit_transform(self, X, y=None):
        return self.standard_scaler.fit_transform(
            self.quantile_transformer.fit_transform(X)
        )

    def inverse_transform(self, X):
        return self.quantile_transformer.inverse_transform(
            self.standard_scaler.inverse_transform(X)
        )


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, embedding_dim=512, mlp_dim=2048, mlp_layers=2):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("MLPDiffusion requires an even embedding_dim")

        self.embedding_dim = embedding_dim
        self.dim_t = embedding_dim
        self.mlp_dim = mlp_dim

        self.proj = nn.Linear(d_in, embedding_dim)

        layers = [nn.Linear(embedding_dim, self.mlp_dim), nn.SiLU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(self.mlp_dim, self.mlp_dim), nn.SiLU()]
        layers += [
            nn.Linear(self.mlp_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, d_in),
        ]
        self.mlp = nn.Sequential(*layers)

        self.map_noise = PositionalEmbedding(num_channels=embedding_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x, timesteps, class_labels=None):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


class FastTensorDataLoader:
    """Iterate over tensors in mini-batches."""

    def __init__(self, *data, batch_size=32, shuffle=False, drop_last=False):
        self.dataset_len = next(t.shape[0] for t in data if t is not None)
        assert all(t.shape[0] == self.dataset_len for t in data if t is not None)
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        self.n_batches = n_batches if drop_last else n_batches + (remainder > 0)
        self.iter_len = (
            self.n_batches * self.batch_size if drop_last else self.dataset_len
        )

    def __iter__(self):
        self.indices = torch.randperm(self.dataset_len) if self.shuffle else None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.iter_len:
            raise StopIteration

        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(
                torch.index_select(t, 0, indices) if t is not None else None
                for t in self.data
            )
        else:
            batch = tuple(
                t[self.i : self.i + self.batch_size] if t is not None else None
                for t in self.data
            )

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
