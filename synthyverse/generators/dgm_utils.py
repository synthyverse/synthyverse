import torch
import torch.nn as nn


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

        if drop_last:
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        self.n_batches = n_batches + (remainder > 0)

    def __iter__(self):
        self.indices = torch.randperm(self.dataset_len) if self.shuffle else None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
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
