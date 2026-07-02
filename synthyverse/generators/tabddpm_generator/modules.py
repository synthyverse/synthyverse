# Third-party notice: based on Apache-2.0-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution, NOTICE, and modification details.
"""
Code was adapted from https://github.com/Yura52/rtdl
"""
# stdlib
import math
from typing import Optional, Union

# third party
import torch
import torch.optim
from torch import Tensor, nn


def get_nonlin(nonlin: Union[str, nn.Module]) -> nn.Module:
    if isinstance(nonlin, nn.Module):
        return nonlin
    return {
        "none": nn.Identity,
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }[nonlin.lower()]()


class LinearLayer(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0,
        batch_norm: bool = False,
        nonlin: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(n_units_in, n_units_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units_out))
        if nonlin is not None:
            layers.append(get_nonlin(nonlin))
        self.model = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X.float())


class MLP(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        *,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if n_layers_hidden > 0:
            layers = [
                LinearLayer(
                    n_units_in,
                    n_units_hidden,
                    batch_norm=batch_norm,
                    nonlin=nonlin,
                )
            ]
            for _ in range(n_layers_hidden - 1):
                layers.append(
                    LinearLayer(
                        n_units_hidden,
                        n_units_hidden,
                        dropout=dropout,
                        batch_norm=batch_norm,
                        nonlin=nonlin,
                    )
                )
            layers.append(nn.Linear(n_units_hidden, n_units_out))
        else:
            layers = [nn.Linear(n_units_in, n_units_out)]
        self.model = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X.float())


class TimeStepEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        n_layers: int = 2,
        nonlin: Union[str, nn.Module] = "silu",
    ) -> None:
        """
        Create sinusoidal timestep embeddings.

        Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers

        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")

        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(get_nonlin(nonlin))

        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
        - timesteps (Tensor): 1D Tensor of N indices, one per batch element.
        """
        d, T = self.dim, self.max_period
        mid = d // 2
        fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
        fs = fs.to(timesteps.device)
        args = timesteps[:, None].float() * fs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.fc(emb)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_emb: int = 128,
        *,
        model_params: dict = {},
        conditional: bool = False,
        num_classes: int = 0,
        emb_nonlin: Union[str, nn.Module] = "silu",
        max_time_period: int = 10000,
    ) -> None:
        super().__init__()
        self.dim_t = dim_emb
        self.num_classes = num_classes
        self.has_label = conditional

        if isinstance(emb_nonlin, str):
            self.emb_nonlin = get_nonlin(emb_nonlin)
        else:
            self.emb_nonlin = emb_nonlin

        self.proj = nn.Linear(dim_in, dim_emb)
        self.time_emb = TimeStepEmbedding(dim_emb, max_time_period)

        if conditional:
            if self.num_classes > 0:
                self.label_emb = nn.Embedding(self.num_classes, dim_emb)
            elif self.num_classes == 0:  # regression
                self.label_emb = nn.Linear(1, dim_emb)

        if not model_params:
            model_params = {}  # avoid changing the default dict

        if not model_params:
            model_params = dict(n_units_hidden=256, n_layers_hidden=3, dropout=0.0)
        model_params.update(n_units_in=dim_emb, n_units_out=dim_in)
        self.model = MLP(**model_params)

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        emb = self.time_emb(t)
        if self.has_label:
            if y is None:
                raise ValueError("y must be provided if conditional is True")
            if self.num_classes == 0:
                y = y.reshape(-1, 1).float()
            else:
                y = y.squeeze().long()
            emb += self.emb_nonlin(self.label_emb(y))
        x = self.proj(x) + emb
        return self.model(x)
