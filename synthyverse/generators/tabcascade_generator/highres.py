import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


def low_discrepancy_sampler(num_samples, device):
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (
        single_u
        + torch.arange(0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False)
    ) % 1


class PolyNoiseSchedule(nn.Module):
    def __init__(self, emb_dim, num_features, gamma_min=0.0, gamma_max=1.0, grad_min_epsilon=0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_range = self.gamma_max - self.gamma_min
        self.grad_min_epsilon = grad_min_epsilon
        self.h_net = nn.Sequential(
            nn.Linear(emb_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.SiLU(),
        )
        self.l_a = nn.Linear(num_features, num_features)
        nn.init.zeros_(self.l_a.weight)
        nn.init.zeros_(self.l_a.bias)
        self.l_b = nn.Linear(num_features, num_features)
        self.l_c = nn.Linear(num_features, num_features)

    def forward(self, emb, t):
        if t.numel() == 1:
            t = t * torch.ones((emb.shape[0], 1), device=emb.device)
        else:
            t = t.unsqueeze(-1)
        a, b, c = self.get_params(emb)
        return self._eval_poly(t, a, b, c)

    def get_grads(self, emb, t):
        t = t.unsqueeze(-1)
        a, b, c = self.get_params(emb)
        return self._grad_t(t, a, b, c)

    def get_params(self, emb):
        h = self.h_net(emb)
        return self.l_a(h), self.l_b(h), 1e-3 + F.softplus(self.l_c(h))

    def _eval_poly(self, t, a, b, c):
        polynomial = (
            (a**2) * (t**5) / 5.0
            + (b**2 + 2 * a * c) * (t**3) / 3.0
            + a * b * (t**4) / 2.0
            + b * c * (t**2)
            + (c**2 + self.grad_min_epsilon) * t
        )
        scale = (
            (a**2) / 5.0
            + (b**2 + 2 * a * c) / 3.0
            + a * b / 2.0
            + b * c
            + (c**2 + self.grad_min_epsilon)
        )
        return self.gamma_min + self.gamma_range * polynomial / scale

    def _grad_t(self, t, a, b, c):
        polynomial = (
            (a**2) * (t**4)
            + (b**2 + 2 * a * c) * (t**2)
            + a * b * (t**3) * 2.0
            + b * c * t * 2
            + c**2
        )
        scale = (a**2) / 5.0 + (b**2 + 2 * a * c) / 3.0 + a * b / 2.0 + b * c + c**2
        return self.gamma_range * polynomial / scale


class CatEmbedding(nn.Module):
    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False):
        super().__init__()
        self.categories = torch.tensor(categories)
        categories_offset = self.categories.cumsum(dim=-1)[:-1]
        categories_offset = torch.cat((torch.zeros((1,), dtype=torch.long), categories_offset))
        self.register_buffer("categories_offset", categories_offset)
        self.cat_emb = nn.Embedding(sum(categories), dim)
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma)
        self.bias = bias
        if self.bias:
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x):
        x = self.cat_emb(x + self.categories_offset)
        if self.bias:
            x += self.cat_bias
        return x


class TimeStepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000, n_layers=2, fourier=False, scale=16):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.fourier = fourier
        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")
        if fourier:
            self.register_buffer("freqs", torch.randn(dim // 2) * scale)
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps):
        if not self.fourier:
            mid = self.dim // 2
            fs = torch.exp(-math.log(self.max_period) / mid * torch.arange(mid, dtype=torch.float32))
            args = timesteps[:, None].float() * fs.to(timesteps.device)[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)
        return self.fc(emb)


class HighResMLP(nn.Module):
    def __init__(self, num_features, x_low_dim, emb_dim, n_layers, n_units, act="relu"):
        super().__init__()
        self.num_features = num_features
        self.time_emb = TimeStepEmbedding(emb_dim)
        self.proj = nn.Linear(self.num_features, emb_dim)
        self.proj_low = nn.Sequential(
            nn.Linear(x_low_dim, 2 * emb_dim),
            nn.SiLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = (n_layers - 1) * [n_units] + [emb_dim]
        layers = []
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.final_layer = nn.Linear(out_dims[-1], self.num_features)

    def forward(self, x_t, x_low, t):
        c_noise = torch.log(t + 1e-8) * 0.25 * 1000
        x = self.proj(x_t) + self.time_emb(c_noise) + self.proj_low(x_low)
        return self.final_layer(self.mlp(x))


class HighResFlowModel(nn.Module):
    def __init__(
        self,
        group_means,
        group_stds,
        categories,
        emb_dim,
        n_layers,
        n_units,
        gamma_input_dim,
        cat_emb_dim,
    ):
        super().__init__()
        self.num_features = len(group_means)
        n_groups = torch.tensor([len(m) for m in group_means])
        group_offset = n_groups.cumsum(dim=-1)[:-1]
        group_offset = torch.cat((torch.zeros((1,), dtype=torch.long), group_offset))
        self.register_buffer("group_offset", group_offset)
        self.get_group_means = nn.Embedding.from_pretrained(torch.cat(group_means).unsqueeze(-1), freeze=True)
        self.get_group_stds = nn.Embedding.from_pretrained(torch.cat(group_stds).unsqueeze(-1), freeze=True)
        self.emb = CatEmbedding(cat_emb_dim, categories, cat_emb_init_sigma=1)
        self.proj_to_gamma = nn.Sequential(
            nn.Linear(len(categories) * cat_emb_dim, 2 * gamma_input_dim),
            nn.SiLU(),
            nn.Linear(2 * gamma_input_dim, gamma_input_dim),
            nn.SiLU(),
        )
        self.gamma = PolyNoiseSchedule(gamma_input_dim, self.num_features)
        self.mlp = HighResMLP(
            self.num_features,
            len(categories) * cat_emb_dim,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_units=n_units,
        )

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def loss_fn(self, x_num, x_cat, z_num, mask):
        means = self.get_group_means(z_num + self.group_offset).squeeze(-1)
        stds = self.get_group_stds(z_num + self.group_offset).squeeze(-1)
        x_1 = x_num
        x_0 = means + stds * torch.randn_like(x_num)
        d_cat = torch.column_stack((x_cat, z_num)) if x_cat is not None else z_num
        x_low = self.emb(d_cat).flatten(1)
        e_gamma = self.proj_to_gamma(x_low)
        t = low_discrepancy_sampler(x_num.shape[0], device=x_num.device).to(torch.float32)
        gamma_t = self.gamma(e_gamma, t)
        gamma_t_grad = self.gamma.get_grads(e_gamma, t)
        x_t = gamma_t * x_1 + (1 - gamma_t) * x_0
        loss = gamma_t_grad.pow(2) * (x_1 - x_0 - self.mlp(x_t, x_low, t)).pow(2)
        obs_mask = ~mask
        return ((loss * obs_mask).sum(1) / (obs_mask.sum(1) + 1e-8)).mean()

    def u_t(self, x_t, x_low, t, gamma_t_grad=None):
        if gamma_t_grad is None:
            gamma_t_grad = self.gamma.get_grads(self.proj_to_gamma(x_low), t)
        return gamma_t_grad * self.mlp(x_t, x_low, t)

    @torch.inference_mode()
    def sampler(self, x_cat, z_num, num_steps=200):
        batch_size = x_cat.shape[0]
        x_low = self.emb(torch.column_stack((x_cat, z_num))).flatten(1)
        t_steps = torch.linspace(0, 1, num_steps + 1, device=self.device, dtype=torch.float32)
        means = self.get_group_means(z_num + self.group_offset).squeeze(-1)
        stds = self.get_group_stds(z_num + self.group_offset).squeeze(-1)
        x_next = means + stds * torch.randn_like(means)
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            t_cur = t_cur.repeat((batch_size,))
            t_next = t_next.repeat((batch_size,))
            x_next = x_next + (t_next - t_cur).unsqueeze(1) * self.u_t(x_next, x_low, t_cur)
        return x_next.cpu()

    @torch.inference_mode()
    def sample_data(self, x_cat, z_num, num_steps=200, batch_size=4096, seed=42, verbose=True):
        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(x_cat.shape[0], batch_size)
        sample_sizes = n_batches * [batch_size] + ([remainder] if remainder else [])
        x_cat_parts = torch.split(x_cat, sample_sizes, dim=0)
        z_num_parts = torch.split(z_num, sample_sizes, dim=0)
        x = []
        for i in tqdm(range(len(sample_sizes)), disable=not verbose):
            x.append(
                self.sampler(
                    x_cat_parts[i].to(self.device),
                    z_num_parts[i].to(self.device),
                    num_steps=num_steps,
                )
            )
        return torch.cat(x).cpu()
