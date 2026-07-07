import math
import random

from einops import rearrange, repeat
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


def cycle(dl):
    while True:
        for data in dl:
            yield data


def low_discrepancy_sampler(num_samples, device):
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (
        single_u
        + torch.arange(0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False)
    ) % 1


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


class CatEmbedding(nn.Module):
    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False, normalize_emb=False, norm_dim=None):
        super().__init__()
        self.categories = torch.tensor(categories)
        categories_offset = self.categories.cumsum(dim=-1)[:-1]
        categories_offset = torch.cat((torch.zeros((1,), dtype=torch.long), categories_offset))
        self.register_buffer("categories_offset", categories_offset)
        self.dim = torch.tensor(dim if norm_dim is None else norm_dim).pow(2 if norm_dim is not None else 1)
        self.normalize_emb = normalize_emb
        self.cat_emb = nn.Embedding(sum(categories), dim)
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma)
        self.bias = bias
        if self.bias:
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x):
        x = self.cat_emb(x + self.categories_offset)
        if self.bias:
            x += self.cat_bias
        if self.normalize_emb:
            x = F.normalize(x, dim=2, eps=1e-20) * self.dim.sqrt()
        return x


class FourierFeatures(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.register_buffer("weights", torch.randn(1, emb_dim // 2))

    def forward(self, x):
        freqs = x.unsqueeze(1) * self.weights * 2 * np.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1)


class WeightNetwork(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fourier = FourierFeatures(emb_dim)
        self.fc = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, u):
        return self.fc(self.fourier(u)).squeeze()


class Timewarp(nn.Module):
    def __init__(self, sigma_min, sigma_max, num_bins=100, decay=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))
        self.logits_t = nn.Parameter(torch.full((1, num_bins), -torch.tensor(num_bins).log()))
        self.logits_u = nn.Parameter(torch.full((1, num_bins), -torch.tensor(num_bins).log()))
        self.decay = decay
        self.register_buffer("logits_t_shadow", self.logits_t.clone().detach())
        self.register_buffer("logits_u_shadow", self.logits_u.clone().detach())

    def get_bins(self, invert, normalize):
        logits_t, logits_u = (self.logits_u, self.logits_t) if invert else (self.logits_t, self.logits_u)
        weights_u = F.softmax(logits_u, dim=1) if normalize or invert else logits_u.exp()
        weights_t = F.softmax(logits_t, dim=1)
        weights_u = weights_u + 1e-7
        weights_t = weights_t + 1e-7
        if normalize or invert:
            weights_u = weights_u / weights_u.sum(dim=1, keepdims=True)
        weights_t = weights_t / weights_t.sum(dim=1, keepdims=True)
        edges_t_right = torch.cumsum(weights_t, dim=1)
        edges_u_right = torch.cumsum(weights_u, dim=1)
        return (
            F.pad(edges_t_right[:, :-1], (1, 0), value=0),
            edges_t_right,
            F.pad(edges_u_right[:, :-1], (1, 0), value=0),
            weights_u / weights_t,
        )

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        edges_t_left, edges_t_right, edges_u_left, slopes = self.get_bins(invert, normalize)
        if not invert:
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)
        bin_idx = torch.searchsorted(edges_t_right, x.unsqueeze(0).contiguous(), right=False)
        bin_idx[bin_idx > self.num_bins - 1] = self.num_bins - 1
        slope = slopes.gather(dim=1, index=bin_idx)
        if return_pdf:
            return slope.T.squeeze(1).detach()
        interpolation = (edges_u_left.gather(dim=1, index=bin_idx) + (x - edges_t_left.gather(dim=1, index=bin_idx)) * slope).T.squeeze(1)
        if normalize:
            interpolation = torch.clamp(interpolation, 0, 1)
        if invert:
            interpolation = interpolation * (self.sigma_max - self.sigma_min) + self.sigma_min
        return interpolation

    def get_sigmas(self, u):
        return self.forward(u, invert=True, normalize=True).to(torch.float32)

    def get_t(self, sigma):
        return self.forward(sigma, invert=False, normalize=True).to(torch.float32)

    def loss_fn(self, sigmas, losses):
        return ((self.forward(sigmas) - losses) ** 2) / (self.forward(sigmas, return_pdf=True, normalize=True).detach() + 1e-8)


class TimewarpLogistic(nn.Module):
    def __init__(self, timewarp_type, num_cat_features, num_cont_features, sigma_min, sigma_max, weight_low_noise=1.0):
        super().__init__()
        self.timewarp_type = timewarp_type
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features
        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)
        self.num_funcs = 1 if timewarp_type == "single" else 2 if timewarp_type == "bytype" else self.num_features
        v = torch.tensor(1.01)
        logit_v = torch.log(torch.exp(v - 1) - 1)
        self.logits_v = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_v))
        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        logit_mu = torch.log(((1 / (1 - p_large_noise)) - 1)) / v
        self.logits_mu = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_mu))
        self.logits_gamma = nn.Parameter((torch.ones((self.num_funcs, 1)).exp() - 1).log())

    def get_params(self):
        return self.logits_mu, 1 + F.softplus(self.logits_v), F.softplus(self.logits_gamma)

    def cdf_fn(self, x, logit_mu, v):
        return 1 / (1 + ((x / (1 - x)) / logit_mu.exp()) ** (-v))

    def pdf_fn(self, x, logit_mu, v):
        z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1 - x))) * (z / ((1 + z) ** 2))

    def quantile_fn(self, u, logit_mu, v):
        return F.sigmoid(logit_mu + 1 / v * torch.special.logit(u, eps=1e-7))

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        logit_mu, v, scale = self.get_params()
        if not invert:
            if normalize:
                scale = 1.0
            x = torch.clamp((x - self.sigma_min) / (self.sigma_max - self.sigma_min), 1e-7, 1 - 1e-7)
            input = x[:, 0].unsqueeze(0) if self.timewarp_type == "single" else x.T
            if self.timewarp_type == "bytype":
                input = torch.stack((x[:, 0], x[:, -1]), dim=0)
            output = torch.vmap(self.pdf_fn if return_pdf else self.cdf_fn, in_dims=0)(input, logit_mu, v).T
            return output if return_pdf else output * scale

        output = torch.vmap(self.quantile_fn, in_dims=0)(repeat(x, "b -> f b", f=self.num_funcs), logit_mu, v).T
        if self.timewarp_type == "single":
            output = repeat(output, "b 1 -> b f", f=self.num_features)
        elif self.timewarp_type == "bytype":
            output = torch.column_stack(
                (
                    repeat(output[:, 0], "b -> b f", f=self.num_cat_features),
                    repeat(output[:, 1], "b -> b f", f=self.num_cont_features),
                )
            )
        output = output.masked_fill((x == 0.0).unsqueeze(-1), 0.0)
        output = output.masked_fill((x == 1.0).unsqueeze(-1), 1.0)
        return output * (self.sigma_max - self.sigma_min) + self.sigma_min

    def get_sigmas(self, t):
        return self.forward(t, invert=True, normalize=True).to(torch.float32)

    def loss_fn(self, sigmas, losses):
        if self.timewarp_type == "single":
            losses = losses.mean(1, keepdim=True)
        elif self.timewarp_type == "bytype":
            losses = torch.cat((losses[:, : self.num_cat_features].mean(1, keepdim=True), losses[:, self.num_cat_features :].mean(1, keepdim=True)), dim=1)
        return ((self.forward(sigmas) - losses) ** 2) / (self.forward(sigmas, return_pdf=True).detach() + 1e-7)


class FinalLayer(nn.Module):
    def __init__(self, dim_in, categories, bias_init=None):
        super().__init__()
        self.linear = nn.Linear(dim_in, sum(categories))
        nn.init.zeros_(self.linear.weight)
        self.linear.bias = nn.Parameter(bias_init) if bias_init is not None else self.linear.bias
        if bias_init is None:
            nn.init.zeros_(self.linear.bias)
        self.categories = categories

    def forward(self, x):
        return torch.split(self.linear(x), self.categories, dim=-1)


class LowResMLP(nn.Module):
    def __init__(self, num_classes, cat_emb_dim, emb_dim, n_layers, n_units, proportions, act="relu"):
        super().__init__()
        self.num_features = len(num_classes)
        self.time_emb = TimeStepEmbedding(emb_dim)
        self.proj = nn.Linear(self.num_features * cat_emb_dim, emb_dim)
        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = (n_layers - 1) * [n_units] + [emb_dim]
        layers = []
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.final_layer = FinalLayer(out_dims[-1], num_classes, torch.cat(proportions).log() if proportions is not None else None)

    def forward(self, x_emb_t, t):
        x = self.proj(rearrange(x_emb_t, "B F D -> B (F D)")) + self.time_emb(t)
        return self.final_layer(self.mlp(x))


class CatCDTD(nn.Module):
    def __init__(
        self,
        score_model,
        num_classes,
        proportions,
        emb_dim,
        sigma_min=1e-5,
        sigma_max=100,
        sigma_data=1,
        normalize_by_entropy=True,
        weight_low_noise=1.0,
        timewarp_variant="logistic",
        cat_emb_init_sigma=0.001,
    ):
        super().__init__()
        self.num_features = len(num_classes)
        self.num_classes = num_classes
        self.sigma_data = sigma_data
        self.emb_dim = emb_dim
        entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions]) if normalize_by_entropy else torch.ones(self.num_features)
        self.register_buffer("entropy", entropy)
        self.score_model = score_model
        self.weightnet = WeightNetwork(1024)
        self.encoder = CatEmbedding(emb_dim, num_classes, cat_emb_init_sigma, bias=True, normalize_emb=True)
        self.timewarp_variant = timewarp_variant
        if timewarp_variant == "logistic":
            self.timewarp = TimewarpLogistic(
                "single",
                self.num_features,
                0,
                torch.tensor(sigma_min),
                torch.tensor(sigma_max),
                weight_low_noise=weight_low_noise,
            )
        elif timewarp_variant == "pwl":
            self.timewarp = Timewarp(sigma_min=sigma_min, sigma_max=sigma_max, decay=0.1)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def loss_fn(self, x, t=None, validation=False):
        batch_size = x.shape[0]
        x_emb = self.encoder(x)
        with torch.no_grad():
            if t is None:
                t = low_discrepancy_sampler(batch_size, device=self.device)
            if validation:
                sigma = (
                    low_discrepancy_sampler(batch_size, device=self.device).to(torch.float32)
                    * (self.timewarp.sigma_max - self.timewarp.sigma_min)
                    + self.timewarp.sigma_min
                )
                t = self.timewarp.get_t(sigma)
            else:
                sigma = self.timewarp.get_sigmas(t)
            sigma = repeat(sigma, "B F -> B F G", F=self.num_features, G=1)
            t = t.to(torch.float32)
        logits = self.precondition(x_emb + torch.randn_like(x_emb) * sigma, t, sigma)
        ce_losses = torch.stack(
            [F.cross_entropy(logits[i], x[:, i], reduction="none") for i in range(self.num_features)],
            dim=1,
        )
        weighted = ce_losses / (self.entropy + 1e-8)
        time_reweight = self.weightnet(t).unsqueeze(1)
        weighted_calibrated = weighted / time_reweight.exp().detach()
        if self.timewarp_variant == "logistic":
            timewarping = self.timewarp.loss_fn(sigma.squeeze(-1).detach(), weighted.detach())
        else:
            timewarping = self.timewarp.loss_fn(sigma.detach()[:, 0].squeeze(-1), weighted.detach().mean(1).squeeze(-1))
        weightnet = (time_reweight.exp() - weighted.detach().mean(1)) ** 2
        return {"train_loss": weighted_calibrated.mean() + timewarping.mean() + weightnet.mean()}

    def precondition(self, x_emb_t, t, sigma):
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        return self.score_model(c_in * x_emb_t, torch.log(t + 1e-8) * 0.25 * 1000)

    def init_score_interpolation(self):
        full_emb = self.encoder.cat_emb.weight.data.detach()
        bias = torch.row_stack(
            [self.encoder.cat_bias[i].unsqueeze(0).expand(self.num_classes[i], -1) for i in range(self.num_features)]
        )
        full_emb = F.normalize(full_emb + bias, dim=1, eps=1e-20) * torch.tensor(self.emb_dim).sqrt()
        self.embeddings_score_interp = torch.split(full_emb.to(torch.float64), self.num_classes, dim=0)

    def score_interpolation(self, x_emb_t, logits, sigma, return_probs=False):
        probs = [F.softmax(l.to(torch.float64), dim=1) for l in logits]
        if return_probs:
            return probs
        x_emb_hat = torch.zeros_like(x_emb_t, device=self.device, dtype=torch.float64)
        for i, prob in enumerate(probs):
            x_emb_hat[:, i, :] = torch.matmul(prob, self.embeddings_score_interp[i])
        return (x_emb_t - x_emb_hat) / sigma, x_emb_hat

    @torch.inference_mode()
    def sampler(self, latents, num_steps=200):
        batch_size = latents.shape[0]
        t_steps = torch.linspace(1, 0, num_steps + 1, device=self.device, dtype=torch.float64)
        s_steps = self.timewarp.get_sigmas(t_steps).to(torch.float64)
        x_next = latents.to(torch.float64) * s_steps[0].unsqueeze(1)
        for s_cur, s_next, t_cur in zip(s_steps[:-1], s_steps[1:], t_steps[:-1]):
            s_cur = s_cur.repeat((batch_size, 1))
            s_next = s_next.repeat((batch_size, 1))
            logits = self.precondition(x_next.to(torch.float32), t_cur.to(torch.float32).repeat((batch_size,)), s_cur.to(torch.float32).unsqueeze(-1))
            d_cur, _ = self.score_interpolation(x_next, logits, s_cur.unsqueeze(-1))
            x_next = x_next + (s_next - s_cur).unsqueeze(-1) * d_cur
        s_final = s_steps[:-1][-1].repeat(batch_size, 1)
        logits = self.precondition(
            x_next.to(torch.float32),
            t_steps[:-1][-1].to(torch.float32).repeat((batch_size,)),
            s_final.to(torch.float32).unsqueeze(-1),
        )
        probs = self.score_interpolation(x_next, logits, s_final.unsqueeze(-1), return_probs=True)
        x_gen = torch.empty(batch_size, self.num_features, device=self.device)
        for i in range(self.num_features):
            x_gen[:, i] = probs[i].argmax(1)
        return x_gen.cpu()

    @torch.inference_mode()
    def sample_data(self, num_samples, num_steps=200, batch_size=4096, seed=42, verbose=True):
        self.init_score_interpolation()
        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(num_samples, batch_size)
        sample_sizes = n_batches * [batch_size] + ([remainder] if remainder else [])
        x = []
        for num in tqdm(sample_sizes, disable=not verbose):
            latents = torch.randn((num, self.num_features, self.emb_dim), device=self.device)
            x.append(self.sampler(latents, num_steps=num_steps))
        return torch.cat(x).cpu().long()
