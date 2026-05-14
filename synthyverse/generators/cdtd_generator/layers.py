import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from .utils import low_discrepancy_sampler


def normalize_emb(emb, dim):
    return F.normalize(emb, dim=dim, eps=1e-20)


class MixedTypeDiffusion(nn.Module):
    def __init__(
        self,
        model,
        dim,
        categories,
        proportions,
        num_features,
        sigma_data_cat,
        sigma_data_cont,
        sigma_min_cat,
        sigma_max_cat,
        sigma_min_cont,
        sigma_max_cont,
        cat_emb_init_sigma,
        timewarp_type="bytype",
        timewarp_weight_low_noise=1.0,
    ):
        super(MixedTypeDiffusion, self).__init__()

        self.dim = dim
        self.num_features = num_features
        self.num_cat_features = len(categories)
        self.num_cont_features = num_features - self.num_cat_features
        self.num_unique_cats = sum(categories)
        self.categories = categories
        self.model = model

        self.cat_emb = CatEmbedding(dim, categories, cat_emb_init_sigma, bias=True)
        self.register_buffer("sigma_data_cat", torch.tensor(sigma_data_cat))
        self.register_buffer("sigma_data_cont", torch.tensor(sigma_data_cont))

        entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions])
        self.register_buffer(
            "normal_const",
            torch.cat((entropy, torch.ones((self.num_cont_features,)))),
        )
        self.weight_network = WeightNetwork(1024)

        # timewarping
        self.timewarp_type = timewarp_type
        self.sigma_min_cat = torch.tensor(sigma_min_cat)
        self.sigma_max_cat = torch.tensor(sigma_max_cat)
        self.sigma_min_cont = torch.tensor(sigma_min_cont)
        self.sigma_max_cont = torch.tensor(sigma_max_cont)

        # combine sigma boundaries for transforming sigmas to [0,1]
        sigma_min = torch.cat(
            (
                torch.tensor(sigma_min_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_min_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        sigma_max = torch.cat(
            (
                torch.tensor(sigma_max_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_max_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        self.register_buffer("sigma_max", sigma_max)
        self.register_buffer("sigma_min", sigma_min)

        self.timewarp_cdf = Timewarp_Logistic(
            self.timewarp_type,
            self.num_cat_features,
            self.num_cont_features,
            sigma_min,
            sigma_max,
            weight_low_noise=timewarp_weight_low_noise,
            decay=0.0,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def diffusion_loss(self, x_cat_0, x_cont_0, cat_logits, cont_preds):
        assert len(cat_logits) == self.num_cat_features
        assert cont_preds.shape == x_cont_0.shape

        # cross entropy over categorical features for each individual
        ce_losses = torch.stack(
            [
                F.cross_entropy(cat_logits[i], x_cat_0[:, i], reduction="none")
                for i in range(self.num_cat_features)
            ],
            dim=1,
        )

        # MSE loss over numerical features
        mse_losses = (cont_preds - x_cont_0) ** 2

        return ce_losses, mse_losses

    def add_noise(self, x_cat_emb_0, x_cont_0, sigma):
        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        x_cat_emb_t = x_cat_emb_0 + torch.randn_like(x_cat_emb_0) * sigma_cat.unsqueeze(
            2
        )
        x_cont_t = x_cont_0 + torch.randn_like(x_cont_0) * sigma_cont

        return x_cat_emb_t, x_cont_t

    def loss_fn(self, x_cat, x_cont, u=None):
        batch = x_cat.shape[0] if x_cat is not None else x_cont.shape[0]

        # get ground truth data
        x_cat_emb_0 = self.cat_emb(x_cat)
        x_cont_0 = x_cont
        x_cat_0 = x_cat

        # draw u and convert to standard deviations for noise
        with torch.no_grad():
            if u is None:
                u = low_discrepancy_sampler(batch, device=self.device)  # (B,)
            sigma = self.timewarp_cdf(u, invert=True).detach().to(torch.float32)
            u = u.to(torch.float32)
            assert sigma.shape == (batch, self.num_features)

        x_cat_emb_t, x_cont_t = self.add_noise(x_cat_emb_0, x_cont_0, sigma)
        cat_logits, cont_preds = self.precondition(x_cat_emb_t, x_cont_t, u, sigma)
        ce_losses, mse_losses = self.diffusion_loss(
            x_cat_0, x_cont_0, cat_logits, cont_preds
        )

        # compute EDM weight
        sigma_cont = sigma[:, self.num_cat_features :]
        cont_weight = (sigma_cont**2 + self.sigma_data_cont**2) / (
            (sigma_cont * self.sigma_data_cont) ** 2 + 1e-7
        )

        losses = {}
        losses["unweighted"] = torch.cat((ce_losses, mse_losses), dim=1)
        losses["unweighted_calibrated"] = losses["unweighted"] / self.normal_const
        weighted_calibrated = (
            torch.cat((ce_losses, cont_weight * mse_losses), dim=1) / self.normal_const
        )
        c_noise = torch.log(u.to(torch.float32) + 1e-8) * 0.25
        time_reweight = self.weight_network(c_noise).unsqueeze(1)

        losses["timewarping"] = self.timewarp_cdf.loss_fn(
            sigma.detach(), losses["unweighted_calibrated"].detach()
        )
        weightnet_loss = (
            time_reweight.exp() - weighted_calibrated.detach().mean(1)
        ) ** 2
        losses["weighted_calibrated"] = (
            weighted_calibrated / time_reweight.exp().detach()
        )
        train_loss = (
            losses["weighted_calibrated"].mean()
            + losses["timewarping"].mean()
            + weightnet_loss.mean()
        )

        losses["train_loss"] = train_loss

        return losses, sigma

    def precondition(self, x_cat_emb_t, x_cont_t, u, sigma):
        """
        Improved preconditioning proposed in the paper "Elucidating the Design
        Space of Diffusion-Based Generative Models" (EDM) adjusted for categorical data
        """

        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        c_in_cat = (
            1 / (self.sigma_data_cat**2 + sigma_cat.unsqueeze(2) ** 2).sqrt()
        )  # batch, num_features, 1
        c_in_cont = 1 / (self.sigma_data_cont**2 + sigma_cont**2).sqrt()
        # c_noise = u.log() / 4
        c_noise = torch.log(u + 1e-8) * 0.25 * 1000

        cat_logits, cont_preds = self.model(
            c_in_cat * x_cat_emb_t,
            c_in_cont * x_cont_t,
            c_noise,
        )

        assert len(cat_logits) == self.num_cat_features
        assert cont_preds.shape == x_cont_t.shape

        # apply preconditioning to continuous features
        c_skip = self.sigma_data_cont**2 / (sigma_cont**2 + self.sigma_data_cont**2)
        c_out = (
            sigma_cont
            * self.sigma_data_cont
            / (sigma_cont**2 + self.sigma_data_cont**2).sqrt()
        )
        D_x = c_skip * x_cont_t + c_out * cont_preds

        return cat_logits, D_x

    def score_interpolation(self, x_cat_emb_t, cat_logits, sigma, return_probs=False):
        if return_probs:
            # transform logits for categorical features to probabilities
            probs = []
            for logits in cat_logits:
                probs.append(F.softmax(logits.to(torch.float64), dim=1))
            return probs

        def interpolate_emb(i):
            p = F.softmax(cat_logits[i].to(torch.float64), dim=1)
            true_emb = self.cat_emb.get_all_feat_emb(i).to(torch.float64)
            return torch.matmul(p, true_emb)

        # take prob-weighted average of normalized ground truth embeddings
        x_cat_emb_0_hat = torch.zeros_like(
            x_cat_emb_t, device=self.device, dtype=torch.float64
        )
        for i in range(self.num_cat_features):
            x_cat_emb_0_hat[:, i, :] = interpolate_emb(i)

        # plug interpolated embedding into score function to interpolate score
        sigma_cat = sigma[:, : self.num_cat_features]
        interpolated_score = (x_cat_emb_t - x_cat_emb_0_hat) / sigma_cat.unsqueeze(2)

        return interpolated_score, x_cat_emb_0_hat

    @torch.inference_mode()
    def sampler(self, cat_latents, cont_latents, num_steps=200):
        B = (
            cont_latents.shape[0]
            if self.num_cont_features > 0
            else cat_latents.shape[0]
        )

        # construct time steps
        u_steps = torch.linspace(
            1, 0, num_steps + 1, device=self.device, dtype=torch.float64
        )
        t_steps = self.timewarp_cdf(u_steps, invert=True)

        assert torch.allclose(t_steps[0].to(torch.float32), self.sigma_max.float())
        assert torch.allclose(t_steps[-1].to(torch.float32), self.sigma_min.float())
        # the final step goes onto t = 0, i.e., sigma = sigma_min = 0

        # initialize latents at maximum noise level
        t_cat_next = t_steps[0, : self.num_cat_features]
        t_cont_next = t_steps[0, self.num_cat_features :]
        x_cat_next = cat_latents.to(torch.float64) * t_cat_next.unsqueeze(1)
        x_cont_next = cont_latents.to(torch.float64) * t_cont_next

        for i, (t_cur, t_next, u_cur) in enumerate(
            zip(t_steps[:-1], t_steps[1:], u_steps[:-1])
        ):
            t_cur = t_cur.repeat((B, 1))
            t_next = t_next.repeat((B, 1))
            t_cont_cur = t_cur[:, self.num_cat_features :]

            # get score model output
            cat_logits, x_cont_denoised = self.precondition(
                x_cat_emb_t=x_cat_next.to(torch.float32),
                x_cont_t=x_cont_next.to(torch.float32),
                u=u_cur.to(torch.float32).repeat((B,)),
                sigma=t_cur.to(torch.float32),
            )

            # estimate scores
            d_cat_cur, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur)
            d_cont_cur = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur

            # adjust data samples
            h = t_next - t_cur
            x_cat_next = (
                x_cat_next + h[:, : self.num_cat_features].unsqueeze(2) * d_cat_cur
            )
            x_cont_next = x_cont_next + h[:, self.num_cat_features :] * d_cont_cur

        # final prediction of classes for categorical feature
        u_final = u_steps[:-1][-1]
        t_final = t_steps[:-1][-1].repeat(B, 1)

        cat_logits, _ = self.precondition(
            x_cat_emb_t=x_cat_next.to(torch.float32),
            x_cont_t=x_cont_next.to(torch.float32),
            u=u_final.to(torch.float32).repeat((B,)),
            sigma=t_final.to(torch.float32),
        )

        # get probabilities for each category and derive generated classes
        probs = self.score_interpolation(
            x_cat_next, cat_logits, t_final, return_probs=True
        )
        x_cat_gen = torch.empty(B, self.num_cat_features, device=self.device)
        for i in range(self.num_cat_features):
            x_cat_gen[:, i] = probs[i].argmax(1)

        return x_cat_gen.cpu(), x_cont_next.cpu()


class FourierFeatures(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        assert (emb_dim % 2) == 0
        self.half_dim = emb_dim // 2
        self.register_buffer("weights", torch.randn(1, self.half_dim))

    def forward(self, x):
        freqs = x.unsqueeze(1) * self.weights * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class WeightNetwork(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.fourier = FourierFeatures(emb_dim)
        self.fc = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, u):
        x = self.fourier(u)
        return self.fc(x).squeeze()

    def loss_fn(self, preds, avg_loss):
        # learn to fit expected average loss
        return (preds - avg_loss) ** 2


class TimeStepEmbedding(nn.Module):
    """
    Layer that embeds diffusion timesteps.

     Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        - fourer (bool): whether to use random fourier features as embeddings
    """

    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        n_layers: int = 2,
        fourier: bool = False,
        scale=16,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers
        self.fourier = fourier

        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")

        if fourier:
            self.register_buffer("freqs", torch.randn(dim // 2) * scale)

        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps):
        if not self.fourier:
            d, T = self.dim, self.max_period
            mid = d // 2
            fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
            fs = fs.to(timesteps.device)
            args = timesteps[:, None].float() * fs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)

        return self.fc(emb)


class FinalLayer(nn.Module):
    """
    Final layer that predicts logits for each category for categorical features
    and scalers for continuous features.
    """

    def __init__(self, dim_in, categories, num_cont_features, bias_init=None):
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categories)
        dim_out = sum(categories) + self.num_cont_features
        self.linear = nn.Linear(dim_in, dim_out)
        nn.init.zeros_(self.linear.weight)
        if bias_init is None:
            nn.init.zeros_(self.linear.bias)
        else:
            self.linear.bias = nn.Parameter(bias_init)
        self.split_chunks = [self.num_cont_features, *categories]
        self.cat_idx = 0
        if self.num_cont_features > 0:
            self.cat_idx = 1

    def forward(self, x):
        x = self.linear(x)
        out = torch.split(x, self.split_chunks, dim=-1)

        if self.num_cont_features > 0:
            cont_logits = out[0]
        else:
            cont_logits = None
        if self.num_cat_features > 0:
            cat_logits = out[self.cat_idx :]
        else:
            cat_logits = None

        return cat_logits, cont_logits


class PositionalEmbedder(nn.Module):
    """
    Positional embedding layer for encoding continuous features.
    Adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    """

    def __init__(self, dim, num_features, trainable=False, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(1, num_features, self.half_dim), requires_grad=trainable
        )
        self.sigma = freq_init_scale
        bound = self.sigma * 3
        nn.init.trunc_normal_(self.weights, 0.0, self.sigma, a=-bound, b=bound)

    def forward(self, x):
        x = rearrange(x, "b f -> b f 1")
        freqs = x * self.weights * 2 * torch.pi
        fourier = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fourier


class CatEmbedding(nn.Module):
    """
    Feature-specific embedding layer for categorical features.
    bias = True adds a learnable bias term to each feature, which is is same across categories.
    """

    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False):
        super().__init__()

        self.categories = torch.tensor(categories)
        categories_offset = self.categories.cumsum(dim=-1)[:-1]
        categories_offset = torch.cat(
            (torch.zeros((1,), dtype=torch.long), categories_offset)
        )
        self.register_buffer("categories_offset", categories_offset)
        self.dim = torch.tensor(dim)

        self.cat_emb = nn.Embedding(sum(categories), dim)
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma)

        self.bias = bias
        if self.bias:
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x):
        x = self.cat_emb(x + self.categories_offset)
        if self.bias:
            x += self.cat_bias
        # l2 normalize embedding
        x = normalize_emb(x, dim=2) * self.dim.sqrt()
        return x

    def get_all_feat_emb(self, feat_idx):
        emb_idx = (
            torch.arange(self.categories[feat_idx], device=self.cat_emb.weight.device)
            + self.categories_offset[feat_idx]
        )
        x = self.cat_emb(emb_idx)
        if self.bias:
            x += self.cat_bias[feat_idx]
        x = normalize_emb(x, dim=1) * self.dim.sqrt()
        return x


class MLP(nn.Module):
    """
    TabDDPM-like architecture for both continuous and categorical features.
    Used for TabDDPM and CDTD.
    """

    def __init__(
        self,
        num_cont_features,
        cat_emb_dim,
        categories,
        proportions,
        emb_dim,
        n_layers,
        n_units,
        act="relu",
    ):
        super().__init__()

        num_cat_features = len(categories)
        self.time_emb = TimeStepEmbedding(emb_dim, fourier=False)

        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = n_layers * [n_units]
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.fc = nn.Sequential(*layers)

        dim_in = num_cont_features + num_cat_features * cat_emb_dim
        self.proj = nn.Linear(dim_in, emb_dim)

        # init final layer
        cont_bias_init = torch.zeros((num_cont_features,))
        cat_bias_init = torch.cat(proportions).log()
        bias_init = torch.cat((cont_bias_init, cat_bias_init))

        self.final_layer = FinalLayer(
            out_dims[-1], categories, num_cont_features, bias_init=bias_init
        )

    def forward(self, x_cat_emb_t, x_cont_t, time):
        cond_emb = self.time_emb(time)
        x = torch.concat((rearrange(x_cat_emb_t, "B F D -> B (F D)"), x_cont_t), dim=-1)
        x = self.proj(x) + cond_emb
        x = self.fc(x)

        return self.final_layer(x)


class Timewarp_Logistic(nn.Module):
    """
    Our version of timewarping with exact cdfs instead of p.w.l. functions.
    We use a domain-adapted cdf of the logistic distribution.

    timewarp_type selects the type of timewarping:
        - single (single noise schedule, like CDCD)
        - bytype (per type noise schedule)
        - all (per feature noise schedule)
    """

    def __init__(
        self,
        timewarp_type,
        num_cat_features,
        num_cont_features,
        sigma_min,
        sigma_max,
        weight_low_noise=1.0,
        decay=0.0,
    ):
        super(Timewarp_Logistic, self).__init__()

        self.timewarp_type = timewarp_type
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features

        # save bounds for min max scaling
        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)

        if timewarp_type == "single":
            self.num_funcs = 1
        elif timewarp_type == "bytype":
            self.num_funcs = 2
        elif timewarp_type == "all":
            self.num_funcs = self.num_cat_features + self.num_cont_features

        # init parameters
        v = torch.tensor(1.01)
        logit_v = torch.log(torch.exp(v - 1) - 1)
        self.logits_v = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_v))
        self.register_buffer("init_v", self.logits_v.clone())

        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        logit_mu = torch.log(((1 / (1 - p_large_noise)) - 1)) / v
        self.logits_mu = nn.Parameter(
            torch.full((self.num_funcs,), fill_value=logit_mu)
        )
        self.register_buffer("init_mu", self.logits_mu.clone())

        # init gamma, scaling parameter to 1
        self.logits_gamma = nn.Parameter(
            (torch.ones((self.num_funcs, 1)).exp() - 1).log()
        )

        # for ema
        self.decay = decay
        logits_v_shadow = torch.clone(self.logits_v).detach()
        logits_mu_shadow = torch.clone(self.logits_mu).detach()
        logits_gamma_shadow = torch.clone(self.logits_gamma).detach()
        self.register_buffer("logits_v_shadow", logits_v_shadow)
        self.register_buffer("logits_mu_shadow", logits_mu_shadow)
        self.register_buffer("logits_gamma_shadow", logits_gamma_shadow)

    def update_ema(self):
        with torch.no_grad():
            self.logits_v.copy_(
                self.decay * self.logits_v_shadow
                + (1 - self.decay) * self.logits_v.detach()
            )
            self.logits_mu.copy_(
                self.decay * self.logits_mu_shadow
                + (1 - self.decay) * self.logits_mu.detach()
            )
            self.logits_gamma.copy_(
                self.decay * self.logits_gamma_shadow
                + (1 - self.decay) * self.logits_gamma.detach()
            )
            self.logits_v_shadow.copy_(self.logits_v)
            self.logits_mu_shadow.copy_(self.logits_mu)
            self.logits_gamma_shadow.copy_(self.logits_gamma)

    def get_params(self):
        logit_mu = self.logits_mu  # let underlying parameter be ln(mu / (1-mu))
        v = 1 + F.softplus(self.logits_v)  # v > 1
        scale = F.softplus(self.logits_gamma)
        return logit_mu, v, scale

    def cdf_fn(self, x, logit_mu, v):
        "mu in (0,1), v >= 1"
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return 1 / (1 + Z)

    def pdf_fn(self, x, logit_mu, v):
        Z = ((x / (1 - x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1 - x))) * (Z / ((1 + Z) ** 2))

    def quantile_fn(self, u, logit_mu, v):
        c = logit_mu + 1 / v * torch.special.logit(u, eps=1e-7)
        return F.sigmoid(c)

    def forward(self, x, invert=False, normalize=False, return_pdf=False):
        logit_mu, v, scale = self.get_params()

        if not invert:
            if normalize:
                scale = 1.0

            # can have more sigmas than cdfs
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)

            # ensure x is never 0 or 1 to ensure robustness
            x = torch.clamp(x, 1e-7, 1 - 1e-7)

            if self.timewarp_type == "single":
                # all sigmas are the same so just take first one
                input = x[:, 0].unsqueeze(0)

            elif self.timewarp_type == "bytype":
                # first sigma belongs to categorical feature, last to continuous feature
                input = torch.stack((x[:, 0], x[:, -1]), dim=0)

            elif self.timewarp_type == "all":
                input = x.T  # (num_features, batch)

            if return_pdf:
                output = (torch.vmap(self.pdf_fn, in_dims=0)(input, logit_mu, v)).T
            else:
                output = (
                    torch.vmap(self.cdf_fn, in_dims=0)(input, logit_mu, v) * scale
                ).T

        else:
            # have single u, need to repeat u
            input = repeat(x, "b -> f b", f=self.num_funcs)
            output = (torch.vmap(self.quantile_fn, in_dims=0)(input, logit_mu, v)).T

            if self.timewarp_type == "single":
                output = repeat(output, "b 1 -> b f", f=self.num_features)
            elif self.timewarp_type == "bytype":
                output = torch.column_stack(
                    (
                        repeat(output[:, 0], "b -> b f", f=self.num_cat_features),
                        repeat(output[:, 1], "b -> b f", f=self.num_cont_features),
                    )
                )

            zero_mask = x == 0.0
            one_mask = x == 1.0
            output = output.masked_fill(zero_mask.unsqueeze(-1), 0.0)
            output = output.masked_fill(one_mask.unsqueeze(-1), 1.0)

            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min

        return output

    def loss_fn(self, sigmas, losses):
        # losses and sigmas have shape (B, num_features)

        if self.timewarp_type == "single":
            # fit average loss (over all feature)
            losses = losses.mean(1, keepdim=True)  # (B,1)
        elif self.timewarp_type == "bytype":
            # fit average loss over cat and over cont features separately
            losses_cat = losses[:, : self.num_cat_features].mean(
                1, keepdim=True
            )  # (B,1)
            losses_cont = losses[:, self.num_cat_features :].mean(
                1, keepdim=True
            )  # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)

        losses_estimated = self.forward(sigmas)

        with torch.no_grad():
            pdf = self.forward(sigmas, return_pdf=True).detach()

        return ((losses_estimated - losses) ** 2) / (pdf + 1e-7)
