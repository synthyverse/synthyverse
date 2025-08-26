import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from .layers import MLP, CatEmbedding, Timewarp_Logistic, WeightNetwork
from .utils import (
    FastTensorDataLoader,
    LinearScheduler,
    cycle,
    low_discrepancy_sampler,
    set_seeds,
)


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


class CDTD:
    def __init__(
        self,
        X_cat_train,
        X_cont_train,
        cat_emb_dim=16,
        mlp_emb_dim=256,
        mlp_n_layers=5,
        mlp_n_units=1024,
        sigma_data_cat=1.0,
        sigma_data_cont=1.0,
        sigma_min_cat=0.0,
        sigma_min_cont=0.0,
        sigma_max_cat=100.0,
        sigma_max_cont=80.0,
        cat_emb_init_sigma=0.001,
        timewarp_type="bytype",  # 'single', 'bytype', or 'all'
        timewarp_weight_low_noise=1.0,
    ):
        super().__init__()

        self.num_cat_features = X_cat_train.shape[1]
        self.num_cont_features = X_cont_train.shape[1]
        self.num_features = self.num_cat_features + self.num_cont_features
        self.cat_emb_dim = cat_emb_dim

        # derive number of categories for each categorical feature
        self.categories = []
        for i in range(self.num_cat_features):
            uniq_vals = np.unique(X_cat_train[:, i])
            self.categories.append(len(uniq_vals))

        # derive proportions for max CE losses at t = 1 for normalization
        self.proportions = []
        n_sample = X_cat_train.shape[0]
        for i in range(len(self.categories)):
            _, counts = X_cat_train[:, i].unique(return_counts=True)
            self.proportions.append(counts / n_sample)

        score_model = MLP(
            self.num_cont_features,
            self.cat_emb_dim,
            self.categories,
            self.proportions,
            mlp_emb_dim,
            mlp_n_layers,
            mlp_n_units,
        )

        self.diff_model = MixedTypeDiffusion(
            model=score_model,
            dim=self.cat_emb_dim,
            categories=self.categories,
            num_features=self.num_features,
            sigma_data_cat=sigma_data_cat,
            sigma_data_cont=sigma_data_cont,
            sigma_min_cat=sigma_min_cat,
            sigma_max_cat=sigma_max_cat,
            sigma_min_cont=sigma_min_cont,
            sigma_max_cont=sigma_max_cont,
            proportions=self.proportions,
            cat_emb_init_sigma=cat_emb_init_sigma,
            timewarp_type=timewarp_type,
            timewarp_weight_low_noise=timewarp_weight_low_noise,
        )

    def fit(
        self,
        X_cat_train,
        X_cont_train,
        num_steps_train=30_000,
        num_steps_warmup=1000,
        batch_size=4096,
        lr=1e-3,
        seed=42,
        ema_decay=0.999,
        log_steps=100,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = FastTensorDataLoader(
            X_cat_train,
            X_cont_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_iter = cycle(train_loader)

        set_seeds(seed, cuda_deterministic=True)
        self.diff_model = self.diff_model.to(self.device)
        self.diff_model.train()

        self.ema_diff_model = ExponentialMovingAverage(
            self.diff_model.parameters(), decay=ema_decay
        )

        self.optimizer = torch.optim.AdamW(
            self.diff_model.parameters(), lr=lr, weight_decay=0
        )
        self.scheduler = LinearScheduler(
            num_steps_train,
            base_lr=lr,
            final_lr=1e-6,
            warmup_steps=num_steps_warmup,
            warmup_begin_lr=1e-6,
            anneal_lr=True,
        )

        self.current_step = 0
        n_obs = sum_loss = 0
        train_start = time.time()

        with tqdm(
            initial=self.current_step,
            total=num_steps_train,
        ) as pbar:
            while self.current_step < num_steps_train:
                self.optimizer.zero_grad()

                inputs = next(train_iter)
                x_cat, x_cont = (
                    input.to(self.device) if input is not None else None
                    for input in inputs
                )

                losses, _ = self.diff_model.loss_fn(x_cat, x_cont, None)
                losses["train_loss"].backward()

                # update parameters
                self.optimizer.step()
                self.diff_model.timewarp_cdf.update_ema()
                self.ema_diff_model.update()

                sum_loss += losses["train_loss"].detach().mean().item() * x_cat.shape[0]
                n_obs += x_cat.shape[0]
                self.current_step += 1
                pbar.update(1)

                if self.current_step % log_steps == 0:
                    pbar.set_description(
                        f"Loss (last {log_steps} steps): {(sum_loss / n_obs):.3f}"
                    )
                    n_obs = sum_loss = 0

                # anneal learning rate
                if self.scheduler:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.scheduler(self.current_step)

        # compute training duration
        train_duration = time.time() - train_start
        print(f"Training took {(train_duration / 60):.2f} min.")

        # take EMA of model parameters
        self.ema_diff_model.copy_to()
        self.diff_model.eval()

        return self.diff_model

    def sample(self, num_samples, num_steps=200, batch_size=4096, seed=42):
        # set_seeds(seed, cuda_deterministic=True)
        # random_state is only updated across fits not across inferences. resetting seed will thus cause same sample each inference.

        n_batches, remainder = divmod(num_samples, batch_size)
        sample_sizes = (
            n_batches * [batch_size] + [remainder]
            if remainder != 0
            else n_batches * [batch_size]
        )

        x_cat_list = []
        x_cont_list = []

        for num_samples in tqdm(sample_sizes):
            cat_latents = torch.randn(
                (num_samples, self.num_cat_features, self.cat_emb_dim),
                device=self.device,
            )
            cont_latents = torch.randn(
                (num_samples, self.num_cont_features), device=self.device
            )
            x_cat_gen, x_cont_gen = self.diff_model.sampler(
                cat_latents, cont_latents, num_steps
            )
            x_cat_list.append(x_cat_gen)
            x_cont_list.append(x_cont_gen)

        x_cat = torch.cat(x_cat_list).cpu()
        x_cont = torch.cat(x_cont_list).cpu()

        return x_cat.long().numpy(), x_cont.numpy()
