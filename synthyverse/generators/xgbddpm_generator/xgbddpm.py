from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from xgb_diffusion import XGBDDPMRegressor, XGBDDPMClassifier

from joblib import Parallel, delayed
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.utils import check_random_state
from tqdm import tqdm

from ..base import BaseGenerator


def _vp_sched(
    T: int,
    beta_min: float,
    beta_max: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.linspace(eps, 1.0, T)
    dt = (1.0 - eps) / T
    betas = dt * (beta_min + ts * (beta_max - beta_min))
    betas = np.clip(betas, 1e-8, (1.0 - 1e-8))
    alphas = 1.0 - betas
    a_bars = np.cumprod(alphas, axis=0)
    return betas, alphas, a_bars


class XGBDDPMGenerator(BaseGenerator):
    """DDPM using XGBoost as backbone.

    Uses Gaussian diffusion for numerical features and multinomial diffusion for
    categorical features.

    Allows marginalizing over noise seeds across boosting rounds to avoid
    massively extending the training set.

    Supports stochastic (DDPM) and deterministic (DDIM) sampling.

    Args:
        target_column (str): Column used for conditional generation when it is
            categorical and ``model_per_label=True``.
        num_timesteps (int): Number of diffusion timesteps for training and
            sampling. Default: 50.
        refresh_every_k (int): Number of boosting rounds between noise seed
            refreshes. Default: 1.
        noise_samples_per_row (int): Number of times to extend the training set,
            to marginalize over noise seeds. Default: 1.
        n_jobs (int): Number of parallel jobs used to fit and run column models.
            Default: -1.
        n_jobs_xgb (int): Number of threads used inside each XGBoost estimator.
            Default: 1.
        beta_min (float): Minimum beta value for the variance-preserving noise
            schedule. Default: 0.1.
        beta_max (float): Maximum beta value for the variance-preserving noise
            schedule. Default: 8.0.
        eps (float): Lower endpoint of the diffusion time grid. Default: 0.0.
        xgboost_params (dict, optional): Parameters passed to each
            DDPM-enabled XGBoost estimator. Default: ``{"n_estimators": 500,
            "max_depth": 6, "early_stopping_rounds": 20,
            "min_boosting_round": 50, "eta": 0.06}``.
        clip_extremes (bool): Whether to clip generated numerical values to
            the training data range. Default: True.
        deterministic_sampler (bool): Whether to use DDIM-style deterministic
            numerical sampling instead of DDPM sampling. Default: False.
        objective (str): Numerical prediction objective. Options: "x" predicts
            the clean value, "epsilon" predicts the added noise, and "v" predicts
            the velocity parameterization. Default: "epsilon".
        model_per_timestep (bool): Whether to train separate models per diffusion
            timestep. If False, timestep is appended as an input feature. Default:
            True.
        model_per_label (bool): Whether to train separate models per categorical
            ``target_column`` value. Default: True.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional keyword arguments accepted for API compatibility.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import XGBDDPMGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator
        >>> generator = XGBDDPMGenerator(
        ...     target_column="target",
        ...     num_timesteps=50,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "xgbddpm"

    def __init__(
        self,
        target_column: str,
        num_timesteps: int = 50,
        refresh_every_k: int = 1,
        noise_samples_per_row: int = 1,
        n_jobs: int = -1,
        n_jobs_xgb: int = 1,
        beta_min: float = 0.1,
        beta_max: float = 8.0,
        eps: float = 0.0,
        xgboost_params: Optional[Dict[str, Any]] = {
            "n_estimators": 500,
            "max_depth": 6,
            "early_stopping_rounds": 20,
            "min_boosting_round": 50,
            "eta": 0.06,
        },
        clip_extremes: bool = True,
        deterministic_sampler: bool = False,
        objective: str = "epsilon",  # x, epsilon, or v
        model_per_timestep: bool = True,
        model_per_label: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        self.objective = str(objective).lower()
        assert self.objective in (
            "x",
            "epsilon",
            "v",
        ), "objective must be either x, epsilon, or v"

        self.deterministic_sampler = deterministic_sampler
        self.target_column = target_column
        self.num_timesteps = num_timesteps
        self.timesteps = num_timesteps
        self.noise_samples_per_row = noise_samples_per_row
        self.n_jobs = n_jobs
        self.n_jobs_xgb = n_jobs_xgb
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

        self.random_state = random_state
        self.clip_extremes = clip_extremes
        self.model_per_timestep = model_per_timestep
        self.model_per_label = model_per_label
        self.refresh_every_k = refresh_every_k
        self.xgboost_params = (
            xgboost_params.copy() if xgboost_params is not None else {}
        )
        self.xgboost_params.update(
            {
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs": n_jobs_xgb,
                "refresh_every_k": refresh_every_k,
            }
        )

        self.num_sampler = (
            self._ddim_update if self.deterministic_sampler else self._ddpm_update
        )

        self.rng = check_random_state(self.random_state)

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        # store original training column order for exact reconstruction
        self.ori_cols = X.columns.tolist()

        self.discrete_features = [x for x in X.columns if x in discrete_features]
        self.numerical_features = [c for c in X.columns if c not in discrete_features]
        self.numerical_features_set = set(self.numerical_features)

        # When conditioning on a categorical target, keep it outside the diffusion state.
        self.disc_features_x = [
            col
            for col in self.discrete_features
            if col != self.target_column or not self.model_per_label
        ]
        self.num_features_x = self.numerical_features
        self.is_conditional = (
            self.target_column in self.discrete_features and self.model_per_label
        )

        X_tr = X.copy()

        # ensure contiguous encoding of categoricals
        if len(self.discrete_features) > 0:
            self.ord_enc = OrdinalEncoder()
            X_tr[self.discrete_features] = self.ord_enc.fit_transform(
                X_tr[self.discrete_features]
            ).astype(int)
        self.n_cls = (
            X_tr[self.discrete_features].max(axis=0) + 1
            if len(self.discrete_features) > 0
            else None
        )

        # store numeric min/max in ORIGINAL space for clipping during generation
        self.clip_extremes = len(self.numerical_features) > 0 and self.clip_extremes
        if self.clip_extremes:
            self.num_min_ = X[self.numerical_features].min(axis=0).to_numpy()
            self.num_max_ = X[self.numerical_features].max(axis=0).to_numpy()

        X_tr = self._scale(X_tr)

        # build diffusion schedule used by the DDPM estimators and sampler
        self.betas_, self.alphas_, self.alpha_bars_ = _vp_sched(
            self.timesteps, self.beta_min, self.beta_max, self.eps
        )

        self.n_cls_ = (
            self.n_cls.loc[self.disc_features_x].to_numpy(dtype=np.int64)
            if self.n_cls is not None
            else None
        )

        self.labels = X_tr[self.target_column]

        cols = self.num_features_x + self.disc_features_x
        cond = [None]

        if self.is_conditional:
            cond = X_tr[self.target_column].unique()
        else:
            cols = self.numerical_features + self.discrete_features

        self.model_cols_x = cols
        self.model_col_idx = {col: i for i, col in enumerate(cols)}
        self.model_disc_features = [x for x in self.discrete_features if x in cols]
        model_disc_features_set = set(self.model_disc_features)
        self.model_feature_types = [
            "c" if x in model_disc_features_set else "q" for x in cols
        ]
        if not self.model_per_timestep:
            self.model_feature_types.append("c")
        self.model_n_classes = (
            self.n_cls.loc[self.model_disc_features].to_numpy(dtype=np.int64)
            if self.model_disc_features
            else None
        )

        fit_timesteps = range(self.timesteps) if self.model_per_timestep else [None]
        tasks = (
            delayed(self._fit_one)(X_tr, t, lv, col)
            for t in tqdm(fit_timesteps, desc="Fitting models")
            for lv in cond
            for col in cols
        )
        res = Parallel(n_jobs=self.n_jobs, prefer="threads")(tasks)

        self.models = {}
        self.label_encoders = {}

        for model, label_enc, t, lv, col in res:
            self.models.setdefault(t, {})
            self.models[t].setdefault(lv, {})
            self.models[t][lv].setdefault(col, model)

            if label_enc is not None:
                self.label_encoders.setdefault(t, {})
                self.label_encoders[t].setdefault(lv, {})
                self.label_encoders[t][lv].setdefault(col, label_enc)

    def _generate(self, n: int):

        if self.is_conditional:
            lv_samples = self.labels.sample(n, replace=True, random_state=self.rng)
        else:
            lv_samples = np.zeros(n, dtype=np.int64)  # dummy for consistent logic

        cols = self.model_cols_x
        cond = np.unique(lv_samples)
        syn = np.empty((n, len(cols)))
        for lv in cond:
            idx = np.where(lv_samples == lv)[0]
            if len(idx) == 0:
                continue

            # initialize pure noise array
            x_t = self._init_noise(
                len(idx), len(self.num_features_x), len(self.disc_features_x)
            )

            for t in reversed(range(self.timesteps)):
                # parallelized inference over cols
                tasks = (
                    delayed(self._inference_one_col)(x_t, j, col, t, lv)
                    for j, col in enumerate(cols)
                )
                res = Parallel(n_jobs=self.n_jobs, prefer="threads")(tasks)

                # build x0_hat
                x0_hat = np.empty_like(x_t)
                for j, out in res:
                    x0_hat[:, j] = out

                if t == 0:
                    break

                # numerical update (ddpm or ddim)
                if len(self.num_features_x) > 0:
                    x_t = self.num_sampler(x_t, x0_hat, t)

                # categorical update
                if len(self.disc_features_x) > 0:
                    x_t = self._cat_update(x_t, x0_hat, t)

            # synthesized data per label
            syn[idx, :] = x0_hat
        syn = pd.DataFrame(syn, columns=cols)

        if self.is_conditional:
            syn[self.target_column] = lv_samples.to_numpy()

        # reinstate original column order
        ori_cols = [x for x in self.ori_cols if x in syn.columns]
        syn = syn[ori_cols]

        # inverse scale
        syn = self._inverse_scale(syn)

        if len(self.discrete_features) > 0:
            syn[self.discrete_features] = self.ord_enc.inverse_transform(
                syn[self.discrete_features].astype(float)
            )

        # clip to original training ranges
        if self.clip_extremes:
            syn[self.numerical_features] = np.clip(
                syn[self.numerical_features],
                self.num_min_[None, :],
                self.num_max_[None, :],
            )

        return syn

    def _fit_one(
        self,
        X: pd.DataFrame,
        t: Optional[int],
        lv: Optional[int],
        col: str,
    ):
        cols = self.model_cols_x
        mask = X[self.target_column] == lv if lv is not None else slice(None)
        x_in = X.loc[mask, cols].to_numpy(dtype=np.float32, copy=True)

        y_in = X.loc[mask, col].to_numpy()

        params = self.xgboost_params.copy()
        params.update(
            {
                "alpha_bars": self.alpha_bars_,
                "n_classes": (
                    self.model_n_classes.copy()
                    if self.model_n_classes is not None
                    else None
                ),
                "noise_samples_per_row": self.noise_samples_per_row,
                "timestep": t,
                "target_index": self.model_col_idx[col],
                "feature_types": self.model_feature_types.copy(),
            }
        )

        label_enc = None
        if col in self.numerical_features_set:
            params["ddpm_objective"] = (
                "eps" if self.objective == "epsilon" else self.objective
            )
            model = XGBDDPMRegressor(**params)
        else:
            # for some timesteps and/or subsets there may not be an ordinal encoding so label encode
            label_enc = LabelEncoder()
            y_in = label_enc.fit_transform(y_in)
            params["ddpm_objective"] = "x"
            model = XGBDDPMClassifier(**params)

        fit_kwargs = {"verbose": False}
        if params.get("early_stopping_rounds"):
            fit_kwargs["eval_set"] = [(x_in, y_in)]
        model.fit(x_in, y_in, **fit_kwargs)
        if lv is None:
            lv = 0
        return model, label_enc, 0 if t is None else t, lv, col

    def _init_noise(self, n: int, d_num: int, d_disc: int):
        dtype = float if d_num > 0 else np.int64
        x_t = np.empty((n, d_num + d_disc), dtype=dtype)
        if d_num > 0:
            x_t[:, :d_num] = self.rng.normal(size=(n, d_num))

        if d_disc > 0:
            for j in range(d_disc):
                x_t[:, d_num + j] = self.rng.randint(0, int(self.n_cls_[j]), size=n)
        return x_t

    def _ddpm_update(
        self,
        x_t,
        x0_hat,
        t,
    ):
        coef1 = (np.sqrt(self.alpha_bars_[t - 1]) * self.betas_[t]) / (
            1.0 - self.alpha_bars_[t]
        )
        coef2 = (np.sqrt(self.alphas_[t]) * (1.0 - self.alpha_bars_[t - 1])) / (
            1.0 - self.alpha_bars_[t]
        )

        dnum = len(self.num_features_x)

        _x0_hat = x0_hat[:, :dnum]
        _x_t = x_t[:, :dnum]
        shape = (len(x_t), dnum)

        mu = coef1 * _x0_hat + coef2 * _x_t
        var = (
            (1.0 - self.alpha_bars_[t - 1]) / (1.0 - self.alpha_bars_[t])
        ) * self.betas_[t]
        var = max(var, 1e-20)
        z = self.rng.normal(size=shape)
        _x_t = mu + np.sqrt(var) * z

        x_t[:, :dnum] = _x_t
        return x_t

    def _ddim_update(
        self,
        x_t,
        x0_hat,
        t,
    ):

        dnum = len(self.num_features_x)

        _x_t = x_t[:, :dnum]
        _x0_hat = x0_hat[:, :dnum]

        eps_hat = (_x_t - np.sqrt(self.alpha_bars_[t]) * _x0_hat) / np.sqrt(
            1.0 - self.alpha_bars_[t]
        )
        _x_t = (
            np.sqrt(self.alpha_bars_[t - 1]) * _x0_hat
            + np.sqrt(1.0 - self.alpha_bars_[t - 1]) * eps_hat
        )
        x_t[:, :dnum] = _x_t

        return x_t

    def _cat_update(self, x_t, x0_hat, t):
        dnum = len(self.num_features_x)
        for j in range(len(self.disc_features_x)):
            jj = dnum + j
            K = int(self.n_cls_[j])
            x_t[:, jj] = self._sample_disc_posterior(
                x_t[:, jj].astype(np.int64),
                x0_hat[:, jj].astype(np.int64),
                K=K,
                t=t,
            )
        return x_t

    def _sample_disc_posterior(
        self,
        xt: np.ndarray,  # (N,) int in [0..K-1]
        x0_hat: np.ndarray,  # (N,) int in [0..K-1]
        K: int,
        t: int,  # current timestep index (1..T-1)
    ) -> np.ndarray:
        """
        Sample x_{t-1} for the categorical forward kernel:
          q(x_t | x_{t-1}) = alpha_t * I + (1-alpha_t) * Uniform(K)
        and prior:
          q(x_{t-1} | x0) = a_bar_{t-1} * onehot(x0) + (1-a_bar_{t-1}) * Uniform(K)

        Posterior:
          q(x_{t-1}=k | x_t, x0) is proportional to
          q(x_t | x_{t-1}=k) * q(x_{t-1}=k | x0)
        """

        base_prior = (1.0 - self.alpha_bars_[t - 1]) / K
        base_lik = (1.0 - self.alphas_[t]) / K

        N = xt.shape[0]
        probs = np.full((N, K), base_prior * base_lik, dtype=np.float64)
        rows = np.arange(N)

        probs[rows, xt] += base_prior * self.alphas_[t]
        probs[rows, x0_hat] += base_lik * self.alpha_bars_[t - 1]

        same = xt == x0_hat
        if np.any(same):
            idx = rows[same]
            probs[idx, xt[idx]] += self.alpha_bars_[t - 1] * self.alphas_[t]

        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(probs_sum, 1e-12, None)

        u = self.rng.random(size=N)
        cdf = np.cumsum(probs, axis=1)
        x_prev = (cdf < u[:, None]).sum(axis=1).astype(np.int64)
        return x_prev

    def _inference_one_col(self, x_t, j, col, t, lv):
        # grab model
        model_t = t if self.model_per_timestep else 0
        model = self.models[model_t][lv][col]
        x_in = (
            x_t
            if self.model_per_timestep
            else np.column_stack([x_t, np.full(len(x_t), t)])
        )

        if col in self.numerical_features_set:
            pred = model.predict(x_in)
            if self.objective == "x":
                out = pred
            else:
                alpha = np.sqrt(self.alpha_bars_[t])
                sigma = np.sqrt(max(1.0 - self.alpha_bars_[t], 1e-12))
                if self.objective == "epsilon":
                    out = (x_t[:, j] - sigma * pred) / alpha
                else:
                    out = alpha * x_t[:, j] - sigma * pred

        else:
            le = self.label_encoders[model_t][lv][col]
            cls_ids = le.classes_.astype(np.int64)

            # If only one class was seen in training for this (t, lv, col), output it deterministically.
            if cls_ids.shape[0] == 1:
                out = cls_ids[0]
            else:
                proba = model.predict_proba(x_in)
                proba = proba[:, : cls_ids.shape[0]]
                proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)

                u = self.rng.random(size=len(x_t))
                cdf = np.cumsum(proba, axis=1)
                cdf[:, -1] = 1.0  # ensure never fall off the edge
                pick = (cdf < u[:, None]).sum(axis=1)
                out = cls_ids[pick]
        return j, out

    def _scale(self, X: pd.DataFrame):

        if len(self.num_features_x) > 0:
            self.qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(min(len(X) // 30, 1000), 10),
                subsample=int(1e9),
                random_state=self.random_state,
            )
            X[self.num_features_x] = self.qt.fit_transform(X[self.num_features_x])
            self.st_scaler = StandardScaler()
            X[self.num_features_x] = self.st_scaler.fit_transform(
                X[self.num_features_x]
            )

        else:
            self.qt, self.st_scaler = None, None
        return X

    def _inverse_scale(self, X: pd.DataFrame):
        if self.st_scaler is not None:
            X[self.num_features_x] = self.st_scaler.inverse_transform(
                X[self.num_features_x]
            )
        if self.qt is not None:
            X[self.num_features_x] = self.qt.inverse_transform(X[self.num_features_x])

        return X
