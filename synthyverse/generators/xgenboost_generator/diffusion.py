import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Any, Dict, List, Optional, Tuple

from joblib import Parallel, delayed
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.utils import check_random_state
from tqdm import tqdm

from ..base import TabularBaseGenerator


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


class XGB_Diffusion_Generator(TabularBaseGenerator):
    """XGenBoost diffusion generator.

    Denoising Diffusion Probabilistic Model (DDPM) using XGBoost as score estimator.

    Args:
        target_column (str): Name of the target column.
        timesteps (int): Number of diffusion timesteps. Default: 50.
        noise_samples_per_row (int): Number of noise levels per row. Default: 100.
        n_jobs (int): Number of parallel jobs used across timesteps/features. Default: -1.
        n_jobs_xgb (int): Number of threads used per XGBoost model. Default: 1.
        beta_min (float): Minimum beta value for the variance-preserving schedule. Default: 0.1.
        beta_max (float): Maximum beta value for the variance-preserving schedule. Default: 8.0.
        eps (float): Lower bound for the diffusion time grid. Default: 0.0.
        xgboost_params (Optional[Dict[str, Any]]): Base parameters for XGBoost regressors/classifiers.
            Default: {"n_estimators": 100, "max_depth": 7, "reg_lambda": 0.0}.
        random_state (int): Random seed for reproducibility. Default: 0.
        clip_extremes (bool): Whether to clip synthesized numerical values to observed training min/max. Default: True.
        sampler (str): Numerical reverse sampler. Options: "ddpm", "ddim". Default: "ddpm".
        objective (str): Numerical prediction objective. Options: "x", "v". Default: "x".
        dropout (float): Feature dropout probability applied to numerical inputs during training. Default: 0.1.
        dropout_token (str): Token used when dropping numerical inputs. Options: "mean", "missing", "random". Default: "mean".
        **kwargs: Additional arguments passed to `TabularBaseGenerator`.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import XGB_Diffusion_Generator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator (requires target column)
        >>> generator = XGB_Diffusion_Generator(
        ...     target_column="target",
        ...     timesteps=50,
        ...     sampler="ddpm",
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "xgenboost_diffusion"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        timesteps: int = 50,
        noise_samples_per_row: int = 100,
        n_jobs: int = -1,
        n_jobs_xgb: int = 1,
        beta_min: float = 0.1,
        beta_max: float = 8.0,
        eps: float = 0.0,
        xgboost_params: Optional[Dict[str, Any]] = {
            "n_estimators": 100,
            "max_depth": 7,
            "reg_lambda": 0.0,
        },
        random_state: int = 0,
        clip_extremes: bool = True,
        sampler: str = "ddpm",  # ddpm or ddim
        objective: str = "x",  # x or v
        dropout: float = 0.1,  # dropout rate for numerical features
        dropout_token: str = "mean",  # mean or missing
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.objective = str(objective).lower()
        assert self.objective in ["x", "v"], "objective must be either x or v"

        self.dropout_token = str(dropout_token).lower()
        assert self.dropout_token in [
            "mean",
            "missing",
            "random",
        ], "dropout_token must be either mean, missing, or random"
        self.sampler = str(sampler).lower()
        assert self.sampler in ["ddpm", "ddim"], "sampler must be either ddpm or ddim"
        self.target_column = target_column
        self.timesteps = timesteps
        self.noise_samples_per_row = noise_samples_per_row
        self.n_jobs = n_jobs
        self.n_jobs_xgb = n_jobs_xgb
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

        self.random_state = random_state
        self.clip_extremes = clip_extremes
        self.xgboost_params = (
            xgboost_params.copy() if xgboost_params is not None else {}
        )
        self.xgboost_params.update(
            {
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs": n_jobs_xgb,
            }
        )

        num_samplers = {"ddpm": self._ddpm_update, "ddim": self._ddim_update}
        self.num_sampler = num_samplers[self.sampler]

        self.rng = check_random_state(self.random_state)

        self.dropout = dropout

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: List[str], X_val: pd.DataFrame = None
    ):
        # store original training column order for exact reconstruction
        self.ori_cols = X.columns.tolist()

        self.discrete_features = [x for x in X.columns if x in discrete_features]
        self.numerical_features = [c for c in X.columns if c not in discrete_features]

        # in none of the conditioning modes we retain discrete target in X
        self.disc_features_x = [
            col for col in self.discrete_features if col != self.target_column
        ]
        self.num_features_x = self.numerical_features
        self.is_conditional = self.target_column in self.discrete_features

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

        # build corrupted dataset (shape N,D,T)
        self.betas_, self.alphas_, self.alpha_bars_ = _vp_sched(
            self.timesteps, self.beta_min, self.beta_max, self.eps
        )

        x_corrupted = self._build_corrupted_dataset(X_tr)

        self.labels = X_tr[self.target_column]

        cols = self.num_features_x + self.disc_features_x
        cond = [None]

        if self.is_conditional:
            cond = X_tr[self.target_column].unique()
        else:
            cols = self.numerical_features + self.discrete_features

        tasks = (
            delayed(self._fit_one)(X_tr, x_corrupted, t, lv, col, cols)
            for t in tqdm(range(self.timesteps), desc="Fitting models")
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

    def _generate_data(self, n: int) -> pd.DataFrame:

        if self.is_conditional:
            lv_samples = self.labels.sample(n, replace=True, random_state=self.rng)
        else:
            lv_samples = np.zeros(n, dtype=np.int64)  # dummy for consistent logic

        cond = np.unique(lv_samples)
        syn = np.empty((n, len(self.num_features_x) + len(self.disc_features_x)))
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
                    for j, col in enumerate(self.num_features_x + self.disc_features_x)
                )
                res = Parallel(n_jobs=self.n_jobs, prefer="threads")(tasks)

                # build x0_hat
                x0_hat = np.zeros_like(x_t)
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
        idx = list(range(n))
        syn = pd.DataFrame(
            syn, index=idx, columns=self.num_features_x + self.disc_features_x
        )

        if self.is_conditional:
            y = pd.Series(lv_samples.to_numpy(), index=idx, name=self.target_column)
            syn = pd.concat([syn, y], axis=1)

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
        x_corrupted: np.ndarray,
        t: int,
        lv: Optional[int],
        col: str,
        cols: List[str],
    ):
        # get corrupted data at current timestep
        x_in = x_corrupted[lv][:, :, t] if lv is not None else x_corrupted[:, :, t]
        x_in = x_in.copy()

        disc_features = [x for x in self.discrete_features if x in cols]
        feat_types = ["c" if x in disc_features else "q" for x in cols]

        # apply dropout on numerical features
        x_in_v = x_in.copy()
        if self.dropout > 0.0 and len(self.num_features_x) > 0:
            mask = (
                self.rng.random(size=(len(x_in), len(self.num_features_x)))
                < self.dropout
            )
            if self.dropout_token == "random":
                for j in range(len(self.num_features_x)):
                    m = mask[:, j]
                    if not np.any(m):
                        continue

                    # randomly resample rows
                    s_idx = self.rng.randint(0, len(x_in_v), size=m.sum())
                    x_in[m, j] = x_in_v[s_idx, j]
            else:
                token = 0.0 if self.dropout_token == "mean" else np.nan
                x_in[:, : len(self.num_features_x)] = np.where(
                    mask, token, x_in[:, : len(self.num_features_x)]
                )

        # get target values, i.e., the original data values
        y_in = (
            X[col][X[self.target_column] == lv].values
            if lv is not None
            else X[col].values
        )
        # repeat to match input data shape
        y_in = np.repeat(y_in, self.noise_samples_per_row)

        # align target to v-prediction
        if (col in self.numerical_features) and (self.objective == "v"):
            j = cols.index(col)

            alpha = np.sqrt(self.alpha_bars_[t])
            sigma = np.sqrt(
                max(1.0 - self.alpha_bars_[t], 1e-12)
            )  # avoid divide-by-zero

            eps = (x_in_v[:, j] - alpha * y_in) / sigma
            y_in = alpha * eps - sigma * y_in

        params = self.xgboost_params.copy()
        params["feature_types"] = feat_types

        label_enc = None
        if col in self.numerical_features:
            model = xgb.XGBRegressor(**params)
        else:
            # for some timesteps and/or subsets there may not be an ordinal encoding so label encode
            label_enc = LabelEncoder()
            y_in = label_enc.fit_transform(y_in)
            model = xgb.XGBClassifier(**params)

        model.fit(x_in, y_in)
        if lv is None:
            lv = 0
        return model, label_enc, t, lv, col

    def _build_corrupted_dataset(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Build a corrupted dataset (combining Gaussian and multinomial diffusion) of shape N,D,T
        """

        self.n_cls_ = (
            self.n_cls.loc[self.disc_features_x].to_numpy(dtype=np.int64)
            if self.n_cls is not None
            else None
        )
        x_tr = X[[col for col in X.columns if col != self.target_column]]

        if self.is_conditional:
            # per-label corruption on X
            x_corrupted = {}
            for lv in X[self.target_column].unique():
                x_corrupted[lv] = self._corrupt_num_cat(
                    x_tr[X[self.target_column] == lv],
                    self.disc_features_x,
                    self.num_features_x,
                )
        else:
            # full corruption on X,y
            x_corrupted = self._corrupt_num_cat(
                X,
                self.disc_features_x,
                self.num_features_x,
            )

        return x_corrupted

    def _corrupt_num_cat(
        self,
        X: pd.DataFrame,
        discrete_features: List[str],
        numerical_features: List[str],
    ) -> np.ndarray:
        """
        Combine corrupted datasets for numerical and categorical features
        """
        x_num_corrupted = None
        x_cat_corrupted = None

        if len(numerical_features) > 0:
            x_num_corrupted = self._corrupt_num_full(
                X[numerical_features].to_numpy(copy=False)
            )
        if len(discrete_features) > 0:
            x_cat_corrupted = self._corrupt_cat_full(
                X[discrete_features].to_numpy(copy=False)
            )

        x = (
            np.concatenate([x_num_corrupted, x_cat_corrupted], axis=1)
            if (x_num_corrupted is not None and x_cat_corrupted is not None)
            else (x_num_corrupted if x_num_corrupted is not None else x_cat_corrupted)
        )
        return x

    def _corrupt_cat_full(self, x0: np.ndarray) -> np.ndarray:
        """
        Create a fully corrupted dataset (multinomial diffusion) of shape (N*K, D, T)
        """
        # extend dataset K times
        x0k = np.repeat(x0[:, :, None], self.noise_samples_per_row, axis=0)

        a_bar = self.alpha_bars_[None, None, :]
        keep = (
            self.rng.random(size=(x0k.shape[0], x0k.shape[1], self.timesteps)) < a_bar
        )

        rep = (
            self.rng.random(size=(x0k.shape[0], x0k.shape[1], self.timesteps))
            * self.n_cls_[None, :, None]
        ).astype(np.int64)

        x_corrupted = np.where(keep, x0k, rep)
        return x_corrupted.astype(np.int64)

    def _corrupt_num_full(self, x0: np.ndarray) -> np.ndarray:
        """
        Create a fully corrupted dataset (Gaussian diffusion) of shape (N*K, D, T)
        """
        x0k = np.repeat(x0[:, :, None], self.noise_samples_per_row, axis=0)
        a_bar = self.alpha_bars_[None, None, :]
        z = self.rng.normal(size=(x0k.shape[0], x0k.shape[1], self.timesteps))
        x_corrupted = np.sqrt(a_bar) * x0k + np.sqrt(1.0 - a_bar) * z
        return x_corrupted

    def _init_noise(self, n: int, d_num: int, d_disc: int):

        if d_num > 0:
            x_t_num = self.rng.normal(size=(n, d_num))

        if d_disc > 0:
            x_t_cat = np.vstack(
                [
                    self.rng.randint(0, int(self.n_cls_[j]), size=n)
                    for j in range(d_disc)
                ]
            ).T.astype(np.int64)

        x_t = (
            np.concatenate([x_t_num, x_t_cat], axis=1)
            if (d_num > 0 and d_disc > 0)
            else x_t_num if d_num > 0 else x_t_cat
        )
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
        for j in range(len(self.disc_features_x)):
            jj = len(self.num_features_x) + j
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
          q(x_{t-1}=k | x_t, x0) ∝ q(x_t | x_{t-1}=k) * q(x_{t-1}=k | x0)
        """

        base_prior = (1.0 - self.alpha_bars_[t - 1]) / K
        base_lik = (1.0 - self.alphas_[t]) / K

        N = xt.shape[0]
        probs = np.full((N, K), base_prior * base_lik, dtype=np.float64)

        probs[np.arange(N), xt] += base_prior * self.alphas_[t]
        probs[np.arange(N), x0_hat] += base_lik * self.alpha_bars_[t - 1]

        same = xt == x0_hat
        if np.any(same):
            idx = np.where(same)[0]
            probs[idx, xt[idx]] += self.alpha_bars_[t - 1] * self.alphas_[t]

        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(probs_sum, 1e-12, None)

        u = self.rng.random(size=N)
        cdf = np.cumsum(probs, axis=1)
        x_prev = (cdf < u[:, None]).sum(axis=1).astype(np.int64)
        return x_prev

    def _inference_one_col(self, x_t, j, col, t, lv):
        # grab model
        model = self.models[t][lv][col]

        if col in self.numerical_features:
            pred = model.predict(x_t)
            if self.objective == "x":
                out = pred
            elif self.objective == "v":
                alpha = np.sqrt(self.alpha_bars_[t])
                sigma = np.sqrt(max(1.0 - self.alpha_bars_[t], 1e-12))
                out = alpha * x_t[:, j] - sigma * pred

        else:
            le = self.label_encoders[t][lv][col]
            cls_ids = le.classes_.astype(np.int64)

            # If only one class was seen in training for this (t, lv, col), output it deterministically.
            if cls_ids.shape[0] == 1:
                out = cls_ids[0]
            else:
                proba = model.predict_proba(x_t)
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
                n_quantiles=min(2000, len(X)),
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
