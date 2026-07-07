from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from xgb_diffusion import XGBDiffusionRegressor

from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.utils import check_random_state
from tqdm import tqdm

from ..base import BaseGenerator


def _vp_alpha_bars(times: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
    return np.exp(-0.5 * times**2 * (beta_max - beta_min) - times * beta_min)


class XGBDiffusionGenerator(BaseGenerator):
    """Diffusion model using XGBoost as backbone.

    Similar to ForestDiffusion, but allows marginalizing over noise seeds
    across boosting rounds to avoid massively extending the training set.

    Supports flow-matching and VP diffusion.

    Args:
        target_column (str, optional): Column used for conditional generation
            when it is categorical and ``model_per_label=True``. Default: None.
        num_timesteps (int): Number of diffusion timesteps for training and
            sampling. Default: 50.
        refresh_every_k (int): Number of boosting rounds between noise seed
            refreshes. Default: 1.
        noise_samples_per_row (int): Number of times to extend the training set,
            to marginalize over noise seeds. Default: 1.
        n_jobs (int): Number of parallel jobs used to fit column models. Default:
            -1.
        n_jobs_xgb (int): Number of threads used inside each XGBoost estimator.
            Default: 1.
        diffusion_type (str): Sampling objective. Options: "flow" for
            flow-matching sampling and "vp" for variance-preserving diffusion
            sampling. Default: "flow".
        beta_min (float): Minimum beta value for the variance-preserving noise
            schedule. Default: 0.1.
        beta_max (float): Maximum beta value for the variance-preserving noise
            schedule. Default: 8.0.
        eps (float): Lower endpoint of the diffusion time grid. Default: 1e-3.
        xgboost_params (dict, optional): Parameters passed to each
            diffusion-enabled XGBoost regressor. Default: ``{"n_estimators": 500,
            "max_depth": 6, "early_stopping_rounds": 20,
            "min_boosting_round": 50, "eta": 0.06}``.
        clip_extremes (bool): Whether to clip generated values to the encoded
            training data range before inverse transforming. Default: True.
        model_per_timestep (bool): Whether to train separate models per diffusion
            timestep. If False, timestep is appended as an input feature. Default:
            True.
        model_per_label (bool): Whether to train separate models per categorical
            ``target_column`` value. Default: True.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional keyword arguments accepted for API compatibility.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import XGBDiffusionGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator
        >>> generator = XGBDiffusionGenerator(
        ...     target_column="target",
        ...     diffusion_type="flow",
        ...     num_timesteps=50,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "xgbdiffusion"

    def __init__(
        self,
        target_column: Optional[str] = None,
        num_timesteps: int = 50,
        refresh_every_k: int = 1,
        noise_samples_per_row: int = 1,
        n_jobs: int = -1,
        n_jobs_xgb: int = 1,
        diffusion_type: str = "flow",
        beta_min: float = 0.1,
        beta_max: float = 8.0,
        eps: float = 1e-3,
        xgboost_params: Optional[Dict[str, Any]] = {
            "n_estimators": 500,
            "max_depth": 6,
            "early_stopping_rounds": 20,
            "min_boosting_round": 50,
            "eta": 0.06,
        },
        clip_extremes: bool = True,
        model_per_timestep: bool = True,
        model_per_label: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        self.diffusion_type = str(diffusion_type).lower()
        assert self.diffusion_type in (
            "flow",
            "vp",
        ), "diffusion_type must be flow or vp"

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
        self.rng = check_random_state(self.random_state)

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.ori_cols = X.columns.tolist()
        self.discrete_features = [x for x in X.columns if x in discrete_features]
        self.numerical_features = [c for c in X.columns if c not in discrete_features]
        self.is_conditional = (
            self.target_column in self.discrete_features and self.model_per_label
        )
        self.features_x = [
            c for c in X.columns if c != self.target_column or not self.is_conditional
        ]

        X_tr = X.copy()
        if self.discrete_features:
            self.ord_enc = OrdinalEncoder()
            X_tr[self.discrete_features] = self.ord_enc.fit_transform(
                X_tr[self.discrete_features]
            ).astype(int)
        self.n_cls = (
            X_tr[self.discrete_features].max(axis=0) + 1
            if self.discrete_features
            else None
        )

        self.disc_features_x = [
            c for c in self.features_x if c in self.discrete_features
        ]
        self.num_features_x = [
            c for c in self.features_x if c not in self.discrete_features
        ]
        self.cat_features_x = [
            c for c in self.disc_features_x if int(self.n_cls[c]) > 2
        ]
        self.int_features_x = [
            c for c in self.disc_features_x if int(self.n_cls[c]) <= 2
        ]

        self.X_min = X_tr[self.features_x].min(axis=0)
        self.X_max = X_tr[self.features_x].max(axis=0)
        self.labels = X_tr[self.target_column] if self.is_conditional else None

        X_model = self._dummify(X_tr[self.features_x])
        self.model_cols = X_model.columns.tolist()
        self.model_col_idx = {col: i for i, col in enumerate(self.model_cols)}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X_model[self.model_cols] = self.scaler.fit_transform(X_model[self.model_cols])

        self.times_ = np.linspace(self.eps, 1.0, self.timesteps)
        self.alpha_bars_ = _vp_alpha_bars(self.times_, self.beta_min, self.beta_max)
        self.vp_sample_ts_ = np.linspace(1.0, self.eps, self.timesteps)
        self.vp_sample_hs_ = self.vp_sample_ts_ - np.append(self.vp_sample_ts_, 0)[1:]
        self.eps_sigma_ = np.sqrt(max(1.0 - self.alpha_bars_[0], 1e-12))

        cond = X_tr[self.target_column].unique() if self.is_conditional else [0]
        fit_timesteps = range(self.timesteps) if self.model_per_timestep else [None]
        tasks = (
            delayed(self._fit_one)(X_model, t, lv, col)
            for t in tqdm(fit_timesteps, desc="Fitting models")
            for lv in cond
            for col in self.model_cols
        )
        res = Parallel(n_jobs=self.n_jobs, prefer="threads")(tasks)

        self.models = {}
        for model, t, lv, col in res:
            self.models.setdefault(t, {})
            self.models[t].setdefault(lv, {})
            self.models[t][lv][col] = model

    def _generate(self, n: int):
        if self.is_conditional:
            lv_samples = self.labels.sample(n, replace=True, random_state=self.rng)
        else:
            lv_samples = np.zeros(n, dtype=np.int64)

        syn = np.empty((n, len(self.model_cols)))
        for lv in np.unique(lv_samples):
            idx = np.where(lv_samples == lv)[0]
            x = self.rng.normal(size=(len(idx), len(self.model_cols)))
            syn[idx, :] = (
                self._sample_flow(x, lv)
                if self.diffusion_type == "flow"
                else self._sample_vp(x, lv)
            )

        syn = pd.DataFrame(syn, columns=self.model_cols)
        syn[self.model_cols] = self.scaler.inverse_transform(syn[self.model_cols])
        syn = self._clean_onehot_data(syn)
        syn = self._clip_extremes(syn)

        if self.is_conditional:
            syn[self.target_column] = lv_samples.to_numpy()

        if self.discrete_features:
            syn[self.discrete_features] = self.ord_enc.inverse_transform(
                syn[self.discrete_features].astype(float)
            )

        return syn[self.ori_cols]

    def _fit_one(self, X: pd.DataFrame, t: Optional[int], lv: int, col: str):
        mask = self.labels == lv if self.is_conditional else slice(None)
        x_in = X.loc[mask, self.model_cols].to_numpy(dtype=np.float32, copy=True)
        y_in = X.loc[mask, col].to_numpy(dtype=np.float32, copy=True)

        params = self.xgboost_params.copy()
        params.update(
            {
                "alpha_bars": self.alpha_bars_,
                "times": self.times_,
                "noise_samples_per_row": self.noise_samples_per_row,
                "timestep": t,
                "target_index": self.model_col_idx[col],
                "diffusion_type": self.diffusion_type,
                "feature_types": ["q"] * (len(self.model_cols) + int(t is None)),
            }
        )

        model = XGBDiffusionRegressor(**params)
        fit_kwargs = {"verbose": False}
        if params.get("early_stopping_rounds"):
            fit_kwargs["eval_set"] = [(x_in, y_in)]
        model.fit(x_in, y_in, **fit_kwargs)
        return model, 0 if t is None else t, lv, col

    def _sample_flow(self, x: np.ndarray, lv: int) -> np.ndarray:
        h = 1.0 / (self.timesteps - 1)
        t = 0.0
        for _ in range(self.timesteps - 1):
            x = x + h * self._model(t, x, lv)
            t += h
        return x

    def _sample_vp(self, x: np.ndarray, lv: int) -> np.ndarray:
        for i in range(self.timesteps - 1):
            t = self.vp_sample_ts_[i]
            beta = self.beta_min + t * (self.beta_max - self.beta_min)
            drift = -0.5 * beta * x - beta * self._model(t, x, lv)
            x_mean = x - drift * self.vp_sample_hs_[i]
            x = x_mean + np.sqrt(beta * self.vp_sample_hs_[i]) * self.rng.normal(
                size=x.shape
            )
        return x + self.eps_sigma_**2 * self._model(self.eps, x, lv)

    def _model(self, t: float, x: np.ndarray, lv: int) -> np.ndarray:
        step = int(round(t * (self.timesteps - 1)))
        step = min(max(step, 0), self.timesteps - 1)
        model_t = step if self.model_per_timestep else 0
        x_in = (
            x if self.model_per_timestep else np.column_stack([x, np.full(len(x), t)])
        )
        out = np.zeros_like(x)

        for j, col in enumerate(self.model_cols):
            out[:, j] = self.models[model_t][lv][col].predict(x_in)

        if self.diffusion_type == "vp":
            alpha_bar = _vp_alpha_bars(
                np.array([max(t, self.eps)]), self.beta_min, self.beta_max
            )[0]
            out = -out / np.sqrt(max(1.0 - alpha_bar, 1e-12))
        return out

    def _dummify(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.dummy_cols = {}
        if self.cat_features_x:
            X = pd.get_dummies(
                X,
                columns=self.cat_features_x,
                prefix=self.cat_features_x,
                dtype=float,
                drop_first=True,
            )
        for col in self.cat_features_x:
            self.dummy_cols[col] = [c for c in X.columns if c.startswith(f"{col}_")]
        return X

    def _clean_onehot_data(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        drop_cols = []
        for col in self.cat_features_x:
            dummy_cols = self.dummy_cols[col]
            vals = X[dummy_cols].to_numpy()
            X[col] = np.argmax(np.column_stack([np.full(len(X), 0.5), vals]), axis=1)
            drop_cols.extend(dummy_cols)
        if drop_cols:
            X = X.drop(columns=drop_cols)
        return X[self.features_x]

    def _clip_extremes(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.disc_features_x:
            X[self.disc_features_x] = X[self.disc_features_x].round()
        if self.clip_extremes:
            X[self.features_x] = np.clip(
                X[self.features_x],
                self.X_min.to_numpy()[None, :],
                self.X_max.to_numpy()[None, :],
            )
        return X
