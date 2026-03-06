import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from typing import Union, Tuple
from tqdm import tqdm

from .xgenboost import XGenBoost
from .utils import sample_from_posterior

from .eqf import EmpiricalInterpolatedQuantile


ArrayLike = Union[np.ndarray, list, Tuple[float, ...]]


class XGB_MC_Generator(XGenBoost):
    name = "xgenboost_multiclass"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        conditioning: str = "inference",  # "generation", "inference"
        xgboost_params: dict = {
            "n_estimators": 100,
            "max_depth": 6,
            "max_bin": 256,
            "early_stopping_rounds": 20,
            "device": "cpu",
        },
        use_early_stopping: bool = True,
        temperature: float = 1.0,
        discretization: str = "quantile",  # uniform, quantile, kmeans
        n_bins: int = 30,
        per_bin_sampling: str = "bootstrap",
        cat_merge_type: str = "clustering",
        cat_merge_n_infrequent: int = 5,
        visit_order_method: str = "centrality",
        visit_order_mode: str = "ascending",
        random_state: int = 0,
        n_jobs_xgb: int = 1,
        n_jobs=-1,
        start_method: str = "bootstrap",
        **kwargs,
    ) -> None:
        super().__init__(
            target_column=target_column,
            conditioning=conditioning,
            use_early_stopping=use_early_stopping,
            discretization=discretization,
            n_bins=n_bins,
            per_bin_sampling=per_bin_sampling,
            cat_merge_type=cat_merge_type,
            cat_merge_n_infrequent=cat_merge_n_infrequent,
            random_state=random_state,
            **kwargs,
        )
        self.__dict__.update(locals())

        device = self.xgboost_params.get("device", "cpu")
        self.xgboost_params.update(
            {
                "objective": "multi:softprob",
                "random_state": self.random_state,
                "n_jobs": self.n_jobs_xgb,  # sklearn param name
                "tree_method": "hist" if device == "cpu" else "gpu_hist",
                "enable_categorical": True,
            }
        )
        self.rng = check_random_state(self.random_state)

        assert start_method in [
            "bootstrap",
            "eqf",
        ], "start_method must be either 'bootstrap' or 'eqf'"
        self.start_method = start_method

    def _train_model(self, X, X_enc, val_X, val_X_enc):
        self.feature_names = X.columns.tolist()

        # feature_types aligned with *raw X* columns
        self.feature_types = [
            "c" if c in self.discrete_columns else "q" for c in self.feature_names
        ]

        # Use numpy arrays for training/prediction
        x = X.to_numpy()
        x_enc = X_enc.to_numpy()

        if val_X is not None:
            val_x = val_X.to_numpy()
            val_x_enc = val_X_enc.to_numpy()
        else:
            val_x = None
            val_x_enc = None

        # num_class: max across label encoders (same as before)
        self.xgboost_params["num_class"] = int(
            max(len(le.classes_) for le in self.label_encoders.values())
        )

        cols = self.feature_names
        n_cols = len(cols)

        def _fit_one(i: int, col: str):
            if i == 0:
                return col, None

            y = x_enc[:, i]
            x_input = x[:, :i]
            f_types = self.feature_types[:i]

            params = dict(self.xgboost_params)
            params.update({"feature_types": f_types})  # "feature_names": f_names,

            # If no valid early stopping setup, remove it for this model
            use_es = (val_x is not None) and self.use_early_stopping
            if not use_es:
                params.pop("early_stopping_rounds", None)

            model = xgb.XGBClassifier(**params)

            if use_es:
                x_val = val_x[:, :i]
                y_val = val_x_enc[:, i]
                model.fit(
                    x_input,
                    y,
                    eval_set=[(x_val, y_val)],
                    verbose=False,
                )
            else:
                model.fit(
                    x_input,
                    y,
                    verbose=False,
                )

            payload = {
                "model": model,
                "f_types": f_types,
            }
            return col, payload

        with tqdm(total=n_cols - 1, desc="Training models") as pbar:

            def _wrapped(i, col):
                out = _fit_one(i, col)
                if i != 0:
                    pbar.update(1)
                return out

            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_wrapped)(i, col) for i, col in enumerate(cols)
            )

        self.models = {}
        for col, payload in results:
            if payload is None:
                continue
            self.models[col] = payload

    def _sample_data(self, n: int):
        syn = pd.DataFrame(index=range(n), columns=[self.feature_names[0]])

        for i, col in enumerate(self.feature_names):
            curr_X = syn.copy()

            if i == 0:
                if self.start_method == "bootstrap":
                    syn[col] = (
                        self.X[col]
                        .sample(n=n, replace=True, random_state=self.rng)
                        .to_numpy()
                    )
                elif self.start_method == "eqf":
                    if col in self.discrete_columns:
                        syn[col] = (
                            self.X[col]
                            .sample(n=n, replace=True, random_state=self.rng)
                            .to_numpy()
                        )
                    else:
                        eqf = EmpiricalInterpolatedQuantile(
                            n_knots=-1,  # use all training samples as knots
                            use_spline=False,  # whether to use monotonic cubic spline interpolation
                        )
                        eqf.fit(self.X[col].to_numpy())
                        syn[col] = eqf.rvs(size=n, rng=self.rng)
            else:
                x_input = curr_X.to_numpy()[:, :i]

                probs_all = self.models[col]["model"].predict_proba(x_input)

                # clip to relevant class labels
                classes = np.arange(len(self.label_encoders[col].classes_))
                probs = probs_all[:, : len(classes)]

                syn[col] = sample_from_posterior(
                    probs,
                    col,
                    n,
                    self.temperature,
                    self.discrete_columns,
                    self.rng,
                    self.per_bin_sampling,
                    self.label_encoders,
                    self.discretizers,
                    self.repo,
                )
        return syn
