# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
import copy
import pickle
import shutil
from pathlib import Path

import pandas as pd

from ..base import BaseGenerator
from .model import ForestModel

FOREST_MODEL_DIR = "forest_model"


class ForestDiffusionGenerator(BaseGenerator):
    """ForestDiffusion: an XGBoost-based diffusion model.

    Based on a more memory-efficient implementation of ForestDiffusion: https://github.com/layer6ai-labs/forest-diffusion-mo

    Paper: "Scaling up diffusion and flow-based XGBoost models" by Cresswell et al. (2024).

    Original ForestDiffusion paper: "Generating and imputing tabular data via diffusion and flow-based gradient-boosted trees" by Jolicoeur-Martineau et al. (2024).


    Args:
        target_column: The column to use as the target column. Default: None.
        logdir: The directory to write/read XGBoost models. When None,
            trained XGBoost models are kept in memory. Default: None.
        num_timesteps: The number of noise levels to use. Default: 50.
        diffusion_type: The type of diffusion to use. Default: "vp".
        multi_output: Whether to use multi-output XGBoost. Default: False.
        xgb_hypers: The hyperparameters for the XGBoost models. Default: {"n_estimators": 100, "max_depth": 6, "early_stopping_rounds": 50}.
        noise_samples_per_row: The number of different noise samples per real data sample. Default: 100.
        eps: The timestep to stop generation at. Default: 0.001.
        beta_min: The minimum beta for the diffusion. Default: 0.1.
        beta_max: The maximum beta for the diffusion. Default: 8.
        n_jobs: The number of parallel jobs to use. Default: -1.
        backend: The joblib backend to use. Default: "loky".
        n_batch: The number of batches to use in XGBoost data iterator. Set to <=0 to disable data iterator. Default: -1.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import ForestDiffusionGenerator
        >>>
        >>> # Load data and define categorical columns
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator
        >>> generator = ForestDiffusionGenerator(
        ...     target_column="target",
        ...     random_state=42,
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "forestdiffusion"

    def __init__(
        self,
        target_column: str = None,
        logdir: str = None,
        num_timesteps: int = 50,
        diffusion_type: str = "vp",
        multi_output: bool = False,
        xgb_hypers: dict = None,
        noise_samples_per_row: int = 100,
        eps: float = 0.001,
        beta_min: float = 0.1,
        beta_max: float = 8,
        n_jobs: int = -1,
        backend: str = "loky",
        n_batch: int = -1,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        self.target_column = target_column
        self.logdir = logdir
        self.num_timesteps = num_timesteps
        self.diffusion_type = diffusion_type
        self.multi_output = multi_output
        if xgb_hypers is None:
            xgb_hypers = {
                "n_estimators": 100,
                "max_depth": 6,
                "early_stopping_rounds": 50,
            }
        self.xgb_hypers = xgb_hypers.copy()
        self.noise_samples_per_row = noise_samples_per_row
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_jobs = n_jobs
        self.backend = backend
        self.n_batch = n_batch
        self.models_to_disk = logdir is not None
        self.model = None
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        if self.target_column is not None and self.target_column not in X.columns:
            raise ValueError(
                f"target_column {self.target_column!r} is not present in X."
            )

        self.output_columns = list(X.columns)
        self.conditional_generation = (
            self.target_column is not None and self.target_column in discrete_features
        )
        y_train = X[self.target_column] if self.conditional_generation else None
        X_train = (
            X.drop(columns=[self.target_column]).copy()
            if self.conditional_generation
            else X.copy()
        )
        self.feature_columns = list(X_train.columns)
        self.discrete_features = list(discrete_features)
        feature_discrete_features = [
            col for col in discrete_features if col in X_train.columns
        ]
        feature_bin_features = [
            col for col in feature_discrete_features if X_train[col].nunique() == 2
        ]
        feature_discrete_features = [
            col for col in feature_discrete_features if col not in feature_bin_features
        ]
        cat_indexes = [
            i
            for i, col in enumerate(X_train.columns)
            if col in feature_discrete_features
        ]
        bin_indexes = [
            i for i, col in enumerate(X_train.columns) if col in feature_bin_features
        ]

        self.model = ForestModel(
            logdir=self.logdir,
            n_t=self.num_timesteps,  # number of noise levels
            diffusion_type=self.diffusion_type,  # vp or flow
            multi_output=self.multi_output,  # True for multi-output XGB ensembles, otherwise uses single-output ensembles
            xgb_hypers=self.xgb_hypers.copy(),
            duplicate_K=self.noise_samples_per_row,  # number of different noise samples per real data sample
            cat_indexes=cat_indexes,  # vector which indicates which column is categorical (>=3 categories)
            bin_indexes=bin_indexes,  # vector which indicates which column is binary
            int_indexes=[],  # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
            true_min_max_values=None,  # List of form [[min_x, min_y], [max_x, max_y]]; If provided, we use these values as the min/max for each variables when using clipping
            eps=self.eps,  # timestep to stop generation at, often used with diffusion models which can explode as t->0.
            beta_min=self.beta_min,  # vp only
            beta_max=self.beta_max,  # vp only
            solver="euler",  # euler, heun, or rk4
            scaler="min_max",  # min_max creates one scaler per class. single_min_max creates one scaler overall.
            n_jobs=self.n_jobs,  # number of parallel jobs to create. xgb_hypers contains xgb_n_jobs, which is the number of cpus per job.
            backend=self.backend,  # joblib Parallel backend. Can be "loky", "multiprocessing", or "threading". We recommend not changing this.
            n_batch=self.n_batch,  # If >0, use data iterator with the specified number of batches when constructing QuantileDMatrix
            models_to_disk=self.models_to_disk,
            seed=self.random_state,
        )

        X_train = self.model.preprocess(
            X_train.to_numpy(),
            y_train.to_numpy() if y_train is not None else None,
        )
        self.model.train(X_train)

        return self

    def _generate(self, n: int):
        syn = self.model.generate(n=n, seed=self.random_state, n_jobs=self.n_jobs)

        columns = list(getattr(self, "feature_columns", []))
        if self.conditional_generation and getattr(self.model, "trained_with_y", False):
            columns.append(self.target_column)
        syn = pd.DataFrame(syn, columns=columns)
        return syn[self.output_columns]

    def _state(self):
        state = {
            "target_column": self.target_column,
            "conditional_generation": getattr(self, "conditional_generation", False),
            "logdir": self.logdir,
            "num_timesteps": self.num_timesteps,
            "diffusion_type": self.diffusion_type,
            "multi_output": self.multi_output,
            "xgb_hypers": self.xgb_hypers,
            "noise_samples_per_row": self.noise_samples_per_row,
            "eps": self.eps,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "n_jobs": self.n_jobs,
            "backend": self.backend,
            "n_batch": self.n_batch,
            "models_to_disk": self.models_to_disk,
        }
        for attr in ("output_columns", "feature_columns", "discrete_features"):
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    def _save_extra(self, path: Path) -> None:
        if self.model is None:
            return

        model_path = path / FOREST_MODEL_DIR
        train_dir = model_path / "train"
        source_logdir = (
            Path(self.model.logdir).resolve()
            if getattr(self.model, "logdir", None) is not None
            else None
        )
        saving_to_model_logdir = source_logdir == model_path.resolve()

        if model_path.exists() and not saving_to_model_logdir:
            shutil.rmtree(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        if getattr(self.model, "regr", None) is not None:
            train_dir.mkdir(parents=True, exist_ok=True)
            self._save_in_memory_xgb_models(train_dir)
        elif getattr(self.model, "models_to_disk", False):
            source_train_dir = Path(self.model.train_dir)
            if not source_train_dir.exists():
                raise RuntimeError(
                    "Cannot save ForestDiffusionGenerator because its XGBoost "
                    "model files are missing."
                )
            if source_train_dir.resolve() != train_dir.resolve():
                if train_dir.exists():
                    shutil.rmtree(train_dir)
                shutil.copytree(source_train_dir, train_dir)
        else:
            raise RuntimeError(
                "Cannot save ForestDiffusionGenerator because no fitted XGBoost "
                "models are available."
            )

        model_for_disk = copy.copy(self.model)
        if hasattr(model_for_disk, "regr"):
            model_for_disk.regr = None
        model_for_disk.models_to_disk = True
        model_for_disk.set_logdir(str(model_path))
        with (model_path / "forest_model.pkl").open("wb") as f:
            pickle.dump(model_for_disk, f, protocol=pickle.HIGHEST_PROTOCOL)
        (model_path / "TRAIN_CHECKPOINT.txt").write_text("Done\n")

    def _save_in_memory_xgb_models(self, train_dir: Path) -> None:
        for j, models_for_label in enumerate(self.model.regr):
            for i, booster in enumerate(models_for_label):
                if booster is None:
                    raise RuntimeError(
                        "Cannot save ForestDiffusionGenerator because an XGBoost "
                        "model is missing."
                    )
                booster.save_model(str(train_dir / f"model_{i}_{j}.ubj"))

    def _load_extra(self, path: Path) -> None:
        model_path = path / FOREST_MODEL_DIR
        if not model_path.exists():
            self.model = None
            return

        self.model = ForestModel.load_model(str(model_path))
        self.model.models_to_disk = True
        self.model.set_logdir(str(model_path))
        self.logdir = str(model_path)
        self.models_to_disk = True
