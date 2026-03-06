import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from typing import Union, Tuple
from tqdm import tqdm

ArrayLike = Union[np.ndarray, list, Tuple[float, ...]]

from .xgenboost import XGenBoost
from .utils import sample_from_posterior

from .eqf import EmpiricalInterpolatedQuantile


class XGB_AR_Generator(XGenBoost):
    """XGenBoost autoregressive generator.

    Trains a hierarchical autoregressive model where conditionals are learned by XGBoost classifiers.

    Args:
        target_column (str): Name of the target column.
        conditioning (str): Conditioning mode. Options: "generation", "inference". Default: "inference".
        xgboost_params (dict): Parameters passed to each underlying XGBoost model.
            Default: {"n_estimators": 30, "max_depth": 3, "max_bin": 256, "early_stopping_rounds": 20, "device": "cpu"}.
        use_early_stopping (bool): Whether to use validation-based early stopping when validation data is provided. Default: False.
        temperature (float): Sampling temperature for posterior sampling. Default: 1.0.
        discretization (str): Numerical discretization strategy. Default: "quantile".
        per_bin_sampling (str): Sampling method within numerical bins. Default: "eqf".
        cat_merge_type (str): Strategy for merging infrequent categories. Default: "clustering".
        cat_merge_n_infrequent (int): Number of infrequent category clusters to merge into. Default: 5.
        visit_order_method (str): Feature visit-order method. Default: "naive".
        visit_order_mode (str): Visit-order direction. Options: "ascending", "descending". Default: "ascending".
        random_state (int): Random seed for reproducibility. Default: 0.
        n_jobs_xgb (int): Number of threads used per XGBoost model. Default: 1.
        n_jobs (int): Number of parallel jobs used to train/sample across tasks. Default: -1.
        H (int): Meta-tree height for numerical features. The number of bins is ``2**H``. Default: 5.
        route_method (str): Numerical routing method. Options: "propagate", "routing". Default: "routing".
        start_method (str): Initialization method for the first feature. Options: "bootstrap", "eqf". Default: "bootstrap".
        **kwargs: Additional arguments passed to `TabularBaseGenerator`.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import XGB_AR_Generator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["target", "category_col"]
        >>>
        >>> # Create generator (requires target column)
        >>> generator = XGB_AR_Generator(
        ...     target_column="target",
        ...     H=5,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "xgenboost_ar"
    needs_target_column = True

    def __init__(
        self,
        target_column: str,
        conditioning: str = "inference",  # "generation", "inference"
        xgboost_params: dict = {
            "n_estimators": 30,
            "max_depth": 3,
            "max_bin": 256,
            "early_stopping_rounds": 20,
            "device": "cpu",
        },
        use_early_stopping: bool = False,
        temperature: float = 1.0,
        discretization: str = "quantile",
        per_bin_sampling: str = "eqf",
        cat_merge_type: str = "clustering",
        cat_merge_n_infrequent: int = 5,
        visit_order_method: str = "naive",
        visit_order_mode: str = "ascending",
        random_state: int = 0,
        n_jobs_xgb: int = 1,
        n_jobs: int = -1,
        H: int = 5,  # meta-tree height; n_bins = 2^H for continuous discretizers
        route_method: str = "routing",  # "propagate" or "routing"
        start_method: str = "bootstrap",  # "bootstrap" or "eqf"
        **kwargs,
    ) -> None:
        super().__init__(
            target_column=target_column,
            conditioning=conditioning,
            use_early_stopping=use_early_stopping,
            discretization=discretization,
            n_bins=2**H,
            per_bin_sampling=per_bin_sampling,
            cat_merge_type=cat_merge_type,
            cat_merge_n_infrequent=cat_merge_n_infrequent,
            random_state=random_state,
            **kwargs,
        )
        assert route_method in [
            "propagate",
            "routing",
        ], "route_methods must be either 'propagate' or 'routing'"
        self.__dict__.update(locals())

        assert start_method in [
            "bootstrap",
            "eqf",
        ], "start_method must be either 'bootstrap' or 'eqf'"

        device = self.xgboost_params.get("device", "cpu")
        self.xgboost_params.update(
            {
                "random_state": self.random_state,
                "nthread": self.n_jobs_xgb,
                "tree_method": "hist" if device == "cpu" else "gpu_hist",
            }
        )
        self.rng = check_random_state(self.random_state)

        self.models_ut = {}
        self.models_cat = {}
        self.feature_names = None
        self.feature_types = None

    # --------------------------
    # Training
    # --------------------------
    def _train_model(self, X, X_enc, val_X, val_X_enc):
        self.feature_names = X.columns.tolist()
        self.feature_types = [
            "c" if c in self.discrete_columns else "q" for c in self.feature_names
        ]

        x = X.to_numpy()
        x_enc = X_enc.to_numpy()

        if val_X is not None:
            val_x = val_X.to_numpy()
            val_x_enc = val_X_enc.to_numpy()
        else:
            val_x = None
            val_x_enc = None

        cols = self.feature_names

        # create a flat list of all tasks for maximum parallelism
        tasks = []
        for i in range(1, len(cols)):
            col = cols[i]
            if col in self.discrete_columns:
                tasks.append(("cat", i, col, None, None))
            else:
                for d in range(self.H):
                    for node in range(2**d):
                        tasks.append(("ut", i, col, d, node))

        # run all XGB classifiers in parallel
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._train_one_task)(
                kind=kind,
                i=i,
                col=col,
                d=d,
                node=node,
                x=x,
                x_enc=x_enc,
                feature_types=self.feature_types,
                H=self.H,
                val_x=val_x,
                val_x_enc=val_x_enc,
            )
            for (kind, i, col, d, node) in tqdm(tasks, desc="Training", leave=False)
        )

        # init output
        self.models_ut = {
            col: {d: {} for d in range(self.H)}
            for col in cols[1:]
            if col not in self.discrete_columns
        }
        self.models_cat = {}

        # save results
        for kind, col, d, node, clf in results:
            if kind == "cat":
                self.models_cat[col] = clf
            else:
                self.models_ut[col][d][node] = clf

    def _train_one_task(
        self,
        kind: str,
        i: int,
        col: str,
        d,
        node,
        x,
        x_enc,
        feature_types,
        H,
        val_x=None,
        val_x_enc=None,
    ):
        params = self.xgboost_params.copy()
        if kind == "cat":
            params.update({"num_class": int(len(self.label_encoders[col].classes_))})

        if kind == "cat":
            clf = self._train_multiclass(
                i=i,
                x=x,
                x_enc=x_enc,
                feature_types=feature_types,
                xgboost_params=params,
                val_x=val_x,
                val_x_enc=val_x_enc,
            )
            return ("cat", col, None, None, clf)

        clf = self._train_ut_node(
            i=i,
            d=d,
            node=node,
            x=x,
            x_enc=x_enc,
            feature_types=feature_types,
            xgboost_params=params,
            H=H,
            val_x=val_x,
            val_x_enc=val_x_enc,
        )
        return ("ut", col, d, node, clf)

    def _train_ut_node(
        self,
        i,
        d,
        node,
        x,
        x_enc,
        feature_types,
        xgboost_params,
        H,
        val_x=None,
        val_x_enc=None,
    ):

        y = x_enc[:, i]
        x_input = x[:, :i]

        span = 2 ** (H - d)
        half = span // 2
        start = node * span
        mid = start + half
        end = start + span

        idx = (y >= start) & (y < end)
        y_node = (y[idx] >= mid).astype(np.int32)
        x_node = x_input[idx]

        f_types = feature_types[:i]

        params = xgboost_params.copy()
        params.update({"feature_types": f_types})

        clf = xgb.XGBClassifier(**params)

        if val_x is not None and self.use_early_stopping:
            val_y = val_x_enc[:, i]
            val_x_input = val_x[:, :i]
            val_idx = (val_y >= start) & (val_y < end)
            val_y_node = (val_y[val_idx] >= mid).astype(np.int32)
            val_x_node = val_x_input[val_idx]

            clf.fit(x_node, y_node, eval_set=[(val_x_node, val_y_node)])
        else:
            clf.fit(x_node, y_node)

        return clf

    def _train_multiclass(
        self,
        i,
        x,
        x_enc,
        feature_types,
        xgboost_params,
        val_x=None,
        val_x_enc=None,
    ):
        y = x_enc[:, i].astype(np.int32)
        x_input = x[:, :i]

        f_types = feature_types[:i]

        params = xgboost_params.copy()
        params.update(
            {"feature_types": f_types, "objective": "multi:softprob"}
        )  # "feature_names": f_names,

        clf = xgb.XGBClassifier(**params)

        if val_x is not None and self.use_early_stopping:
            y_val = val_x_enc[:, i].astype(np.int32)
            x_val = val_x[:, :i]
            clf.fit(
                x_input,
                y,
                eval_set=[(x_val, y_val)],
            )
        else:
            clf.fit(x_input, y)

        return clf

    def _sample_data(self, n: int):
        syn = pd.DataFrame(index=range(n), columns=self.feature_names)

        if self.start_method == "bootstrap":
            syn[self.feature_names[0]] = (
                self.X[self.feature_names[0]]
                .sample(n=n, replace=True, random_state=self.rng)
                .to_numpy()
            )
        elif self.start_method == "eqf":
            if self.feature_names[0] in self.discrete_columns:
                syn[self.feature_names[0]] = (
                    self.X[self.feature_names[0]]
                    .sample(n=n, replace=True, random_state=self.rng)
                    .to_numpy()
                )
            else:
                eqf = EmpiricalInterpolatedQuantile(
                    n_knots=-1,  # use all training samples as knots
                    use_spline=False,  # whether to use monotonic cubic spline interpolation
                )
                eqf.fit(self.X[self.feature_names[0]].to_numpy())
                syn[self.feature_names[0]] = eqf.rvs(size=n, rng=self.rng)

        else:
            raise ValueError(f"Invalid start method: {self.start_method}")

        for i, col in enumerate(
            tqdm(self.feature_names[1:], desc="Sampling", leave=False), start=1
        ):
            x_input = syn[self.feature_names[:i]].to_numpy(copy=False)

            if col in self.discrete_columns:
                clf = self.models_cat[col]
                probs = clf.predict_proba(x_input)
            else:
                if self.route_method == "propagate":
                    probs = self._meta_tree_leaf_probs(col=col, x_input=x_input)
                else:  # "routing"
                    probs = self._meta_tree_leaf_probs_routing(col=col, x_input=x_input)

            # clip to the column’s label space and renormalize
            k = len(self.label_encoders[col].classes_)
            probs = probs[:, :k]
            row_sums = probs.sum(axis=1, keepdims=True)
            probs = np.divide(probs, np.maximum(row_sums, 1e-12))

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

    def _meta_tree_leaf_probs(self, col: str, x_input: np.ndarray) -> np.ndarray:
        """
        Compute p(leaf_bin | x_input) by multiplying node decisions along paths.
        Equivalent to computing the full leaf distribution.
        """
        n = x_input.shape[0]

        mass = np.ones((n, 1), dtype=np.float64)

        for d in range(self.H):
            n_nodes = 2**d
            next_mass = np.zeros((n, 2 ** (d + 1)), dtype=np.float64)

            for node in range(n_nodes):
                clf = self.models_ut[col][d][node]
                p_right = clf.predict_proba(x_input)[:, 1].astype(np.float64)

                m = mass[:, node]
                next_mass[:, 2 * node] = m * (1.0 - p_right)
                next_mass[:, 2 * node + 1] = m * p_right

            mass = next_mass

        return mass

    def _meta_tree_leaf_probs_routing(
        self, col: str, x_input: np.ndarray
    ) -> np.ndarray:
        """
        Stochastically route the meta-tree to obtain a bin.
        Retrieves the same leaf distribution in expectation.
        """
        n = x_input.shape[0]

        # current node index per sample
        node_idx = np.zeros(n, dtype=np.int32)

        for d in range(self.H):
            next_node_idx = np.empty_like(node_idx)

            for node in range(2**d):
                mask = node_idx == node
                if not np.any(mask):
                    continue

                clf = self.models_ut[col][d][node]
                p_right = clf.predict_proba(x_input[mask])[:, 1]

                # sample Bernoulli routing decisions
                go_right = self.rng.uniform(size=p_right.shape[0]) < p_right

                next_node_idx[mask] = 2 * node + go_right.astype(np.int32)

            node_idx = next_node_idx

        # convert routed leaves to one-hot mass
        mass = np.zeros((n, 2**self.H), dtype=np.float64)
        mass[np.arange(n), node_idx] = 1.0

        return mass
