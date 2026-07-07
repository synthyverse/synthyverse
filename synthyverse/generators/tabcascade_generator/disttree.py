from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.preprocessing import OrdinalEncoder


class GaussianFamily:
    npar = 2

    def start_params(self, y, weights=None):
        if weights is None:
            mu = float(np.mean(y))
            sigma = float(np.std(y, ddof=1)) or 1.0
        else:
            w = weights / weights.sum()
            mu = float(np.dot(w, y))
            sigma = float(np.sqrt(np.dot(w, (y - mu) ** 2))) or 1.0
        return np.array([mu, np.log(sigma)])

    def inverse_link(self, eta):
        return np.asarray(eta, dtype=float).copy()

    def link_bounds(self):
        return [(None, None), (-100.0, None)]

    def log_likelihood(self, y, params, weights=None):
        mu, log_sigma = float(params[0]), float(params[1])
        weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        ll = (
            -0.5 * np.log(2.0 * np.pi)
            - log_sigma
            - 0.5 * (y - mu) ** 2 * np.exp(-2.0 * log_sigma)
        )
        return float(np.dot(weights, ll))

    def score(self, y, params, weights=None):
        mu, log_sigma = float(params[0]), float(params[1])
        weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        residual = y - mu
        exp_neg2 = np.exp(-2.0 * log_sigma)
        score = np.column_stack(
            [residual * exp_neg2, residual**2 * exp_neg2 - 1.0]
        )
        return score * weights[:, np.newaxis]

    def hessian(self, y, params, weights=None):
        mu, log_sigma = float(params[0]), float(params[1])
        weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        residual = y - mu
        exp_neg2 = np.exp(-2.0 * log_sigma)
        return np.array(
            [
                [-weights.sum() * exp_neg2, -2.0 * np.dot(weights, residual) * exp_neg2],
                [
                    -2.0 * np.dot(weights, residual) * exp_neg2,
                    -2.0 * np.dot(weights, residual**2) * exp_neg2,
                ],
            ]
        )

    def to_user_params(self, params):
        return np.array([float(params[0]), np.exp(float(params[1]))]), ["mu", "sigma"]

    def mean(self, params):
        return float(params[0])


class DistFit:
    def __init__(self, family=None):
        self.family = family or GaussianFamily()

    def fit(self, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        eta0 = self.family.start_params(y, weights)

        def neg_loglik(eta):
            return -self.family.log_likelihood(y, self.family.inverse_link(eta), weights)

        def neg_score_eta(eta):
            params = self.family.inverse_link(eta)
            return -self.family.score(y, params, weights).sum(axis=0)

        result = minimize(
            neg_loglik,
            eta0,
            jac=neg_score_eta,
            method="L-BFGS-B",
            bounds=self.family.link_bounds(),
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
        params = self.family.inverse_link(result.x)
        display, names = self.family.to_user_params(params)
        self.coef_ = {name: float(value) for name, value in zip(names, display)}
        self.coef_array_ = display.copy()
        self.loglik_ = float(-result.fun)
        self.score_matrix_ = self.family.score(y, params, weights)
        self.hessian_ = self.family.hessian(y, params, weights)
        self.estfun_ = self.score_matrix_
        return self


def _quadratic_stat(t, mu, sigma):
    diff = t - mu
    eigvals, eigvecs = np.linalg.eigh(sigma)
    tol = max(1e-14, 1e-10 * float(np.abs(eigvals).max() or 1.0))
    pos = eigvals > tol
    if not pos.any():
        return 0.0, 0
    sigma_pinv = eigvecs[:, pos] @ np.diag(1.0 / eigvals[pos]) @ eigvecs[:, pos].T
    return float(diff @ sigma_pinv @ diff), int(pos.sum())


def select_variable(X, distfit, alpha=0.05):
    h = distfit.estfun_
    n, p_vars = X.shape
    p_values = np.ones(p_vars)
    for j in range(p_vars):
        g = X[:, j]
        h_bar = h.mean(axis=0)
        h_c = h - h_bar
        v_h = (h_c.T @ h_c) / n
        sigma_g = (n / (n - 1)) * float(g @ g) - (1 / (n - 1)) * float(g.sum()) ** 2
        stat, df = _quadratic_stat(g @ h, g.sum() * h_bar, sigma_g * v_h)
        p_values[j] = float(chi2.sf(stat, df=df)) if df > 0 else 1.0
    j_star = int(np.argmin(p_values))
    return j_star if min(p_vars * float(p_values[j_star]), 1.0) < alpha else None


def find_split(x, distfit, minbucket):
    h = distfit.estfun_
    n = h.shape[0]
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    h_sorted = h[order]
    h_bar = h.mean(axis=0)
    h_c = h - h_bar
    v_h = (h_c.T @ h_c) / n
    eigvals, eigvecs = np.linalg.eigh(v_h)
    tol = max(1e-14, 1e-10 * float(np.abs(eigvals).max() or 1.0))
    pos = eigvals > tol
    if not pos.any():
        return None
    v_h_pinv = eigvecs[:, pos] @ np.diag(1.0 / eigvals[pos]) @ eigvecs[:, pos].T
    h_cumsum = np.cumsum(h_sorted, axis=0)
    best_stat = -np.inf
    best_k = None
    for k in range(minbucket, n - minbucket + 1):
        if x_sorted[k - 1] == x_sorted[k]:
            continue
        diff = h_cumsum[k - 1] - k * h_bar
        sigma_g = k * (n - k) / (n - 1)
        stat = float(diff @ v_h_pinv @ diff) / sigma_g
        if stat > best_stat:
            best_stat = stat
            best_k = k
    return None if best_k is None else float(x_sorted[best_k - 1])


@dataclass
class Node:
    node_id: int
    depth: int
    n_samples: int
    distfit: DistFit
    is_leaf: bool = True
    split_var: int = None
    split_val: float = None
    left: "Node" = None
    right: "Node" = None


class DistTreeRegressor:
    def __init__(self, max_depth=None, alpha=0.05, minprob=0.01):
        self.family = GaussianFamily()
        self.max_depth = max_depth
        self.alpha = alpha
        self.minprob = minprob

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._node_counter = 0
        self.tree_ = self._grow(np.arange(len(y)), X, y, 0, math.ceil(10 * self.family.npar))
        return self

    def _next_id(self):
        node_id = self._node_counter
        self._node_counter += 1
        return node_id

    def _grow(self, indices, X, y, depth, min_leaf):
        y_node = y[indices]
        if float(np.var(y_node)) < 1e-14:
            return Node(self._next_id(), depth, len(indices), DistFit(self.family).fit(y_node))

        distfit = DistFit(self.family).fit(y_node)
        node = Node(self._next_id(), depth, len(indices), distfit)
        if (
            not np.all(np.isfinite(distfit.estfun_))
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(indices) < min_leaf
        ):
            return node

        X_node = X[indices]
        split_var = select_variable(X_node, distfit, self.alpha)
        if split_var is None:
            return node
        threshold = find_split(
            X_node[:, split_var],
            distfit,
            max(min_leaf, math.ceil(self.minprob * len(indices))),
        )
        if threshold is None:
            return node

        left_mask = X_node[:, split_var] <= threshold
        node.is_leaf = False
        node.split_var = split_var
        node.split_val = threshold
        node.left = self._grow(indices[left_mask], X, y, depth + 1, min_leaf)
        node.right = self._grow(indices[~left_mask], X, y, depth + 1, min_leaf)
        return node

    def _apply_one(self, x):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.split_var] <= node.split_val else node.right
        return node

    def predict_params(self, X):
        leaves = [self._apply_one(row) for row in np.asarray(X, dtype=float)]
        return {
            "mu": np.array([leaf.distfit.coef_["mu"] for leaf in leaves]),
            "sigma": np.array([leaf.distfit.coef_["sigma"] for leaf in leaves]),
        }


class DistTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.models = []

    def fit(self, x):
        self.ord_enc = []
        self.ord_enc_joint = []
        self.means = []
        self.stds = []
        for i in range(x.shape[1]):
            d = x[:, i].clone()
            df = pd.DataFrame(d, columns=["x"]).dropna()
            model = DistTreeRegressor(max_depth=self.max_depth)
            model.fit(df["x"].to_frame(), df["x"])
            self.models.append(model)
            preds = pd.DataFrame(model.predict_params(df["x"].to_frame()), columns=["mu", "sigma"])
            ord_enc = OrdinalEncoder(dtype=int)
            df_enc = ord_enc.fit_transform(preds)
            self.ord_enc.append(ord_enc)
            joint_group = np.char.add(np.char.add(df_enc[:, 0].astype(str), "_"), df_enc[:, 1].astype(str))
            ord_enc_joint = OrdinalEncoder(dtype=int)
            joint_group = ord_enc_joint.fit_transform(joint_group.reshape(-1, 1))
            self.ord_enc_joint.append(ord_enc_joint)
            df_aux = pd.DataFrame(
                np.column_stack((preds.to_numpy(), joint_group)),
                columns=["mu", "sigma", "group"],
            )
            self.means.append(df_aux.drop_duplicates(["group"]).sort_values("group")["mu"].to_list())
            self.stds.append(df_aux.drop_duplicates(["group"]).sort_values("group")["sigma"].to_list())

    def get_groups(self, x):
        groups = []
        for i in range(x.shape[1]):
            d = x[:, i].clone()
            miss_mask = d.isnan()
            d[miss_mask] = d.nanmean()
            preds = pd.DataFrame(self.models[i].predict_params(d.reshape(-1, 1)), columns=["mu", "sigma"])
            df_enc = self.ord_enc[i].transform(preds)
            joint_group = np.char.add(np.char.add(df_enc[:, 0].astype(str), "_"), df_enc[:, 1].astype(str))
            joint_group = self.ord_enc_joint[i].transform(joint_group.reshape(-1, 1))
            joint_group = joint_group.astype(float).flatten()
            joint_group[miss_mask] = np.nan
            groups.append(joint_group)
        return np.column_stack(groups)
