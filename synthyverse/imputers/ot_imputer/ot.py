import torch
import numpy as np
import pandas as pd
from geomloss import SamplesLoss
from sklearn.preprocessing import OrdinalEncoder

from tqdm import tqdm

from ..base import BaseImputer


class OTImputer(BaseImputer):
    """Optimal Transport (OT) imputer.

    Using implementation from https://github.com/SamsungSAILMontreal/ForestDiffusion/blob/main/imputers.py

    Args:
        lr (float): Learning rate for optimization. Default: 1e-2.
        opt (type): Optimizer class to use (default: RMSprop). Default: torch.optim.RMSprop.
        niter (int): Number of optimization iterations. Default: 3000.
        batchsize (int): Batch size for optimization. Default: 128.
        n_pairs (int): Number of pairs to use per iteration. Default: 1.
        noise (float): Initial noise level for missing values. Default: 0.1.
        scaling (float): Scaling parameter for Sinkhorn algorithm. Default: 0.9.
        epsilon_params (dict): Dictionary with parameters for epsilon selection:
            - quant: Quantile for epsilon selection (default: 0.5)
            - mult: Multiplier for epsilon (default: 0.05)
            - max_points: Maximum points for epsilon estimation (default: 10000)
        Default: {"quant": 0.5, "mult": 0.05, "max_points": 10000}.
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to BaseImputer.

    Example:
        >>> import pandas as pd
        >>> import torch
        >>> from synthyverse.imputers import OTImputer
        >>>
        >>> # Load data with missing values
        >>> X = pd.read_csv("data_with_missing.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create and fit imputer
        >>> imputer = OTImputer(
        ...     lr=1e-2,
        ...     niter=3000,
        ...     batchsize=128,
        ...     random_state=42
        ... )
        >>> imputer.fit(X, discrete_features)
        >>>
        >>> # Transform data
        >>> X_imputed = imputer.transform(X)
    """

    def __init__(
        self,
        lr=1e-2,
        opt=torch.optim.RMSprop,
        niter=3000,
        batchsize=128,
        n_pairs=1,
        noise=0.1,
        scaling=0.9,
        epsilon_params: dict = {
            "quant": 0.5,
            "mult": 0.05,
            "max_points": 10_000,
        },
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.scaling = scaling
        self.epsilon_params = epsilon_params
        # Automatically detect and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X: pd.DataFrame):
        print("OT is fit at inference-time only...")

    def _transform(self, X: pd.DataFrame):
        # OT expects numerically encoded categoricals
        encoder = OrdinalEncoder()
        X[self.discrete_features] = encoder.fit_transform(X[self.discrete_features])

        x = torch.from_numpy(X.to_numpy()).double().to(self.device)
        self.eps = self.pick_epsilon(
            x,
            self.epsilon_params["quant"],
            self.epsilon_params["mult"],
            self.epsilon_params["max_points"],
        )

        self.sk = SamplesLoss(
            "sinkhorn", p=2, blur=self.eps, scaling=self.scaling, backend="tensorized"
        )

        x = x.clone()
        n, d = x.shape

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e

        mask = torch.isnan(x).double()
        imps = (
            self.noise * torch.randn(mask.shape, device=self.device).double()
            + self.nanmean(x, 0)
        )[mask.bool()]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        pbar = tqdm(range(self.niter), desc="OT Imputation")
        for i in pbar:

            X_filled = x.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]

                loss = loss + self.sk(X1, X2)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                print("\nNan or inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item() / self.n_pairs:.4f}")

        X_filled = x.detach().clone()
        X_filled[mask.bool()] = imps

        X_filled = pd.DataFrame(X_filled.cpu().detach().numpy(), columns=X.columns)
        X_filled[self.discrete_features] = encoder.inverse_transform(
            X_filled[self.discrete_features]
        )

        return X_filled

    def nanmean(self, v, *args, **kwargs):
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    def pick_epsilon(self, X, quant=0.5, mult=0.05, max_points=2000):
        """Select epsilon parameter for Sinkhorn distances.

        Returns a quantile (times a multiplier) of the halved pairwise squared
        distances in X. Used to select a regularization parameter for Sinkhorn distances.

        Args:
            X: Input data tensor on which distances will be computed.
            quant: Quantile to return (default: 0.5, median).
            mult: Multiplier to apply to the quantiles.
            max_points: If the length of X is larger than max_points, estimate
                the quantile on a random subset of size max_points to avoid
                memory overloads.

        Returns:
            float: Epsilon value for Sinkhorn regularization.
        """
        means = self.nanmean(X, 0)
        X_ = X.clone()
        mask = torch.isnan(X_)
        X_[mask] = (mask * means)[mask]

        idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
        X = X_[idx]
        dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.0
        dists = dists[dists > 0]

        return self.quantile(dists, quant, 0).item() * mult

    def quantile(self, X, q, dim=None):
        """Return the q-th quantile.

        Args:
            X: Input data tensor.
            q: Quantile level (starting from lower values).
            dim: Dimension along which to compute quantiles. If None, the tensor
                is flattened and one value is returned.

        Returns:
            torch.DoubleTensor: Quantile values.
        """
        return X.kthvalue(int(q * len(X)), dim=dim)[0]
