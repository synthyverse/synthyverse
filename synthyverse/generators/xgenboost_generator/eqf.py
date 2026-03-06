import numpy as np
from typing import Optional, Union, Tuple
from numpy.typing import ArrayLike


class EmpiricalInterpolatedQuantile:

    def __init__(
        self,
        n_knots: int = -1,
        use_spline: bool = False,
    ):
        """
        Parameters
        ----------
        n_knots : int
            Number of quantile knots to summarize the distribution.
            If n_knots <= 0, use all training samples as knots.
        use_spline : bool, optional
            If True, use a monotone cubic Hermite spline between knots.
            If False, use piecewise-linear interpolation between knots.
        """
        self.x_sorted_ = None
        self.n_samples_ = 0

        self.n_knots_ = n_knots
        self.use_spline_ = use_spline

        self.q_knots_ = None
        self.x_knots_ = None
        self.slopes_ = None

    def _check_fitted(self):
        if self.x_sorted_ is None or self.n_samples_ == 0:
            raise RuntimeError("Distribution not fitted. Call `fit` first.")

    @staticmethod
    def _linear_quantile_on_sorted(xs: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute linear quantiles on a *sorted* 1D array xs for given q in [0,1].
        """
        n = xs.size
        q = np.asarray(q, dtype=np.float64)
        if n == 1:
            return np.full_like(q, xs[0], dtype=np.float64)

        q_flat = q.ravel()
        pos = q_flat * (n - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        np.clip(lo, 0, n - 1, out=lo)
        np.clip(hi, 0, n - 1, out=hi)

        t = pos - lo
        x_lo = xs[lo]
        x_hi = xs[hi]
        out = (1.0 - t) * x_lo + t * x_hi
        return out.reshape(q.shape)

    @staticmethod
    def _compute_monotone_slopes(
        q_knots: np.ndarray, x_knots: np.ndarray
    ) -> np.ndarray:
        """
        Compute monotone cubic Hermite slopes (dQ/dq at knots) using a
        Fritsch–Carlson-type method. Assumes q_knots strictly increasing.
        """
        n = x_knots.size
        slopes = np.zeros_like(x_knots, dtype=np.float64)
        if n == 1:
            slopes[0] = 0.0
            return slopes
        if n == 2:
            dq = q_knots[1] - q_knots[0]
            d = (x_knots[1] - x_knots[0]) / dq if dq > 0 else 0.0
            slopes[:] = d
            return slopes

        h = np.diff(q_knots)  # > 0
        delta = np.diff(x_knots) / h

        slopes[0] = delta[0]
        slopes[-1] = delta[-1]

        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0.0:
                slopes[i] = 0.0
            else:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

        mask_flat = delta == 0.0
        if np.any(mask_flat):
            idx = np.where(mask_flat)[0]
            for i in idx:
                slopes[i] = 0.0
                slopes[i + 1] = 0.0

        return slopes

    def fit(self, x: ArrayLike) -> "EmpiricalInterpolatedQuantile":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            x = x.ravel()
        if x.size == 0:
            raise ValueError("x must contain at least one sample")

        self.x_sorted_ = np.sort(x)
        self.n_samples_ = self.x_sorted_.size

        if self.n_knots_ is None or self.n_knots_ <= 0:
            m = self.n_samples_
        else:
            m = min(self.n_knots_, self.n_samples_)

        if m == 1:
            self.q_knots_ = np.array([0.0], dtype=np.float64)
            self.x_knots_ = np.array([self.x_sorted_[0]], dtype=np.float64)
            self.slopes_ = np.array([0.0], dtype=np.float64)
            return self

        self.q_knots_ = np.linspace(0.0, 1.0, m)
        self.x_knots_ = self._linear_quantile_on_sorted(self.x_sorted_, self.q_knots_)

        if self.use_spline_:
            self.slopes_ = self._compute_monotone_slopes(self.q_knots_, self.x_knots_)
        else:
            self.slopes_ = None

        return self

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """
        Empirical CDF: F(x) = P(X <= x), standard ECDF (step)
        based on the full sorted data.
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.x_sorted_, x, side="right")
        return idx / self.n_samples_

    def ppf(self, q: ArrayLike) -> np.ndarray:
        """
        Interpolated / smoothed quantile on the knot representation.
        """
        self._check_fitted()
        q = np.asarray(q, dtype=np.float64)
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("Quantiles q must lie in [0, 1].")

        if self.n_samples_ == 1 or self.x_knots_ is None or self.x_knots_.size == 1:
            return np.full_like(q, self.x_sorted_[0], dtype=np.float64)

        q_flat = q.ravel()
        m = self.x_knots_.size

        q_clip = np.clip(q_flat, 0.0, 1.0)
        pos = q_clip * (m - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = lo + 1
        np.clip(lo, 0, m - 1, out=lo)
        np.clip(hi, 0, m - 1, out=hi)
        t = pos - lo  # ∈ [0,1]

        xk = self.x_knots_

        if not self.use_spline_ or self.slopes_ is None or m < 2:
            x_lo = xk[lo]
            x_hi = xk[hi]
            out = (1.0 - t) * x_lo + t * x_hi
            return out.reshape(q.shape)

        slopes = self.slopes_
        out_flat = np.empty_like(q_flat, dtype=np.float64)

        mask_interval = hi > lo
        if np.any(~mask_interval):
            out_flat[~mask_interval] = xk[lo[~mask_interval]]

        if np.any(mask_interval):
            k = lo[mask_interval]
            tp = t[mask_interval]

            h = 1.0 / (m - 1)

            y0 = xk[k]
            y1 = xk[k + 1]
            m0 = slopes[k]
            m1 = slopes[k + 1]

            tp2 = tp * tp
            tp3 = tp2 * tp

            h00 = 2.0 * tp3 - 3.0 * tp2 + 1.0
            h10 = tp3 - 2.0 * tp2 + tp
            h01 = -2.0 * tp3 + 3.0 * tp2
            h11 = tp3 - tp2

            out_flat[mask_interval] = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

        return out_flat.reshape(q.shape)

    def rvs(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample via inverse transform: U ~ Uniform(0,1), X = Q(U).
        """
        self._check_fitted()
        if rng is None:
            rng = np.random.default_rng()

        u = (
            rng.uniform(0.0, 1.0, size=size)
            if size is not None
            else rng.uniform(0.0, 1.0)
        )
        return self.ppf(u)
