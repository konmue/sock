from functools import cached_property

import numpy as np
import torch


class VARpSimulator:
    """General VAR(p) simulator:
    z_t = sum_{k in lags} A_k z_{t-k} + eps_t,   eps_t ~ N(0, Sigma)
    """

    def __init__(
        self,
        n_steps: int,
        lags: dict[int, np.ndarray],  # {k: A_k} where A_k is (dim, dim)
        cov_trace: float | None = 0.2,
        corr_weight: float | None = 0.5,
        cov: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_steps = int(n_steps)
        self.lags = {int(k): np.asarray(A, dtype=np.float32) for k, A in lags.items()}
        self.p = max(self.lags)
        self.dim = next(iter(self.lags.values())).shape[0]
        for k, A in self.lags.items():
            if A.shape != (self.dim, self.dim):
                raise ValueError(f"Lag {k} has invalid shape {A.shape}.")
            if k < 1:
                raise ValueError("Lag indices must be >= 1.")
        self.rng = rng or np.random.default_rng()

        if cov is not None:
            self._cov = cov
        else:
            self.cov_trace = float(cov_trace)
            self.corr_weight = float(corr_weight)
            self._cov = None

    @cached_property
    def cov(self) -> np.ndarray:
        if self._cov is not None:
            return self._cov
        cov = self.corr_weight * np.ones((self.dim, self.dim), dtype=np.float32)
        cov += (1.0 - self.corr_weight) * np.eye(self.dim, dtype=np.float32)
        cov *= self.cov_trace / np.trace(cov)
        return cov

    @cached_property
    def chol(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov).astype(np.float32)

    def simulate(
        self, bs: int, x0: np.ndarray | None = None, burn: int = 0
    ) -> np.ndarray:
        bs = int(bs)
        p, d = self.p, self.dim

        if x0 is None:
            hist = np.zeros((bs, p, d), dtype=np.float32)
        else:
            # NOTE: we assume that x0 has the opposite ordering to our hist buffer
            hist = np.asarray(x0, dtype=np.float32)
            hist = np.flip(hist, axis=1)
            assert hist.shape == (bs, p, d)

        # NOTE: x0 does not appear in output x
        x = np.empty((bs, self.n_steps, d), dtype=np.float32)
        burn_steps = max(int(burn), 0)
        for t in range(burn_steps + self.n_steps):
            eps = self.rng.normal(size=(bs, d)).astype(np.float32) @ self.chol.T
            x_next = self._one_step_from_hist(hist) + eps
            if t >= burn_steps:
                x[:, t - burn_steps] = x_next
            hist = self._update_hist(hist, x_next)
        return x

    def _one_step_from_hist(self, hist: np.ndarray) -> np.ndarray:
        """
        Given history buffer `hist` of shape (bs, p, dim) with hist[:,0]=z_t,
        compute deterministic part sum_k A_k z_{t+1-k}.
        """
        out = np.zeros_like(hist[:, 0])
        for k, A in self.lags.items():
            out += hist[:, k - 1] @ A.T
        return out

    @staticmethod
    def _update_hist(hist: np.ndarray, next_z: np.ndarray) -> np.ndarray:
        return np.concatenate([next_z[:, None, :], hist[:, :-1, :]], axis=1)


def tgh_transform(x: torch.Tensor, g: float, h: float) -> torch.Tensor:
    """
    Apply the Tukey g-and-h transformation to a tensor.
    Args:
        x (Tensor): input
        g(float): g parameter (controls skewness)
        h(float): h parameter (controls tail heaviness)
    Returns:
        Tensor: transformed input
    """
    if g == 0.0:
        return x * torch.exp(0.5 * h * x.square())
    return (torch.exp(g * x) - 1) / g * torch.exp(0.5 * h * x.square())
