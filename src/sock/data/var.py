from functools import cached_property

import numpy as np


class VAR1Simulator:
    def __init__(
        self,
        n_steps: int,
        phi: float = 0.99,
        corr_weight: float = 0.5,
        dim: int = 3,
        rng: np.random.Generator | None = None,
    ):
        """
        This is very close to the setup describe in section 6.1 of:
            Sig-Wasserstein GANs for conditional time series generation (Liao et al., 2024)
        """
        self.n_steps = n_steps
        self.phi = phi
        self.corr_weight = corr_weight
        self.dim = dim
        self.rng = rng or np.random.default_rng()

    @cached_property
    def cov(self) -> np.ndarray:
        cov = self.corr_weight * np.ones((self.dim, self.dim))
        cov += (1 - self.corr_weight) * np.eye(self.dim)
        return cov

    @cached_property
    def chol(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov).astype(np.float32)

    def simulate(self, bs: int, x0: np.ndarray | None = None) -> np.ndarray:
        x0 = np.zeros((bs, self.dim), dtype=np.float32) if x0 is None else x0
        x = np.empty((bs, self.n_steps + 1, self.dim), dtype=np.float32)
        x[:, 0] = x0  # NOTE: not included in output
        for i in range(self.n_steps):
            z = self.rng.normal(size=(bs, self.dim))
            x[:, i + 1] = self.phi * x[:, i] + z @ self.chol.T
        x = x[:, 1:]  # drop x0
        return x

    def sample_stationary_x0(self, bs: int) -> np.ndarray:
        denom = 1.0 - self.phi**2
        scale = np.float32(1.0 / np.sqrt(denom))
        z = self.rng.normal(size=(bs, self.dim))
        return z @ (self.chol.T * scale)
