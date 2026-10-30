from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float
    min_variance: float = 0.0  # set > 0 for numerical stability

    def __post_init__(self):
        feller_condition = 2 * self.kappa * self.theta >= self.xi**2
        if not feller_condition:
            raise ValueError("Feller condition not met.")


HESTON_DEFAULT_PARAMS = HestonParams(
    mu=0.0,
    kappa=1.0,
    theta=0.04,
    xi=0.25,
    rho=-0.7,
    min_variance=1e-7,
)


class HestonSimulator:
    def __init__(
        self,
        params: HestonParams,
        n_steps: int,
        dt: float = 1 / 250,
        rng: np.random.Generator | None = None,
    ):
        """Heston simulation using the Broadie Kaya scheme.
        Step 3.2 of the precise scheme is replaced with trapz approximation."""
        self.p, self.dt, self.n_steps = params, dt, n_steps
        self.rng = rng or np.random.default_rng()

    @cached_property
    def xi_sq(self):
        return self.p.xi**2

    @cached_property
    def dof(self):
        """degrees of freedom for noncentral chi-squared distribution (Eq. 9)"""
        return 4 * self.p.kappa * self.p.theta / self.xi_sq

    @cached_property
    def exp_k_dt(self):
        return np.exp(-self.p.kappa * self.dt)

    @cached_property
    def pre_non_centrality(self):
        """non_centrality = v_u * pre_non_centrality; (Eq. 10)"""
        return 4 * self.p.kappa * self.exp_k_dt / (self.xi_sq * (1 - self.exp_k_dt))

    @cached_property
    def v_t_scl(self):
        return (self.xi_sq * (1 - self.exp_k_dt)) / (4 * self.p.kappa)

    def sample_v_t(self, v_u: np.ndarray) -> np.ndarray:  # vu.shape = (bs,)
        """Step 3.1 in BK scheme"""
        non_centrality = self.pre_non_centrality * v_u
        v_t = self.rng.noncentral_chisquare(
            self.dof, non_centrality, size=(v_u.shape[0],)
        )
        return np.maximum(v_t * self.v_t_scl, self.p.min_variance)

    def _sample_realized_variance(self, v_u: np.ndarray, v_t: np.ndarray):
        """Step 3.2 in BK scheme with simple approximation -> just using trapz rule"""
        # v_u.shape = (bs,), v_t.shape = (bs,)
        return self.dt * (v_u + v_t) / 2

    def _sample_s_t(
        self, s_u: np.ndarray, v_u: np.ndarray, v_t: np.ndarray, rv: np.ndarray
    ) -> np.ndarray:
        """Step 3.3 in BK scheme"""
        # s_u.shape = (bs,), v_u.shape = (bs,), v_t.shape = (bs,), rv.shape = (bs,)
        # Eq. 23
        int_vol_dW1 = (1 / self.p.xi) * (
            v_t - v_u - self.p.kappa * self.p.theta * self.dt + self.p.kappa * rv
        )

        # m,v are mean variance of log(S_t)
        m = np.log(s_u) + (self.p.mu * self.dt - 0.5 * rv + self.p.rho * int_vol_dW1)
        v = (1 - self.p.rho**2) * rv
        return np.exp(self.rng.normal(m, np.sqrt(v)))

    def simulate(
        self,
        bs: int,
        s0: float | np.ndarray | None = None,
        v0: float | np.ndarray | None = None,
    ):
        s0 = 1.0 if s0 is None else s0
        v0 = self.p.theta if v0 is None else v0
        s_u = s0 if isinstance(s0, np.ndarray) else np.full(bs, s0)
        v_u = v0 if isinstance(v0, np.ndarray) else np.full(bs, v0)

        spot = np.empty((bs, self.n_steps + 1), dtype=np.float32)
        variance = np.empty((bs, self.n_steps + 1), dtype=np.float32)
        spot[:, 0], variance[:, 0] = s_u, v_u

        for i in range(self.n_steps):
            v_t = self.sample_v_t(v_u)
            rv = self._sample_realized_variance(v_u, v_t)
            s_t = self._sample_s_t(s_u, v_u, v_t, rv)

            spot[:, i + 1] = s_t
            variance[:, i + 1] = v_t

            s_u, v_u = s_t, v_t

        return spot, variance

    @staticmethod
    def transform(spot: np.ndarray, variance: np.ndarray) -> np.ndarray:
        # as the raw 'data' we want to use a *stationary* Markov process
        # in discretized Heston (log-returns, variance) or (log-returns, log-variance)
        # is a Markov process (pick the latter to avoid >= 0 requirement)
        r = np.log(spot[:, 1:] / spot[:, :-1])
        v = np.log(variance)[:, 1:]
        return np.stack((r, v), axis=-1)

    def simulate_transformed(
        self,
        bs: int,
        s0: float | np.ndarray | None = None,
        v0: float | np.ndarray | None = None,
    ) -> np.ndarray:
        return self.transform(*self.simulate(bs, s0, v0))

    def sample_stationary_v(self, bs: int) -> np.ndarray:
        """Sample v_0 from the stationary distribution of the CIR process."""
        alpha = 2.0 * self.p.kappa * self.p.theta / self.p.xi**2
        scale = self.p.xi**2 / (2.0 * self.p.kappa)
        return self.rng.gamma(shape=alpha, scale=scale, size=bs).astype(np.float32)
