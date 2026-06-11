"""Goal-scoring models, all driven by the same Elo strength feature.

Four interchangeable models share one interface (fit / score grid /
win-draw-loss / scoreline sampling):

``DixonColesModel`` (dc)
    Independent Poissons with the Dixon-Coles tau correction for the
    0-0 / 1-0 / 0-1 / 1-1 cells (Dixon & Coles 1997).

``BivariatePoissonModel`` (bp)
    Karlis & Ntzoufras (2003) bivariate Poisson: X = X1 + X3, Y = X2 + X3
    with a shared component X3 ~ Poisson(l3) that induces positive
    score correlation and lifts the draw probability.

``NegBinModel`` (nb)
    Independent negative-binomial goals with a shared dispersion
    parameter, capturing the over-dispersion (blowout tails) that a
    Poisson misses.

``EnsembleModel`` (ensemble)
    Equal-weight mixture of the three: probabilities are averaged,
    sampling draws each scoreline from a randomly chosen member, which
    is exactly sampling from the averaged distribution.

Every model maps the pre-match Elo ratings to expected goals through a
log-linear link::

    log lam_home = c0 + c1 * (R_home - R_away)/100 + c2 * home_flag
    log mu_away  = c0 - c1 * (R_home - R_away)/100

and is fit by *weighted* maximum likelihood (exponential time decay x
competition weight), so recent and competitive matches dominate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import SimConfig
from .data import fit_weights

_ELO_SCALE = 100.0  # Elo points per unit of the goals-model covariate

MODEL_NAMES = ("dc", "bp", "nb", "ensemble")


def _poisson_grid(rate: np.ndarray, g: int) -> np.ndarray:
    """Poisson pmf over counts 0..g for each rate: shape (K, g+1)."""
    ks = np.arange(g + 1)
    logp = (
        -rate[:, None]
        + ks[None, :] * np.log(rate[:, None])
        - gammaln(ks + 1)[None, :]
    )
    return np.exp(logp)


@dataclass
class _FitData:
    """Design arrays shared by every model's likelihood."""

    x: np.ndarray          # home goals
    y: np.ndarray          # away goals
    d: np.ndarray          # (R_home - R_away) / 100
    home_flag: np.ndarray  # 1.0 at non-neutral venues
    w: np.ndarray          # MLE weights, normalised to mean 1


class GoalsModel(ABC):
    """Common interface: fit, probabilities, and scoreline sampling."""

    name: str = "base"
    label: str = "base"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._n_fit = 0
        # post-hoc calibration multiplier on the Elo->goals slope; tuned
        # out-of-sample (wcsim/tune.py) and applied at prediction time.
        self.slope_scale = cfg.slope_scale

    # ------------------------------------------------------------------ #
    def _prepare(self, prematch: pd.DataFrame, cutoff: pd.Timestamp) -> _FitData:
        df = prematch
        if self.cfg.fit_min_year is not None:
            df = df[pd.to_datetime(df["date"]).dt.year >= self.cfg.fit_min_year]
        df = df.reset_index(drop=True)
        w = fit_weights(
            df,
            cutoff=cutoff,
            half_life_days=self.cfg.half_life_days,
            friendly_weight=self.cfg.friendly_weight,
        )
        self._n_fit = len(df)
        return _FitData(
            x=df["home_score"].to_numpy(dtype=float),
            y=df["away_score"].to_numpy(dtype=float),
            d=(df["r_home_pre"].to_numpy() - df["r_away_pre"].to_numpy()) / _ELO_SCALE,
            home_flag=(~df["neutral"].to_numpy()).astype(float),
            w=w / w.mean(),
        )

    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, prematch: pd.DataFrame, cutoff: pd.Timestamp) -> "GoalsModel":
        ...

    @abstractmethod
    def expected_goals(self, r_home, r_away, neutral=True):
        """Marginal expected goals (lam_home, mu_away); broadcasts."""
        ...

    @abstractmethod
    def score_pmf(self, r_home, r_away, neutral=True) -> np.ndarray:
        """Normalised P(x, y) grid, shape (K, G+1, G+1)."""
        ...

    @abstractmethod
    def summary(self) -> str:
        ...

    # ------------------------------------------------------------------ #
    # Shared, grid-based outcome probabilities
    # ------------------------------------------------------------------ #
    def win_draw_loss(self, r_home, r_away, neutral=True):
        """Return (p_home_win, p_draw, p_away_win), vectorised."""
        P = self.score_pmf(r_home, r_away, neutral)
        g = self.cfg.max_goals
        idx = np.arange(g + 1)
        p_home = (P * (idx[:, None] > idx[None, :])).sum(axis=(1, 2))
        p_away = (P * (idx[:, None] < idx[None, :])).sum(axis=(1, 2))
        p_draw = (P * (idx[:, None] == idx[None, :])).sum(axis=(1, 2))
        if np.isscalar(r_home) and np.isscalar(r_away):
            return float(p_home[0]), float(p_draw[0]), float(p_away[0])
        return p_home, p_draw, p_away

    # ------------------------------------------------------------------ #
    # Sampling (grid-based defaults; subclasses may override with direct
    # samplers that avoid building the grid)
    # ------------------------------------------------------------------ #
    def sample_scores_fixed(self, r_home: float, r_away: float, neutral: bool,
                            n: int, rng: np.random.Generator):
        """n scoreline draws for ONE fixture (identical across sims)."""
        P = self.score_pmf(r_home, r_away, neutral)[0]
        cdf = np.cumsum(P.ravel())
        cdf[-1] = 1.0
        idx = np.searchsorted(cdf, rng.random(n), side="right")
        g1 = self.cfg.max_goals + 1
        return (idx // g1).astype(np.int64), (idx % g1).astype(np.int64)

    def sample_scores_array(self, r_home: np.ndarray, r_away: np.ndarray,
                            neutral, rng: np.random.Generator):
        """One scoreline draw per element (pairings differ per sim)."""
        P = self.score_pmf(r_home, r_away, neutral)
        K = P.shape[0]
        cdf = np.cumsum(P.reshape(K, -1), axis=1)
        cdf[:, -1] = 1.0
        idx = (cdf >= rng.random(K)[:, None]).argmax(axis=1)
        g1 = self.cfg.max_goals + 1
        return (idx // g1).astype(np.int64), (idx % g1).astype(np.int64)


# =========================================================================== #
# 1. Dixon-Coles
# =========================================================================== #
class DixonColesModel(GoalsModel):
    name = "dc"
    label = "Dixon-Coles Poisson"

    def __init__(self, cfg: SimConfig):
        super().__init__(cfg)
        self.theta: np.ndarray | None = None   # [c0, c1, c2, rho]
        self.cov: np.ndarray | None = None     # approx parameter covariance

    def fit(self, prematch, cutoff):
        fd = self._prepare(prematch, cutoff)
        x, y, d, h, w = fd.x, fd.y, fd.d, fd.home_flag, fd.w
        const = -(gammaln(x + 1.0) + gammaln(y + 1.0))

        def neg_ll(theta):
            c0, c1, c2, rho = theta
            lam = np.exp(c0 + c1 * d + c2 * h)
            mu = np.exp(c0 - c1 * d)
            tau = np.ones_like(lam)
            tau = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, tau)
            tau = np.where((x == 0) & (y == 1), 1.0 + lam * rho, tau)
            tau = np.where((x == 1) & (y == 0), 1.0 + mu * rho, tau)
            tau = np.where((x == 1) & (y == 1), 1.0 - rho, tau)
            tau = np.clip(tau, 1e-10, None)
            ll = -lam + x * np.log(lam) - mu + y * np.log(mu) + np.log(tau) + const
            return -float(np.sum(w * ll))

        res = minimize(
            neg_ll,
            np.array([np.log(1.35), 0.35, 0.25, -0.05]),
            method="L-BFGS-B",
            bounds=[(-2, 2), (-2, 2), (-1, 1), (-0.2, 0.2)],
        )
        self.theta = res.x
        try:
            self.cov = np.asarray(res.hess_inv.todense())
        except Exception:
            self.cov = None
        return self

    def _rates(self, r_home, r_away, neutral):
        c0, c1, c2, _ = self.theta
        c1 = c1 * self.slope_scale
        d = (np.asarray(r_home, dtype=float) - np.asarray(r_away, dtype=float)) / _ELO_SCALE
        h = np.where(np.asarray(neutral), 0.0, 1.0)
        return np.exp(c0 + c1 * d + c2 * h), np.exp(c0 - c1 * d)

    def expected_goals(self, r_home, r_away, neutral=True):
        return self._rates(r_home, r_away, neutral)

    def score_pmf(self, r_home, r_away, neutral=True):
        lam, mu = self._rates(r_home, r_away, neutral)
        lam, mu = np.atleast_1d(lam), np.atleast_1d(mu)
        g = self.cfg.max_goals
        P = _poisson_grid(lam, g)[:, :, None] * _poisson_grid(mu, g)[:, None, :]
        rho = self.theta[3]
        P[:, 0, 0] *= 1.0 - lam * mu * rho
        P[:, 0, 1] *= 1.0 + lam * rho
        P[:, 1, 0] *= 1.0 + mu * rho
        P[:, 1, 1] *= 1.0 - rho
        np.clip(P, 0.0, None, out=P)
        P /= P.sum(axis=(1, 2), keepdims=True)
        return P

    def summary(self):
        c0, c1, c2, rho = self.theta
        return (
            f"Dixon-Coles ({self._n_fit} matches): "
            f"c0={c0:.4f} ({np.exp(c0):.3f} goals/side even+neutral), "
            f"c1={c1:.4f}/100Elo, c2={c2:.4f} (home x{np.exp(c2):.3f}), "
            f"rho={rho:.4f}"
        )


# =========================================================================== #
# 2. Bivariate Poisson (Karlis & Ntzoufras)
# =========================================================================== #
class BivariatePoissonModel(GoalsModel):
    name = "bp"
    label = "Bivariate Poisson"

    def __init__(self, cfg: SimConfig):
        super().__init__(cfg)
        self.theta: np.ndarray | None = None   # [c0, c1, c2, log_l3]

    def fit(self, prematch, cutoff):
        fd = self._prepare(prematch, cutoff)
        x, y, d, h, w = fd.x, fd.y, fd.d, fd.home_flag, fd.w
        xi, yi = x.astype(int), y.astype(int)
        m = np.minimum(xi, yi)
        kmax = int(m.max())
        lgx, lgy = gammaln(x + 1.0), gammaln(y + 1.0)

        def neg_ll(theta):
            c0, c1, c2, ll3 = theta
            l1 = np.exp(c0 + c1 * d + c2 * h)
            l2 = np.exp(c0 - c1 * d)
            l3 = np.exp(ll3)
            # S = sum_k C(x,k) C(y,k) k! * (l3/(l1*l2))^k   (Karlis-Ntzoufras)
            log_u = np.log(l3) - np.log(l1) - np.log(l2)
            S = np.zeros_like(x)
            for k in range(kmax + 1):
                mask = m >= k
                if not mask.any():
                    break
                xs, ys = x[mask], y[mask]
                log_term = (
                    gammaln(xs + 1) - gammaln(xs - k + 1)
                    + gammaln(ys + 1) - gammaln(ys - k + 1)
                    - gammaln(k + 1)
                    + k * log_u[mask]
                )
                S[mask] += np.exp(log_term)
            ll = (
                -(l1 + l2 + l3)
                + x * np.log(l1) - lgx
                + y * np.log(l2) - lgy
                + np.log(np.clip(S, 1e-300, None))
            )
            return -float(np.sum(w * ll))

        res = minimize(
            neg_ll,
            np.array([np.log(1.2), 0.35, 0.25, np.log(0.1)]),
            method="L-BFGS-B",
            bounds=[(-2, 2), (-2, 2), (-1, 1), (np.log(1e-4), np.log(1.5))],
        )
        self.theta = res.x
        return self

    def _rates(self, r_home, r_away, neutral):
        c0, c1, c2, ll3 = self.theta
        c1 = c1 * self.slope_scale
        d = (np.asarray(r_home, dtype=float) - np.asarray(r_away, dtype=float)) / _ELO_SCALE
        h = np.where(np.asarray(neutral), 0.0, 1.0)
        l1 = np.exp(c0 + c1 * d + c2 * h)
        l2 = np.exp(c0 - c1 * d)
        return l1, l2, np.exp(ll3)

    def expected_goals(self, r_home, r_away, neutral=True):
        l1, l2, l3 = self._rates(r_home, r_away, neutral)
        return l1 + l3, l2 + l3

    def score_pmf(self, r_home, r_away, neutral=True):
        l1, l2, l3 = self._rates(r_home, r_away, neutral)
        l1, l2 = np.atleast_1d(l1), np.atleast_1d(l2)
        g = self.cfg.max_goals
        K = l1.shape[0]
        p1 = _poisson_grid(l1, g)
        p2 = _poisson_grid(l2, g)
        p3 = _poisson_grid(np.full(K, l3), g)
        P = np.zeros((K, g + 1, g + 1))
        for k in range(g + 1):
            gk = g + 1 - k
            P[:, k:, k:] += (
                p3[:, k, None, None] * p1[:, :gk, None] * p2[:, None, :gk]
            )
        P /= P.sum(axis=(1, 2), keepdims=True)
        return P

    def sample_scores_fixed(self, r_home, r_away, neutral, n, rng):
        l1, l2, l3 = self._rates(r_home, r_away, neutral)
        x3 = rng.poisson(float(l3), size=n)
        return rng.poisson(float(l1), size=n) + x3, rng.poisson(float(l2), size=n) + x3

    def sample_scores_array(self, r_home, r_away, neutral, rng):
        l1, l2, l3 = self._rates(r_home, r_away, neutral)
        x3 = rng.poisson(np.broadcast_to(l3, l1.shape))
        return rng.poisson(l1) + x3, rng.poisson(l2) + x3

    def summary(self):
        c0, c1, c2, ll3 = self.theta
        return (
            f"Bivariate Poisson ({self._n_fit} matches): "
            f"c0={c0:.4f}, c1={c1:.4f}/100Elo, c2={c2:.4f} (home x{np.exp(c2):.3f}), "
            f"l3={np.exp(ll3):.4f} (shared component / score covariance)"
        )


# =========================================================================== #
# 3. Negative binomial (over-dispersed goals)
# =========================================================================== #
class NegBinModel(GoalsModel):
    name = "nb"
    label = "Negative Binomial"

    def __init__(self, cfg: SimConfig):
        super().__init__(cfg)
        self.theta: np.ndarray | None = None   # [c0, c1, c2, log_r]

    @staticmethod
    def _nb_logpmf(k: np.ndarray, r: float, mean: np.ndarray) -> np.ndarray:
        p = r / (r + mean)  # success prob in scipy's convention
        return (
            gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * np.log(p) + k * np.log1p(-p)
        )

    def fit(self, prematch, cutoff):
        fd = self._prepare(prematch, cutoff)
        x, y, d, h, w = fd.x, fd.y, fd.d, fd.home_flag, fd.w

        def neg_ll(theta):
            c0, c1, c2, logr = theta
            r = np.exp(logr)
            lam = np.exp(c0 + c1 * d + c2 * h)
            mu = np.exp(c0 - c1 * d)
            ll = self._nb_logpmf(x, r, lam) + self._nb_logpmf(y, r, mu)
            return -float(np.sum(w * ll))

        res = minimize(
            neg_ll,
            np.array([np.log(1.35), 0.35, 0.25, np.log(8.0)]),
            method="L-BFGS-B",
            bounds=[(-2, 2), (-2, 2), (-1, 1), (np.log(0.5), np.log(500.0))],
        )
        self.theta = res.x
        return self

    def _rates(self, r_home, r_away, neutral):
        c0, c1, c2, _ = self.theta
        c1 = c1 * self.slope_scale
        d = (np.asarray(r_home, dtype=float) - np.asarray(r_away, dtype=float)) / _ELO_SCALE
        h = np.where(np.asarray(neutral), 0.0, 1.0)
        return np.exp(c0 + c1 * d + c2 * h), np.exp(c0 - c1 * d)

    def expected_goals(self, r_home, r_away, neutral=True):
        return self._rates(r_home, r_away, neutral)

    def score_pmf(self, r_home, r_away, neutral=True):
        lam, mu = self._rates(r_home, r_away, neutral)
        lam, mu = np.atleast_1d(lam), np.atleast_1d(mu)
        g = self.cfg.max_goals
        r = np.exp(self.theta[3])
        ks = np.arange(g + 1, dtype=float)
        ph = np.exp(self._nb_logpmf(ks[None, :], r, lam[:, None]))
        pa = np.exp(self._nb_logpmf(ks[None, :], r, mu[:, None]))
        P = ph[:, :, None] * pa[:, None, :]
        P /= P.sum(axis=(1, 2), keepdims=True)
        return P

    def _sample(self, mean: np.ndarray, rng) -> np.ndarray:
        r = np.exp(self.theta[3])
        return rng.poisson(rng.gamma(r, mean / r))

    def sample_scores_fixed(self, r_home, r_away, neutral, n, rng):
        lam, mu = self._rates(r_home, r_away, neutral)
        lam_a = np.full(n, float(lam))
        mu_a = np.full(n, float(mu))
        return self._sample(lam_a, rng), self._sample(mu_a, rng)

    def sample_scores_array(self, r_home, r_away, neutral, rng):
        lam, mu = self._rates(r_home, r_away, neutral)
        return self._sample(lam, rng), self._sample(mu, rng)

    def summary(self):
        c0, c1, c2, logr = self.theta
        r = np.exp(logr)
        return (
            f"Negative Binomial ({self._n_fit} matches): "
            f"c0={c0:.4f}, c1={c1:.4f}/100Elo, c2={c2:.4f} (home x{np.exp(c2):.3f}), "
            f"dispersion r={r:.1f} (var/mean at 1.3 goals: {1 + 1.3 / r:.3f})"
        )


# =========================================================================== #
# 4. Ensemble (equal-weight mixture)
# =========================================================================== #
class EnsembleModel(GoalsModel):
    name = "ensemble"
    label = "Ensemble (DC + BP + NB)"

    def __init__(self, cfg: SimConfig, members: list[GoalsModel] | None = None):
        super().__init__(cfg)
        self.members = members or [
            DixonColesModel(cfg),
            BivariatePoissonModel(cfg),
            NegBinModel(cfg),
        ]

    def fit(self, prematch, cutoff):
        for m in self.members:
            m.fit(prematch, cutoff)
        self._n_fit = self.members[0]._n_fit
        return self

    def expected_goals(self, r_home, r_away, neutral=True):
        lams, mus = zip(*(m.expected_goals(r_home, r_away, neutral) for m in self.members))
        return np.mean(lams, axis=0), np.mean(mus, axis=0)

    def score_pmf(self, r_home, r_away, neutral=True):
        return np.mean(
            [m.score_pmf(r_home, r_away, neutral) for m in self.members], axis=0
        )

    def sample_scores_fixed(self, r_home, r_away, neutral, n, rng):
        # Mixture sampling == sampling from the averaged pmf.  Shuffle so the
        # member choice is independent of the simulation index.
        counts = rng.multinomial(n, np.full(len(self.members), 1.0 / len(self.members)))
        hs = np.empty(n, dtype=np.int64)
        as_ = np.empty(n, dtype=np.int64)
        pos = 0
        for m, c in zip(self.members, counts):
            if c == 0:
                continue
            h, a = m.sample_scores_fixed(r_home, r_away, neutral, int(c), rng)
            hs[pos:pos + c], as_[pos:pos + c] = h, a
            pos += c
        perm = rng.permutation(n)
        return hs[perm], as_[perm]

    def sample_scores_array(self, r_home, r_away, neutral, rng):
        K = len(np.atleast_1d(r_home))
        which = rng.integers(0, len(self.members), size=K)
        hs = np.empty(K, dtype=np.int64)
        as_ = np.empty(K, dtype=np.int64)
        neutral_arr = np.broadcast_to(np.asarray(neutral), (K,))
        for mi, m in enumerate(self.members):
            sel = which == mi
            if not sel.any():
                continue
            h, a = m.sample_scores_array(r_home[sel], r_away[sel], neutral_arr[sel], rng)
            hs[sel], as_[sel] = h, a
        return hs, as_

    def summary(self):
        return "Ensemble (equal-weight mixture):\n  " + "\n  ".join(
            m.summary() for m in self.members
        )


# --------------------------------------------------------------------------- #
def make_model(name: str, cfg: SimConfig) -> GoalsModel:
    """Factory: dc | bp | nb | ensemble."""
    cls = {
        "dc": DixonColesModel,
        "bp": BivariatePoissonModel,
        "nb": NegBinModel,
        "ensemble": EnsembleModel,
    }.get(name)
    if cls is None:
        raise ValueError(f"unknown model '{name}' (choose from {MODEL_NAMES})")
    return cls(cfg)
