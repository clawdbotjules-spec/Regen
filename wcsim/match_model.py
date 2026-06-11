"""Dixon-Coles Poisson goals model, driven by Elo ratings.

The expected goals for each side are a log-linear function of the Elo
rating difference plus a home-advantage term::

    log lambda_home = c0 + c1 * (R_home - R_away)/100 + c2 * home_flag
    log mu_away     = c0 - c1 * (R_home - R_away)/100

A single league-average intercept ``c0`` sets the baseline scoring rate,
``c1`` translates an Elo edge into goal supremacy, and ``c2`` is the home
effect (applied only at non-neutral venues).  On top of the independent
Poissons we apply the Dixon-Coles low-score correction ``tau(x, y; rho)``
which fixes the well-known under-/over-prediction of 0-0, 1-0, 0-1, 1-1.

All four parameters [c0, c1, c2, rho] are fit jointly by maximising a
*weighted* likelihood (exponential time-decay x competition weight), so
recent and competitive matches count more.

The model exposes:
    * ``win_draw_loss`` -> (p_home, p_draw, p_away)   (used by the backtest)
    * ``sample_scores`` -> sampled scorelines          (used by the simulator)
both vectorised for speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import SimConfig
from .data import fit_weights

_ELO_SCALE = 100.0  # Elo points per unit of the goals-model covariate


def _dc_tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction, vectorised over arrays of x, y."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau = np.where(m00, 1.0 - lam * mu * rho, tau)
    tau = np.where(m01, 1.0 + lam * rho, tau)
    tau = np.where(m10, 1.0 + mu * rho, tau)
    tau = np.where(m11, 1.0 - rho, tau)
    return tau


@dataclass
class DCParams:
    c0: float   # baseline log scoring rate
    c1: float   # Elo-difference -> log goal supremacy slope
    c2: float   # home-advantage log effect
    rho: float  # Dixon-Coles low-score dependence


class DixonColesModel:
    """Elo-driven Dixon-Coles bivariate-Poisson goals model."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.params: DCParams | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, prematch: pd.DataFrame, cutoff: pd.Timestamp) -> "DixonColesModel":
        """Fit [c0, c1, c2, rho] by weighted maximum likelihood."""
        df = prematch
        if self.cfg.fit_min_year is not None:
            df = df[pd.to_datetime(df["date"]).dt.year >= self.cfg.fit_min_year]
        df = df.reset_index(drop=True)

        x = df["home_score"].to_numpy(dtype=float)
        y = df["away_score"].to_numpy(dtype=float)
        d = (df["r_home_pre"].to_numpy() - df["r_away_pre"].to_numpy()) / _ELO_SCALE
        home_flag = (~df["neutral"].to_numpy()).astype(float)

        w = fit_weights(
            df,
            cutoff=cutoff,
            half_life_days=self.cfg.half_life_days,
            friendly_weight=self.cfg.friendly_weight,
        )
        w = w / w.mean()  # normalise so the objective scale is ~O(n)

        # constant (parameter-independent) factorial term, dropped from the
        # optimisation but harmless to keep out.
        const = -(gammaln(x + 1.0) + gammaln(y + 1.0))

        def neg_ll(theta: np.ndarray) -> float:
            c0, c1, c2, rho = theta
            lam = np.exp(c0 + c1 * d + c2 * home_flag)
            mu = np.exp(c0 - c1 * d)
            tau = _dc_tau(x, y, lam, mu, rho)
            tau = np.clip(tau, 1e-10, None)
            ll = (
                -lam + x * np.log(lam)
                - mu + y * np.log(mu)
                + np.log(tau)
                + const
            )
            return -float(np.sum(w * ll))

        x0 = np.array([np.log(1.35), 0.35, 0.25, -0.05])
        bounds = [(-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0), (-0.2, 0.2)]
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
        c0, c1, c2, rho = res.x
        self.params = DCParams(c0=c0, c1=c1, c2=c2, rho=rho)
        self._fit_result = res
        self._n_fit = len(df)
        return self

    # ------------------------------------------------------------------ #
    # Expected goals
    # ------------------------------------------------------------------ #
    def expected_goals(self, r_home, r_away, neutral=True):
        """Return (lambda_home, mu_away) for the given ratings / venue.

        Accepts scalars or numpy arrays (broadcast).
        """
        if self.params is None:
            raise RuntimeError("model is not fit")
        p = self.params
        d = (np.asarray(r_home, dtype=float) - np.asarray(r_away, dtype=float)) / _ELO_SCALE
        home_flag = np.where(np.asarray(neutral), 0.0, 1.0)
        lam = np.exp(p.c0 + p.c1 * d + p.c2 * home_flag)
        mu = np.exp(p.c0 - p.c1 * d)
        return lam, mu

    # ------------------------------------------------------------------ #
    # Internals: normalised score-probability matrix
    # ------------------------------------------------------------------ #
    def _score_pmf(self, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Normalised P(x, y) matrix of shape (..., G+1, G+1)."""
        g = self.cfg.max_goals
        ks = np.arange(g + 1)
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        mu = np.atleast_1d(np.asarray(mu, dtype=float))

        # log Poisson pmf over the grid for each match: shape (K, G+1)
        logph = -lam[:, None] + ks[None, :] * np.log(lam[:, None]) - gammaln(ks + 1)[None, :]
        logpa = -mu[:, None] + ks[None, :] * np.log(mu[:, None]) - gammaln(ks + 1)[None, :]
        ph = np.exp(logph)
        pa = np.exp(logpa)

        P = ph[:, :, None] * pa[:, None, :]  # (K, G+1, G+1)

        rho = self.params.rho
        P[:, 0, 0] *= 1.0 - lam * mu * rho
        P[:, 0, 1] *= 1.0 + lam * rho
        P[:, 1, 0] *= 1.0 + mu * rho
        P[:, 1, 1] *= 1.0 - rho
        np.clip(P, 0.0, None, out=P)
        P /= P.sum(axis=(1, 2), keepdims=True)
        return P

    # ------------------------------------------------------------------ #
    # Outcome probabilities (used by the backtest)
    # ------------------------------------------------------------------ #
    def win_draw_loss(self, r_home, r_away, neutral=True):
        """Return (p_home_win, p_draw, p_away_win), vectorised."""
        lam, mu = self.expected_goals(r_home, r_away, neutral)
        P = self._score_pmf(lam, mu)  # (K, G+1, G+1)
        g = self.cfg.max_goals
        idx = np.arange(g + 1)
        home_more = idx[:, None] > idx[None, :]
        away_more = idx[:, None] < idx[None, :]
        draw = idx[:, None] == idx[None, :]
        p_home = (P * home_more).sum(axis=(1, 2))
        p_away = (P * away_more).sum(axis=(1, 2))
        p_draw = (P * draw).sum(axis=(1, 2))
        if np.isscalar(r_home) and np.isscalar(r_away):
            return float(p_home[0]), float(p_draw[0]), float(p_away[0])
        return p_home, p_draw, p_away

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample_scores_fixed(self, lam: float, mu: float, n: int, rng):
        """Sample ``n`` scorelines for a *single* fixture (scalar lam, mu).

        Much cheaper than the array path: builds one cdf and inverts it for
        all n draws.  Used for the group stage where the fixtures are
        identical across simulations.
        """
        P = self._score_pmf(np.array([lam]), np.array([mu]))[0]  # (G+1, G+1)
        flat = P.ravel()
        cdf = np.cumsum(flat)
        cdf[-1] = 1.0
        u = rng.random(n)
        idx = np.searchsorted(cdf, u, side="right")
        g1 = self.cfg.max_goals + 1
        return idx // g1, idx % g1

    def sample_scores_array(self, lam: np.ndarray, mu: np.ndarray, rng):
        """Sample one scoreline per element of lam/mu (one per simulation).

        Used in the knockout rounds where each simulation has a different
        pairing.
        """
        P = self._score_pmf(lam, mu)  # (K, G+1, G+1)
        K = P.shape[0]
        flat = P.reshape(K, -1)
        cdf = np.cumsum(flat, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random(K)
        idx = (cdf >= u[:, None]).argmax(axis=1)
        g1 = self.cfg.max_goals + 1
        return idx // g1, idx % g1

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        if self.params is None:
            return "DixonColesModel(unfit)"
        p = self.params
        # implied neutral-venue scoring for an even matchup
        lam0 = np.exp(p.c0)
        return (
            "Dixon-Coles fit ({} matches)\n"
            "  c0 (log base goals) = {:.4f}  -> {:.3f} goals/side at neutral, even\n"
            "  c1 (Elo->goals)     = {:.4f}  per 100 Elo\n"
            "  c2 (home effect)    = {:.4f}  -> x{:.3f} home goals\n"
            "  rho (DC low-score)  = {:.4f}"
        ).format(self._n_fit, p.c0, lam0, p.c1, p.c2, np.exp(p.c2), p.rho)
