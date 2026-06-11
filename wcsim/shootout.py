"""Penalty-shootout model.

Instead of an arbitrary "coin flip weighted slightly by strength", the
``empirical`` model *measures* how much team strength matters in shootouts:
it joins ``shootouts.csv`` with the Elo walk's pre-match ratings and fits a
one-parameter logistic by maximum likelihood::

    P(home wins shootout) = 1 / (1 + exp(-b * dElo / 100))

The fitted slope ``b`` is small (shootouts really are close to coin flips),
and the data decides exactly how small.  The ``weighted`` model is the old
hand-tuned interpolation between a coin flip and the Elo-implied win
probability, kept for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .config import SimConfig


class ShootoutModel:
    """P(home side wins a penalty shootout) as a function of Elo diff."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.mode = cfg.shootout_model
        self.slope: float | None = None     # empirical logistic slope per 100 Elo
        self.n_fit = 0

    # ------------------------------------------------------------------ #
    def fit(
        self,
        shootouts: pd.DataFrame | None,
        prematch: pd.DataFrame,
        cutoff: pd.Timestamp | None = None,
    ) -> "ShootoutModel":
        """Fit the empirical slope on shootout history before ``cutoff``."""
        if self.mode != "empirical":
            return self
        if shootouts is None or shootouts.empty:
            print("[warn] no shootouts.csv; falling back to weighted shootout model")
            self.mode = "weighted"
            return self

        s = shootouts.dropna(subset=["winner"])
        if cutoff is not None:
            s = s[s["date"] < cutoff]

        # attach the Elo walk's pre-match ratings via (date, home, away)
        pm = prematch[["date", "home_team", "away_team", "r_home_pre", "r_away_pre"]]
        merged = s.merge(pm, on=["date", "home_team", "away_team"], how="inner")
        merged = merged[
            merged["winner"].isin(merged["home_team"])
            | merged["winner"].isin(merged["away_team"])
        ]
        if len(merged) < 100:
            print(
                f"[warn] only {len(merged)} usable shootouts; "
                "falling back to weighted shootout model"
            )
            self.mode = "weighted"
            return self

        d = (merged["r_home_pre"] - merged["r_away_pre"]).to_numpy() / 100.0
        y = (merged["winner"] == merged["home_team"]).to_numpy().astype(float)

        def neg_ll(b: float) -> float:
            p = 1.0 / (1.0 + np.exp(-b * d))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize_scalar(neg_ll, bounds=(-2.0, 2.0), method="bounded")
        self.slope = float(res.x)
        self.n_fit = len(merged)
        return self

    # ------------------------------------------------------------------ #
    def home_win_prob(self, r_home: np.ndarray, r_away: np.ndarray) -> np.ndarray:
        d = (np.asarray(r_home, dtype=float) - np.asarray(r_away, dtype=float))
        if self.mode == "empirical" and self.slope is not None:
            p = 1.0 / (1.0 + np.exp(-self.slope * d / 100.0))
        else:
            elo_p = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
            p = 0.5 + self.cfg.shootout_elo_weight * (elo_p - 0.5)
        return np.clip(p, 0.02, 0.98)

    def summary(self) -> str:
        if self.mode == "empirical" and self.slope is not None:
            p100 = 1.0 / (1.0 + np.exp(-self.slope))
            return (
                f"Shootouts: empirical logistic on {self.n_fit} historical shootouts; "
                f"slope={self.slope:.4f}/100Elo -> a +100 Elo side wins {100 * p100:.1f}%"
            )
        return (
            f"Shootouts: weighted coin flip "
            f"(elo weight={self.cfg.shootout_elo_weight:.2f})"
        )
