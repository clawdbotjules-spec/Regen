"""Sensitivity analysis and parameter-uncertainty quantification.

Two questions, answered empirically:

1. *Knob sensitivity* - how much do the headline championship probabilities
   move when the modeling choices change?  Re-runs the simulation under a
   grid of variations (time-decay half-life, friendly weight, model family,
   shootout treatment, host advantages) and reports the spread per team.

2. *Parameter uncertainty* - even with the knobs fixed, the Dixon-Coles
   parameters are estimates.  We sample parameter vectors from the
   asymptotic normal approximation N(theta_hat, H^-1) (L-BFGS inverse
   Hessian) and re-simulate, giving a confidence band that reflects how
   well the goals model itself is pinned down.

Monte-Carlo noise (finite simulation count) is reported alongside so the
three uncertainty sources can be compared.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .config import SimConfig
from .match_model import DixonColesModel, make_model
from .shootout import ShootoutModel
from .simulate import WorldCupSimulator
from .tournament import load_bracket


def _champion_pct(groups, ratings, model, bracket, cfg, shootout, seed, n) -> pd.Series:
    rng = np.random.default_rng(seed)
    sim = WorldCupSimulator(groups, ratings, model, bracket, cfg, rng, shootout=shootout)
    rec = sim.run(n).rec
    pct = 100 * (rec.stage >= 6).mean(axis=0)
    return pd.Series(pct, index=rec.teams)


def run_sensitivity(
    groups: pd.DataFrame,
    ratings: pd.Series,
    prematch: pd.DataFrame,
    shootouts: pd.DataFrame | None,
    cfg: SimConfig,
    fit_cutoff: pd.Timestamp,
    top_n: int = 10,
) -> pd.DataFrame:
    """Champion% spread across modeling variations.  Returns the full table."""
    variations: list[tuple[str, dict]] = [
        ("baseline", {}),
        ("half-life 1.5y", {"half_life_days": 540.0}),
        ("half-life 6y", {"half_life_days": 2190.0}),
        ("friendlies x0.25", {"friendly_weight": 0.25}),
        ("friendlies x1.0", {"friendly_weight": 1.0}),
        ("model: DC only", {"model_name": "dc"}),
        ("model: BivPois", {"model_name": "bp"}),
        ("model: NegBin", {"model_name": "nb"}),
        ("shootout: coin flip", {"shootout_model": "weighted", "shootout_elo_weight": 0.0}),
        ("shootout: full Elo", {"shootout_model": "weighted", "shootout_elo_weight": 1.0}),
        ("hosts neutral in groups", {"host_group_home": False}),
        ("host boost +50 Elo", {"host_boost": 50.0}),
    ]

    print(f"\nRunning {len(variations)} variations x {cfg.sens_sims:,} sims each ...")
    cols = {}
    for label, over in variations:
        cfg_v = dataclasses.replace(cfg, **over)
        bracket = load_bracket(cfg_v.bracket_path, set(groups["group"]))
        model = make_model(cfg_v.model_name, cfg_v).fit(prematch, cutoff=fit_cutoff)
        so = ShootoutModel(cfg_v).fit(shootouts, prematch)
        cols[label] = _champion_pct(
            groups, ratings, model, bracket, cfg_v, so, cfg.seed, cfg.sens_sims
        )
        print(f"  done: {label}")

    table = pd.DataFrame(cols)
    table["min"] = table.min(axis=1)
    table["max"] = table.max(axis=1)
    table = table.sort_values("baseline", ascending=False)

    show = table.head(top_n)
    print("\n=== Champion % under each variation (top teams) ===")
    print(show.to_string(float_format=lambda v: f"{v:5.1f}"))

    mc_se = np.sqrt(show["baseline"] / 100 * (1 - show["baseline"] / 100) / cfg.sens_sims) * 100
    print("\nSpread (max-min) vs Monte-Carlo noise (+-2 SE at this sim count):")
    for team in show.index:
        print(
            f"  {team:<15} spread {show.loc[team, 'max'] - show.loc[team, 'min']:5.1f} pp"
            f"   MC noise +-{2 * mc_se[team]:.1f} pp"
        )
    return table


def parameter_uncertainty(
    groups: pd.DataFrame,
    ratings: pd.Series,
    prematch: pd.DataFrame,
    shootouts: pd.DataFrame | None,
    cfg: SimConfig,
    fit_cutoff: pd.Timestamp,
    n_draws: int = 16,
    top_n: int = 8,
) -> pd.DataFrame:
    """Champion% bands from Dixon-Coles parameter uncertainty."""
    dc = DixonColesModel(cfg).fit(prematch, cutoff=fit_cutoff)
    if dc.cov is None:
        print("[warn] no parameter covariance available; skipping")
        return pd.DataFrame()

    bracket = load_bracket(cfg.bracket_path, set(groups["group"]))
    so = ShootoutModel(cfg).fit(shootouts, prematch)
    rng = np.random.default_rng(cfg.seed + 1)
    draws = rng.multivariate_normal(dc.theta, dc.cov, size=n_draws)
    draws[:, 3] = np.clip(draws[:, 3], -0.2, 0.2)  # keep rho in its valid range

    n_each = max(cfg.sens_sims // 2, 1000)
    print(
        f"\nSampling {n_draws} Dixon-Coles parameter vectors from N(theta, H^-1), "
        f"{n_each:,} sims each ..."
    )
    runs = []
    for k, theta in enumerate(draws):
        dc_k = DixonColesModel(cfg)
        dc_k.theta = theta
        dc_k._n_fit = dc._n_fit
        runs.append(
            _champion_pct(groups, ratings, dc_k, bracket, cfg, so, cfg.seed + 2 + k, n_each)
        )
    mat = pd.concat(runs, axis=1)

    out = pd.DataFrame(
        {
            "mean_%": mat.mean(axis=1),
            "p5_%": mat.quantile(0.05, axis=1),
            "p95_%": mat.quantile(0.95, axis=1),
        }
    ).sort_values("mean_%", ascending=False)

    print("\n=== Champion % band from goals-model parameter uncertainty ===")
    print(out.head(top_n).to_string(float_format=lambda v: f"{v:5.1f}"))
    print(
        "(Bands mix parameter uncertainty with Monte-Carlo noise at "
        f"{n_each:,} sims/draw; the asymptotic-normal approximation is "
        "indicative, not exact. Elo-rating uncertainty is NOT included.)"
    )
    return out
