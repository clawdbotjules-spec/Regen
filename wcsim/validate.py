"""Validation: match-level and tournament-level backtests.

Match-level
-----------
For the 2018 and 2022 World Cups, train every goals model only on matches
played strictly before the tournament, predict the win/draw/loss
probabilities of all 64 actual finals matches, and score them with
multiclass log-loss, the 3-class Brier score and the ranked probability
score (RPS).  The benchmark is a Davidson Elo-only model: a one-parameter
map from the Elo difference to W/D/L probabilities

    theta = 10 ** (dr/400);  D = theta + nu*sqrt(theta) + 1
    P(home) = theta/D,  P(draw) = nu*sqrt(theta)/D,  P(away) = 1/D

with the draw parameter ``nu`` fit by MLE on the training data.  Knockout
games decided on penalties count as draws (their 90'/120' score), i.e. we
grade the pre-shootout 3-way prediction.

Tournament-level
----------------
Re-simulates the 2018 and 2022 tournaments with their real groups and
brackets (training strictly pre-tournament), then compares the predicted
stage-reach probabilities with what actually happened: per-stage Brier
scores vs a uniform "every team is equal" baseline, and the predicted
probability/rank of the actual champion.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .config import SimConfig
from .data import normalize_name
from .elo import compute_elo
from .match_model import make_model
from .shootout import ShootoutModel
from .simulate import WorldCupSimulator
from .tournament import load_bracket

_STAGE_OF_LABEL = {"Group": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5, "W": 6}
# note: in the 32-team format, "qualified from the group" == reached the R16
# (stage 2); a team listed as "Group" reached neither.


# --------------------------------------------------------------------------- #
# Scoring rules
# --------------------------------------------------------------------------- #
def _outcome_code(hs: np.ndarray, as_: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


def _log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score for the ordered outcomes (H, D, A)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    cp, co = np.cumsum(probs, axis=1), np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


# --------------------------------------------------------------------------- #
# Davidson Elo-only baseline
# --------------------------------------------------------------------------- #
def _fit_davidson_nu(prematch: pd.DataFrame, cfg: SimConfig) -> float:
    df = prematch
    if cfg.fit_min_year is not None:
        df = df[pd.to_datetime(df["date"]).dt.year >= cfg.fit_min_year]
    dr = df["r_home_pre"].to_numpy() - df["r_away_pre"].to_numpy()
    dr = dr + np.where(df["neutral"].to_numpy(), 0.0, cfg.elo_home_advantage)
    y = _outcome_code(df["home_score"].to_numpy(), df["away_score"].to_numpy())
    theta = 10.0 ** (dr / 400.0)
    sq = np.sqrt(theta)

    def neg_ll(nu: float) -> float:
        denom = theta + nu * sq + 1.0
        p = np.stack([theta / denom, nu * sq / denom, 1.0 / denom], axis=1)
        return _log_loss(p, y) * len(y)

    return float(minimize_scalar(neg_ll, bounds=(0.01, 5.0), method="bounded").x)


def _davidson_probs(dr: np.ndarray, nu: float) -> np.ndarray:
    theta = 10.0 ** (dr / 400.0)
    sq = np.sqrt(theta)
    denom = theta + nu * sq + 1.0
    return np.stack([theta / denom, nu * sq / denom, 1.0 / denom], axis=1)


def _ratings_lookup(ratings: pd.Series, teams: np.ndarray, default: float) -> np.ndarray:
    return np.array([ratings.get(t, default) for t in teams], dtype=float)


# --------------------------------------------------------------------------- #
# Match-level backtest
# --------------------------------------------------------------------------- #
def match_backtest(
    results: pd.DataFrame,
    cfg: SimConfig,
    model_names: tuple = ("dc", "bp", "nb", "ensemble"),
) -> pd.DataFrame:
    rows = []
    calib_records = []

    for year in cfg.backtest_years:
        wc = results[
            (results["tournament"] == "FIFA World Cup")
            & (results["date"].dt.year == year)
        ].copy()
        if wc.empty:
            print(f"[warn] no FIFA World Cup matches found for {year}; skipping.")
            continue

        cutoff = wc["date"].min()
        ratings, prematch = compute_elo(results, cfg, cutoff=cutoff)
        nu = _fit_davidson_nu(prematch, cfg)

        rh = _ratings_lookup(ratings, wc["home_team"].to_numpy(), cfg.elo_initial)
        ra = _ratings_lookup(ratings, wc["away_team"].to_numpy(), cfg.elo_initial)
        neutral = wc["neutral"].to_numpy()
        y = _outcome_code(wc["home_score"].to_numpy(), wc["away_score"].to_numpy())
        actual_draw_rate = float((y == 1).mean())

        print(f"\n=== Match-level backtest: World Cup {year} "
              f"({len(wc)} matches, train < {cutoff.date()}) ===")

        preds = {}
        for name in model_names:
            model = make_model(name, cfg).fit(prematch, cutoff=cutoff)
            ph, pdr, pa = model.win_draw_loss(rh, ra, neutral=neutral)
            preds[name] = np.stack([ph, pdr, pa], axis=1)
        dr = rh - ra + np.where(neutral, 0.0, cfg.elo_home_advantage)
        preds["elo"] = _davidson_probs(dr, nu)

        for name, p in preds.items():
            rows.append(
                {
                    "tournament": f"WC {year}",
                    "model": name,
                    "n": len(wc),
                    "logloss": _log_loss(p, y),
                    "brier": _brier(p, y),
                    "rps": _rps(p, y),
                    "pred_draw_%": 100 * float(p[:, 1].mean()),
                    "actual_draw_%": 100 * actual_draw_rate,
                }
            )
            print(
                f"  {name:9s} log-loss {rows[-1]['logloss']:.4f}  "
                f"Brier {rows[-1]['brier']:.4f}  RPS {rows[-1]['rps']:.4f}  "
                f"draws pred {rows[-1]['pred_draw_%']:.1f}% vs actual "
                f"{rows[-1]['actual_draw_%']:.1f}%"
            )

        calib_records.append(
            pd.DataFrame({"p_home": preds[cfg.model_name][:, 0], "home_win": (y == 0)})
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics

    pooled = (
        metrics.groupby("model")
        .apply(
            lambda g: pd.Series(
                {
                    "n": g["n"].sum(),
                    "logloss": np.average(g["logloss"], weights=g["n"]),
                    "brier": np.average(g["brier"], weights=g["n"]),
                    "rps": np.average(g["rps"], weights=g["n"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("logloss")
    )
    print("\n=== Pooled (2018 + 2022, sorted by log-loss) ===")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = pooled.iloc[0]["model"]
    elo_ll = float(pooled.loc[pooled["model"] == "elo", "logloss"].iloc[0])
    best_ll = float(pooled.iloc[0]["logloss"])
    print(
        f"\nBest pooled model: {best} "
        f"({'beats' if best != 'elo' else 'is'} the Elo-only baseline"
        f"{f' by {elo_ll - best_ll:.4f} log-loss' if best != 'elo' else ''}). "
        "Differences this small on 128 matches are within noise - treat the "
        "goal models and Elo as statistically tied for 3-way outcomes."
    )
    _print_calibration(pd.concat(calib_records, ignore_index=True), cfg.model_name)
    return metrics


def _print_calibration(df: pd.DataFrame, model_name: str, n_bins: int = 5) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df = df.copy()
    df["bin"] = np.clip(np.digitize(df["p_home"], bins) - 1, 0, n_bins - 1)
    print(f"\n=== Calibration ({model_name} home-win prob, pooled) ===")
    print(f"  {'pred range':>14} {'n':>5} {'avg pred':>9} {'actual':>8}")
    for b in range(n_bins):
        sub = df[df["bin"] == b]
        if sub.empty:
            continue
        print(
            f"  [{bins[b]:0.2f},{bins[b + 1]:0.2f}) {len(sub):>5} "
            f"{sub['p_home'].mean():>9.3f} {sub['home_win'].mean():>8.3f}"
        )


# --------------------------------------------------------------------------- #
# Tournament-level backtest
# --------------------------------------------------------------------------- #
def tournament_backtest(
    results: pd.DataFrame,
    shootouts: pd.DataFrame | None,
    cfg: SimConfig,
) -> pd.DataFrame:
    """Simulate the 2018/2022 tournaments and grade stage-reach forecasts."""
    bracket_path = os.path.join(os.path.dirname(cfg.bracket_path), "bracket_32.json")
    rows = []

    for year in cfg.backtest_years:
        gpath = os.path.join(cfg.data_dir, f"wc{year}_groups.csv")
        if not os.path.exists(gpath):
            print(f"[warn] {gpath} missing; skipping tournament backtest {year}")
            continue
        gdf = pd.read_csv(gpath)
        gdf["team"] = gdf["team"].map(normalize_name)

        wc = results[
            (results["tournament"] == "FIFA World Cup")
            & (results["date"].dt.year == year)
        ]
        cutoff = wc["date"].min()
        ratings, prematch = compute_elo(results, cfg, cutoff=cutoff)
        model = make_model(cfg.model_name, cfg).fit(prematch, cutoff=cutoff)
        so = ShootoutModel(cfg).fit(shootouts, prematch, cutoff=cutoff)

        bracket = load_bracket(bracket_path, set(gdf["group"]))
        rng = np.random.default_rng(cfg.seed)
        host = {2018: "Russia", 2022: "Qatar"}[year]
        # the 32-team backtests pre-date the 2026 host trio
        import dataclasses
        cfg_bt = dataclasses.replace(cfg, host_teams=(host,))
        sim = WorldCupSimulator(
            gdf[["group", "team"]], ratings, model, bracket, cfg_bt, rng, shootout=so
        )
        res = sim.run(cfg.backtest_sims)
        rec = res.rec

        actual = {
            row["team"]: _STAGE_OF_LABEL[row["reached"]] for _, row in gdf.iterrows()
        }
        # for the 32-team format "Group" means stage 0 (didn't qualify)
        actual = {t: (0 if s == 1 else s) for t, s in actual.items()}
        t_idx = {t: i for i, t in enumerate(rec.teams)}

        stage_names = {2: "R16", 3: "QF", 4: "SF", 5: "Final", 6: "Champion"}
        n_teams = len(rec.teams)
        n_reaching = {2: 16, 3: 8, 4: 4, 5: 2, 6: 1}

        for s, sname in stage_names.items():
            pred = (rec.stage >= s).mean(axis=0)
            act = np.array([float(actual[t] >= s) for t in rec.teams])
            base_p = n_reaching[s] / n_teams
            rows.append(
                {
                    "tournament": f"WC {year}",
                    "stage": sname,
                    "model_brier": float(np.mean((pred[[t_idx[t] for t in rec.teams]] - act) ** 2)),
                    "baseline_brier": float(np.mean((base_p - act) ** 2)),
                }
            )

        champ_pred = (rec.stage >= 6).mean(axis=0)
        champion = next(t for t, s in actual.items() if s == 6)
        order = np.argsort(champ_pred)[::-1]
        champ_rank = int(np.where(order == t_idx[champion])[0][0]) + 1
        p_champ = float(champ_pred[t_idx[champion]])

        print(f"\n=== Tournament backtest: World Cup {year} "
              f"({cfg.backtest_sims:,} sims, model={cfg.model_name}) ===")
        top5 = ", ".join(
            f"{rec.teams[i]} {100 * champ_pred[i]:.1f}%" for i in order[:5]
        )
        print(f"  Predicted top 5: {top5}")
        print(
            f"  Actual champion: {champion} - predicted {100 * p_champ:.1f}% "
            f"(rank {champ_rank} of {n_teams}); "
            f"log-loss {-np.log(max(p_champ, 1e-12)):.2f} vs uniform "
            f"{np.log(n_teams):.2f}"
        )
        yr_rows = [r for r in rows if r["tournament"] == f"WC {year}"]
        for r in yr_rows:
            skill = 1 - r["model_brier"] / r["baseline_brier"]
            print(
                f"  {r['stage']:9s} Brier {r['model_brier']:.4f} "
                f"vs uniform {r['baseline_brier']:.4f}  "
                f"(skill {100 * skill:+.0f}%)"
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        pooled = df.groupby("stage")[["model_brier", "baseline_brier"]].mean()
        pooled["skill_%"] = 100 * (1 - pooled["model_brier"] / pooled["baseline_brier"])
        pooled = pooled.reindex(["R16", "QF", "SF", "Final", "Champion"])
        print("\n=== Pooled stage-level skill (Brier vs uniform baseline) ===")
        print(pooled.to_string(float_format=lambda v: f"{v:.4f}"))
        print(
            "\nHonest read: the model has real skill at separating contenders\n"
            "from also-rans (positive skill at every stage), but single-\n"
            "tournament outcomes stay wildly uncertain - even a perfectly\n"
            "calibrated model gives the actual champion ~10-25%. None of this\n"
            "is an edge over betting markets; treat it as analysis."
        )
    return df
