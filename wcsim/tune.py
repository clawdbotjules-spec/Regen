"""Out-of-sample hyperparameter tuning for the goals model.

The simulator's accuracy-critical hyperparameters - the time-decay
half-life, the friendly down-weight, and a calibration multiplier on the
Elo->goals slope - were originally hand-picked.  This module tunes them
properly:

    VALIDATION SET : the 1998, 2002, 2006, 2010 and 2014 World Cups
                     (320 matches; for each, training uses only matches
                     strictly before that tournament)
    TEST SET       : the 2018 and 2022 World Cups - never touched during
                     selection, evaluated once afterwards by validate.py

For every (half-life, friendly-weight) pair a Dixon-Coles model is fit per
validation tournament; the slope multiplier is then swept at prediction
time (cheap - no refit needed).  The combination with the best pooled
validation log-loss wins.  Because every model family shares the same
Elo->goals link, the tuned values are applied to all of them.

Why a slope multiplier?  The global fit is dominated by qualifiers and
mismatches; World Cup finals matches are tighter (the field is stronger
and coaches play differently), so the slope that best fits "all
international football" may be too steep or too shallow at a World Cup.
The multiplier lets validation data decide.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .config import SimConfig
from .elo import compute_elo
from .match_model import DixonColesModel
from .validate import _log_loss, _outcome_code, _ratings_lookup

VAL_YEARS = (1998, 2002, 2006, 2010, 2014)

HALF_LIVES = (365.0, 730.0, 1095.0, 1460.0, 2190.0, 2920.0)
FRIENDLY_WEIGHTS = (0.25, 0.5, 1.0)
SLOPE_SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def tune(
    results: pd.DataFrame,
    cfg: SimConfig,
    val_years: tuple = VAL_YEARS,
    half_lives: tuple = HALF_LIVES,
    friendly_weights: tuple = FRIENDLY_WEIGHTS,
    slope_scales: tuple = SLOPE_SCALES,
) -> tuple[dict, pd.DataFrame]:
    """Grid-search on the validation World Cups.

    Returns (best parameter dict, full results table).
    """
    # ---- per-tournament fixed quantities (Elo walk done once per cutoff) --
    folds = []
    for year in val_years:
        wc = results[
            (results["tournament"] == "FIFA World Cup")
            & (results["date"].dt.year == year)
        ]
        if wc.empty:
            print(f"[warn] no data for WC {year}; dropped from validation")
            continue
        cutoff = wc["date"].min()
        ratings, prematch = compute_elo(results, cfg, cutoff=cutoff)
        folds.append(
            {
                "year": year,
                "cutoff": cutoff,
                "prematch": prematch,
                "rh": _ratings_lookup(ratings, wc["home_team"].to_numpy(), cfg.elo_initial),
                "ra": _ratings_lookup(ratings, wc["away_team"].to_numpy(), cfg.elo_initial),
                "neutral": wc["neutral"].to_numpy(),
                "y": _outcome_code(
                    wc["home_score"].to_numpy(), wc["away_score"].to_numpy()
                ),
            }
        )
    n_total = sum(len(f["y"]) for f in folds)
    print(
        f"Validation set: {len(folds)} World Cups "
        f"({', '.join(str(f['year']) for f in folds)}), {n_total} matches"
    )
    print(
        f"Grid: {len(half_lives)} half-lives x {len(friendly_weights)} friendly "
        f"weights x {len(slope_scales)} slope scales "
        f"({len(half_lives) * len(friendly_weights)} fits/tournament) ..."
    )

    rows = []
    for hl in half_lives:
        for fw in friendly_weights:
            cfg_v = dataclasses.replace(
                cfg, half_life_days=hl, friendly_weight=fw, slope_scale=1.0
            )
            # log-loss accumulators per slope scale
            ll_sum = {ss: 0.0 for ss in slope_scales}
            for f in folds:
                model = DixonColesModel(cfg_v).fit(f["prematch"], cutoff=f["cutoff"])
                for ss in slope_scales:
                    model.slope_scale = ss
                    ph, pdr, pa = model.win_draw_loss(
                        f["rh"], f["ra"], neutral=f["neutral"]
                    )
                    probs = np.stack([ph, pdr, pa], axis=1)
                    ll_sum[ss] += _log_loss(probs, f["y"]) * len(f["y"])
            for ss in slope_scales:
                rows.append(
                    {
                        "half_life": hl,
                        "friendly_weight": fw,
                        "slope_scale": ss,
                        "val_logloss": ll_sum[ss] / n_total,
                    }
                )
            print(f"  done: half-life {hl:.0f}d, friendlies x{fw}")

    table = pd.DataFrame(rows).sort_values("val_logloss").reset_index(drop=True)
    best = table.iloc[0]
    best_params = {
        "half_life_days": float(best["half_life"]),
        "friendly_weight": float(best["friendly_weight"]),
        "slope_scale": float(best["slope_scale"]),
    }

    print("\n=== Top 10 combinations by pooled validation log-loss ===")
    print(table.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    base = table[
        (table["half_life"] == 1095.0)
        & (table["friendly_weight"] == 0.5)
        & (table["slope_scale"] == 1.0)
    ]
    if not base.empty:
        print(
            f"\nOld defaults (hl=1095, fw=0.5, ss=1.0): "
            f"val log-loss {float(base['val_logloss'].iloc[0]):.4f}"
        )
    print(
        f"Selected      (hl={best_params['half_life_days']:.0f}, "
        f"fw={best_params['friendly_weight']}, "
        f"ss={best_params['slope_scale']}): "
        f"val log-loss {float(best['val_logloss']):.4f}"
    )
    print(
        "\nNOTE: selection used ONLY 1998-2014. The 2018/2022 backtest "
        "(--validate) remains a clean test set."
    )
    return best_params, table
