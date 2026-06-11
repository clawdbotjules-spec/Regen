"""World-Football-Elo ratings.

Walks the match history chronologically and updates a rating for every
national team using the standard eloratings.net conventions:

    * expected score  We = 1 / (1 + 10 ** (-dr / 400))
      where dr = R_home - R_away (+ home advantage if not neutral)
    * K-factor        K  = importance(tournament) * gd_multiplier(margin)
    * update          R' = R + K * (W - We)        (W in {1, 0.5, 0})

It also records, for every processed match, the *pre-match* ratings of both
sides.  Those pre-match ratings are the strength feature consumed by the
Dixon-Coles goals model, so the goals model is trained on the same dynamic
ratings the simulator later uses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .config import SimConfig, goal_difference_multiplier
from .data import importance_array


def compute_elo(
    results: pd.DataFrame,
    cfg: SimConfig,
    cutoff: Optional[pd.Timestamp] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute Elo ratings by walking matches in chronological order.

    Parameters
    ----------
    results : cleaned results frame (see data.load_results).
    cfg     : simulator configuration.
    cutoff  : if given, only matches strictly before this date are used
              (for backtesting "ratings as of date X").

    Returns
    -------
    ratings : pd.Series indexed by team -> final Elo rating, descending.
    prematch : pd.DataFrame with one row per processed match, columns
               [date, home_team, away_team, home_score, away_score,
                neutral, tournament, r_home_pre, r_away_pre].
    """
    df = results
    if cutoff is not None:
        df = df[df["date"] < cutoff]
    df = df.reset_index(drop=True)

    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy()
    base_k = importance_array(df)
    dates = df["date"].to_numpy()
    tourn = df["tournament"].to_numpy()
    is_friendly = df["is_friendly"].to_numpy()

    ratings: dict[str, float] = defaultdict(lambda: cfg.elo_initial)
    ha = cfg.elo_home_advantage

    n = len(df)
    r_home_pre = np.empty(n)
    r_away_pre = np.empty(n)

    for i in range(n):
        rh = ratings[home[i]]
        ra = ratings[away[i]]
        r_home_pre[i] = rh
        r_away_pre[i] = ra

        dr = rh - ra + (0.0 if neutral[i] else ha)
        we_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        if hs[i] > as_[i]:
            w_home = 1.0
        elif hs[i] == as_[i]:
            w_home = 0.5
        else:
            w_home = 0.0

        k = base_k[i] * goal_difference_multiplier(hs[i] - as_[i])
        delta = k * (w_home - we_home)
        ratings[home[i]] = rh + delta
        ratings[away[i]] = ra - delta

    ratings_series = pd.Series(ratings, name="elo").sort_values(ascending=False)

    prematch = pd.DataFrame(
        {
            "date": dates,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
            "neutral": neutral,
            "tournament": tourn,
            "is_friendly": is_friendly,
            "r_home_pre": r_home_pre,
            "r_away_pre": r_away_pre,
        }
    )
    return ratings_series, prematch
