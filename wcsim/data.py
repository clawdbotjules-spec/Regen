"""Data loading, cleaning and weighting.

Reads the raw CSVs (martj42 "International football results" dataset),
normalises team names, drops un-played fixtures, and provides the
exponential time-decay / competition weighting used by the goals model.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import TEAM_ALIASES, match_importance


def normalize_name(name: str) -> str:
    """Trim whitespace and apply the canonical alias map."""
    if not isinstance(name, str):
        return name
    n = name.strip()
    return TEAM_ALIASES.get(n, n)


def load_results(
    data_dir: str = "data",
    filename: str = "results.csv",
) -> pd.DataFrame:
    """Load and clean the historical results table.

    Returns a frame with columns:
        date (datetime), home_team, away_team, home_score (int),
        away_score (int), tournament, neutral (bool), is_friendly (bool).
    Rows without a recorded score (future/scheduled fixtures) are dropped.
    """
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()

    df["home_team"] = df["home_team"].map(normalize_name)
    df["away_team"] = df["away_team"].map(normalize_name)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # `neutral` may arrive as bool or as the strings "TRUE"/"FALSE".
    if df["neutral"].dtype != bool:
        df["neutral"] = (
            df["neutral"].astype(str).str.strip().str.upper().map(
                {"TRUE": True, "FALSE": False}
            ).fillna(False)
        )

    df["is_friendly"] = df["tournament"].str.lower().eq("friendly")
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def load_groups(data_dir: str = "data", filename: str = "groups.csv") -> pd.DataFrame:
    """Load the 48-team / 12-group draw and normalise team names."""
    path = os.path.join(data_dir, filename)
    g = pd.read_csv(path)
    g["team"] = g["team"].map(normalize_name)
    g["group"] = g["group"].astype(str).str.strip()
    n_groups = g["group"].nunique()
    if len(g) != 48 or n_groups != 12:
        print(
            f"[warn] groups.csv has {len(g)} teams in {n_groups} groups "
            f"(expected 48 in 12)."
        )
    return g


def load_shootouts(
    data_dir: str = "data", filename: str = "shootouts.csv"
) -> Optional[pd.DataFrame]:
    """Load the (optional) penalty-shootout history; None if absent."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    for col in ("home_team", "away_team", "winner"):
        if col in s.columns:
            s[col] = s[col].map(normalize_name)
    return s


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def time_decay_weights(
    dates: pd.Series, cutoff: pd.Timestamp, half_life_days: float
) -> np.ndarray:
    """Exponential time-decay weight: 0.5 ** (age_days / half_life_days).

    Matches on the cutoff date get weight 1.0; a match one half-life older
    gets 0.5, two half-lives 0.25, and so on.
    """
    age_days = (cutoff - dates).dt.total_seconds().to_numpy() / 86400.0
    age_days = np.clip(age_days, 0.0, None)
    return np.power(0.5, age_days / float(half_life_days))


def competition_weights(df: pd.DataFrame, friendly_weight: float) -> np.ndarray:
    """Down-weight friendlies relative to competitive matches."""
    w = np.ones(len(df), dtype=float)
    w[df["is_friendly"].to_numpy()] = friendly_weight
    return w


def fit_weights(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    half_life_days: float,
    friendly_weight: float,
) -> np.ndarray:
    """Combined weight for the goals-model MLE: time-decay x competition."""
    return time_decay_weights(df["date"], cutoff, half_life_days) * competition_weights(
        df, friendly_weight
    )


def importance_array(df: pd.DataFrame) -> np.ndarray:
    """Vectorised Elo base K-factor per match (cached over unique tournaments)."""
    cache = {t: match_importance(t) for t in df["tournament"].unique()}
    return df["tournament"].map(cache).to_numpy(dtype=float)
