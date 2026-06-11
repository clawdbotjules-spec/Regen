"""Central configuration for the World Cup simulator.

Every tunable knob lives here so the rest of the code stays declarative.
Values can be overridden from the CLI (see ``main.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# --------------------------------------------------------------------------- #
# Match-importance tiers (World Football Elo "weight index" K-factor).
# These are the conventional eloratings.net values.
# --------------------------------------------------------------------------- #
def match_importance(tournament: str) -> float:
    """Base Elo K-factor for a match, by competition tier.

    Convention (eloratings.net):
        60  World Cup finals
        50  Continental championship finals & Confederations Cup
        40  World Cup / continental qualifiers, Nations League
        30  All other tournaments
        20  Friendlies
    """
    t = (tournament or "").lower()
    if t == "friendly":
        return 20.0
    if "world cup" in t and "qualification" not in t:
        return 60.0
    if "qualification" in t or "qualifier" in t:
        return 40.0
    if "nations league" in t:
        return 40.0
    continental_finals = (
        "uefa euro",
        "copa am",          # Copa América (with/without accent)
        "african cup of nations",
        "afc asian cup",
        "gold cup",
        "concacaf championship",
        "confederations",
        "oceania nations",
    )
    if any(c in t for c in continental_finals):
        # only the finals, not the qualifiers (qualifiers handled above)
        return 50.0
    return 30.0


def goal_difference_multiplier(goal_diff: int) -> float:
    """Elo K-factor multiplier for margin of victory (eloratings.net)."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    # g == 3 -> 1.75, g == 4 -> 1.875, ... == 1.75 + (g-3)/8
    return (11.0 + g) / 8.0


@dataclass
class SimConfig:
    """All simulator parameters in one place."""

    # ---- I/O -------------------------------------------------------------- #
    data_dir: str = "data"
    output_path: str = "output/predictions.csv"
    output_dir: str = "output"          # extra analysis CSVs land here
    bracket_path: str = "config/bracket.json"

    # ---- Reproducibility / scale ----------------------------------------- #
    # 100k sims take ~15s and push Monte-Carlo noise well below model
    # uncertainty (champion-prob SE ~0.13pp for the favourite); 20k is fine
    # for quick iteration.
    seed: int = 42
    n_sims: int = 100_000

    # ---- Elo -------------------------------------------------------------- #
    elo_initial: float = 1500.0
    elo_home_advantage: float = 100.0   # Elo points added to a non-neutral host
    elo_cutoff: str | None = None       # ISO date; ratings as-of this date (None = all data)

    # ---- Goals models ------------------------------------------------------ #
    # half_life_days / friendly_weight / slope_scale were grid-searched on
    # the 1998-2014 World Cups (wcsim/tune.py, `--tune`), keeping 2018/2022
    # as a held-out test set.  The validation surface is flat (the whole
    # grid spans ~0.003 log-loss) and the argmin's gain did not transfer to
    # the test set, so we apply a one-SE-style rule: keep the prior weights
    # (validation is indifferent) and adopt only the one consistent signal -
    # every top validation combo has slope_scale >= 1.1, i.e. the globally
    # fit Elo->goals slope is slightly too shallow at World Cup level - at
    # its most conservative value.
    model_name: str = "ensemble"        # dc | bp | nb | ensemble
    half_life_days: float = 1095.0      # time-decay half-life for the goals fit (~3 yrs)
    friendly_weight: float = 0.5        # down-weight friendlies in the goals fit
    slope_scale: float = 1.1            # calibration multiplier on the Elo->goals slope
    fit_min_year: int | None = 1990     # ignore very old matches when fitting goals
    max_goals: int = 12                 # truncation of the score grid (per side)

    # ---- Tournament / 2026 specifics ------------------------------------- #
    host_boost: float = 0.0             # extra Elo for USA/Canada/Mexico in WC matches
    host_teams: tuple = ("United States", "Canada", "Mexico")
    host_group_home: bool = True        # hosts play their group matches at home venues
    shootout_model: str = "empirical"   # empirical (logistic fit on shootouts.csv) | weighted
    shootout_elo_weight: float = 0.5    # for shootout_model=weighted: 0 = coin flip, 1 = Elo

    # ---- Validation ------------------------------------------------------- #
    backtest_years: tuple = (2018, 2022)
    backtest_sims: int = 10_000         # sims per tournament-level backtest
    sens_sims: int = 4_000              # sims per sensitivity-analysis variation

    def __post_init__(self) -> None:
        if self.n_sims <= 0:
            raise ValueError("n_sims must be positive")
        if not 0.0 <= self.shootout_elo_weight <= 1.0:
            raise ValueError("shootout_elo_weight must be in [0, 1]")
        if self.model_name not in ("dc", "bp", "nb", "ensemble"):
            raise ValueError(f"unknown model '{self.model_name}'")
        if self.shootout_model not in ("empirical", "weighted"):
            raise ValueError(f"unknown shootout model '{self.shootout_model}'")


# Canonical team-name aliases: maps names that appear in groups.csv (or other
# feeds) onto the canonical names used in the martj42 results.csv history.
TEAM_ALIASES: Dict[str, str] = {
    "Curacao": "Curaçao",
    "Cape Verde Islands": "Cape Verde",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Republic of Ireland": "Republic of Ireland",
}
