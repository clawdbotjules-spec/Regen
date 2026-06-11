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
    bracket_path: str = "config/bracket.json"

    # ---- Reproducibility / scale ----------------------------------------- #
    seed: int = 42
    n_sims: int = 20_000

    # ---- Elo -------------------------------------------------------------- #
    elo_initial: float = 1500.0
    elo_home_advantage: float = 100.0   # Elo points added to a non-neutral host
    elo_cutoff: str | None = None       # ISO date; ratings as-of this date (None = all data)

    # ---- Dixon-Coles goals model ----------------------------------------- #
    half_life_days: float = 1095.0      # time-decay half-life for the goals fit (~3 yrs)
    friendly_weight: float = 0.5        # down-weight friendlies in the goals fit
    fit_min_year: int | None = 1990     # ignore very old matches when fitting goals
    max_goals: int = 12                 # truncation of the score grid (per side)

    # ---- Tournament / 2026 specifics ------------------------------------- #
    host_boost: float = 0.0             # extra Elo for USA/Canada/Mexico in WC matches
    host_teams: tuple = ("United States", "Canada", "Mexico")
    shootout_elo_weight: float = 0.5    # 0 == pure coin flip, 1 == full Elo-implied

    # ---- Validation ------------------------------------------------------- #
    backtest_years: tuple = (2018, 2022)

    def __post_init__(self) -> None:
        if self.n_sims <= 0:
            raise ValueError("n_sims must be positive")
        if not 0.0 <= self.shootout_elo_weight <= 1.0:
            raise ValueError("shootout_elo_weight must be in [0, 1]")


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
