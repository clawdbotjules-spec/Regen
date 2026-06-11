"""Tournament structure: bracket config + stage bookkeeping.

The group-stage / advancement logic itself is implemented (vectorised) in
``simulate.py``; this module just loads and validates the configurable
knockout bracket and defines the stage labels shared across the project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Stage codes for "furthest stage reached" (higher = further).
STAGE_GROUP = 0      # eliminated in the group stage
STAGE_R32 = 1        # reached the Round of 32 (qualified from group)
STAGE_R16 = 2        # reached the Round of 16
STAGE_QF = 3         # reached the Quarter-final
STAGE_SF = 4         # reached the Semi-final
STAGE_FINAL = 5      # reached the Final (runner-up if they lose it)
STAGE_CHAMPION = 6   # won the Final

STAGE_LABELS = {
    STAGE_GROUP: "Group stage",
    STAGE_R32: "Round of 32",
    STAGE_R16: "Round of 16",
    STAGE_QF: "Quarter-final",
    STAGE_SF: "Semi-final",
    STAGE_FINAL: "Final",
    STAGE_CHAMPION: "Champion",
}

# The stage a team is recorded as having *reached* by WINNING a match in a
# given knockout round (e.g. winning an R32 match means you reached the R16).
ROUND_WIN_STAGE = {
    "R32": STAGE_R16,
    "R16": STAGE_QF,
    "QF": STAGE_SF,
    "SF": STAGE_FINAL,
    "F": STAGE_CHAMPION,
}

ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]


@dataclass
class Bracket:
    rounds: dict          # round name -> list of {id, home, away}
    raw: dict             # original JSON

    def all_match_ids(self) -> set:
        ids = set()
        for matches in self.rounds.values():
            for m in matches:
                ids.add(m["id"])
        return ids


def load_bracket(path: str) -> Bracket:
    """Load and validate the knockout bracket JSON."""
    with open(path) as fh:
        raw = json.load(fh)
    rounds = raw["rounds"]

    for r in ROUND_ORDER:
        if r not in rounds:
            raise ValueError(f"bracket is missing round '{r}'")

    counts = {r: len(rounds[r]) for r in ROUND_ORDER}
    expected = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1}
    if counts != expected:
        raise ValueError(f"bracket round sizes {counts} != expected {expected}")

    # validate slot references
    match_ids = {m["id"] for ms in rounds.values() for m in ms}
    valid_group_slots = (
        {f"W_{g}" for g in "ABCDEFGHIJKL"}
        | {f"R_{g}" for g in "ABCDEFGHIJKL"}
        | {f"T_{k}" for k in range(1, 9)}
    )
    for rname in ROUND_ORDER:
        for m in rounds[rname]:
            for side in ("home", "away"):
                slot = m[side]
                if slot not in valid_group_slots and slot not in match_ids:
                    raise ValueError(
                        f"match {m['id']} references unknown slot '{slot}'"
                    )
    return Bracket(rounds=rounds, raw=raw)
