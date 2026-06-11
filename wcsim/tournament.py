"""Tournament structure: bracket config, stages, third-place allocation.

Bracket JSON schema (config/bracket.json):

    {
      "entry_round": "R32",                # or "R16" for the 32-team format
      "rounds": {
        "R32": [ {"id": "M73", "home": "W_A", "away": "3RD_CEFGH"}, ... ],
        "R16": [ {"id": "M89", "home": "M74", "away": "M77"}, ... ],
        ...
      }
    }

Slot grammar:
    W_<g>      winner of group <g>
    R_<g>      runner-up of group <g>
    3RD_<set>  a third-placed team from one of the listed groups
               (e.g. 3RD_ABCDF) - allocated per simulation, see below
    T_<k>      k-th best third overall (legacy rank-based slots)
    M<id>      winner of an earlier match

Third-place allocation
----------------------
FIFA assigns the 8 qualified thirds to the constrained ``3RD_*`` slots based
only on WHICH groups they come from (not their rank).  We mirror that: for
each combination of 8 qualifying groups (at most C(12,8)=495), we compute a
deterministic assignment - walking the slots in bracket order and giving each
the alphabetically-first eligible group that still leaves a perfect matching
for the remaining slots (checked with Kuhn's bipartite-matching algorithm).
Assignments are cached per combination, so the per-simulation cost is a
table lookup.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

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

ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]

ROUND_ENTRY_STAGE = {"R32": STAGE_R32, "R16": STAGE_R16}

# Winning a match in round X means you have *reached* the next stage.
ROUND_WIN_STAGE = {
    "R32": STAGE_R16,
    "R16": STAGE_QF,
    "QF": STAGE_SF,
    "SF": STAGE_FINAL,
    "F": STAGE_CHAMPION,
}

_EXPECTED_SIZES = {
    "R32": {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1},
    "R16": {"R16": 8, "QF": 4, "SF": 2, "F": 1},
}

_THIRD_CONSTRAINED = re.compile(r"^3RD_([A-L]+)$")
_THIRD_RANKED = re.compile(r"^T_([1-8])$")


@dataclass
class Bracket:
    entry_round: str
    rounds: dict                       # round name -> list of {id, home, away}
    raw: dict
    third_slots: dict = field(default_factory=dict)   # slot string -> frozenset(groups)
    ranked_third_slots: list = field(default_factory=list)  # ["T_1", ...]

    @property
    def round_names(self) -> list[str]:
        start = ROUND_ORDER.index(self.entry_round)
        return ROUND_ORDER[start:]

    @property
    def n_third_slots(self) -> int:
        return len(self.third_slots) + len(self.ranked_third_slots)


def load_bracket(path: str, groups_present: set[str] | None = None) -> Bracket:
    """Load and validate a knockout bracket JSON."""
    with open(path) as fh:
        raw = json.load(fh)
    entry = raw.get("entry_round", "R32")
    if entry not in _EXPECTED_SIZES:
        raise ValueError(f"unsupported entry_round '{entry}'")
    rounds = raw["rounds"]

    expected = _EXPECTED_SIZES[entry]
    counts = {r: len(rounds.get(r, [])) for r in expected}
    if counts != expected:
        raise ValueError(f"bracket round sizes {counts} != expected {expected}")

    match_ids = {m["id"] for r in expected for m in rounds[r]}
    third_slots: dict[str, frozenset] = {}
    ranked: list[str] = []

    for rname in expected:
        for m in rounds[rname]:
            for side in ("home", "away"):
                slot = m[side]
                if slot in match_ids:
                    continue
                if _THIRD_RANKED.match(slot):
                    ranked.append(slot)
                    continue
                cm = _THIRD_CONSTRAINED.match(slot)
                if cm:
                    allowed = frozenset(cm.group(1))
                    if groups_present is not None and not allowed <= groups_present:
                        raise ValueError(
                            f"slot {slot}: groups {allowed - groups_present} not in draw"
                        )
                    third_slots[slot] = allowed
                    continue
                if re.match(r"^[WR]_[A-L]$", slot):
                    if groups_present is not None and slot[2] not in groups_present:
                        raise ValueError(f"slot {slot}: group not in draw")
                    continue
                raise ValueError(f"match {m['id']} references unknown slot '{slot}'")

    if third_slots and ranked:
        raise ValueError("bracket mixes constrained (3RD_) and ranked (T_) third slots")
    n_thirds = len(third_slots) + len(ranked)
    if entry == "R32" and n_thirds != 8:
        raise ValueError(f"48-team bracket needs exactly 8 third slots, found {n_thirds}")
    if entry == "R16" and n_thirds != 0:
        raise ValueError("32-team bracket must not contain third-place slots")

    return Bracket(
        entry_round=entry,
        rounds={r: rounds[r] for r in expected},
        raw=raw,
        third_slots=third_slots,
        ranked_third_slots=sorted(ranked, key=lambda s: int(s.split("_")[1])),
    )


# --------------------------------------------------------------------------- #
# Third-place allocation: constrained bipartite assignment
# --------------------------------------------------------------------------- #
def _kuhn_matching(allowed: list[frozenset], groups: tuple) -> dict | None:
    """Maximum bipartite matching slot->group; None unless perfect."""
    match_of_group: dict[str, int] = {}

    def try_slot(s: int, visited: set) -> bool:
        for g in allowed[s]:
            if g not in groups or g in visited:
                continue
            visited.add(g)
            if g not in match_of_group or try_slot(match_of_group[g], visited):
                match_of_group[g] = s
                return True
        return False

    for s in range(len(allowed)):
        if not try_slot(s, set()):
            return None
    return {s: g for g, s in match_of_group.items()}


class ThirdAllocator:
    """Deterministic group->slot assignment for the qualified thirds.

    The assignment depends only on the *set* of qualifying groups (as in
    FIFA's official allocation tables), so results are cached per
    combination.
    """

    def __init__(self, bracket: Bracket):
        # fixed slot order = bracket (match) order
        self.slot_names = [
            m[side]
            for m in bracket.rounds[bracket.entry_round]
            for side in ("home", "away")
            if m[side] in bracket.third_slots
        ]
        self.allowed = [bracket.third_slots[s] for s in self.slot_names]
        self._assign = lru_cache(maxsize=None)(self._assign_uncached)

    def _assign_uncached(self, qualified: tuple) -> dict:
        """slot name -> group letter for one combination of qualified groups."""
        remaining = set(qualified)
        out: dict[str, str] = {}
        pending = list(range(len(self.slot_names)))
        for pos, si in enumerate(pending):
            rest = pending[pos + 1:]
            chosen = None
            for g in sorted(self.allowed[si] & remaining):
                rem = tuple(sorted(remaining - {g}))
                if not rest or _kuhn_matching([self.allowed[j] for j in rest], rem):
                    chosen = g
                    break
            if chosen is None:
                # No feasible perfect matching from here (can only happen if
                # the slot constraint sets are inconsistent with this
                # combination): fall back to any eligible group, else any.
                pool = sorted(self.allowed[si] & remaining) or sorted(remaining)
                chosen = pool[0]
            out[self.slot_names[si]] = chosen
            remaining.discard(chosen)
        return out

    def assign(self, qualified_groups: frozenset) -> dict:
        return self._assign(tuple(sorted(qualified_groups)))
