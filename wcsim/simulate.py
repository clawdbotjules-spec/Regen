"""Vectorised Monte-Carlo simulation of the 2026 World Cup.

The whole tournament is simulated for all ``N`` runs at once: arrays carry a
leading axis of length N, so a single numpy expression advances every
simulation in lock-step.  This makes 20 000 tournaments run in seconds.

Flow
----
1. Group stage - 12 groups x 6 fixtures.  Fixtures are identical across
   simulations, so each is sampled with one cheap inverse-cdf draw of N
   scorelines.  Standings use points -> GD -> GF -> head-to-head -> random.
2. "8 best third-placed teams" ranked across all 12 groups by
   points -> GD -> GF -> random.
3. Knockout - the configurable bracket is walked round by round; each match
   slot is a different pairing per simulation, sampled with the array path.
   Drawn knockout games go to an Elo-weighted shootout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimConfig
from .match_model import DixonColesModel
from .tournament import (
    Bracket,
    ROUND_ORDER,
    ROUND_WIN_STAGE,
    STAGE_LABELS,
    STAGE_R32,
)

_FIXTURES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_GROUP_LETTERS = list("ABCDEFGHIJKL")


def _rank_desc(keys: list[np.ndarray]) -> np.ndarray:
    """Return per-row ranking (best first) given ascending-priority keys.

    ``keys`` is ordered lowest-priority first; the last key is primary.
    Sorts each row of a (N, k) block independently and reverses to descending.
    """
    order = np.lexsort(keys, axis=1)   # ascending: worst first
    return order[:, ::-1]              # best first


class WorldCupSimulator:
    def __init__(
        self,
        groups: pd.DataFrame,
        ratings: pd.Series,
        model: DixonColesModel,
        bracket: Bracket,
        cfg: SimConfig,
        rng: np.random.Generator,
    ):
        self.cfg = cfg
        self.model = model
        self.bracket = bracket
        self.rng = rng

        # Stable team index over all 48 teams.
        self.teams = list(groups["team"])
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.n_teams = len(self.teams)

        # group letter -> 4 global team indices (draw order preserved)
        self.group_idx: dict[str, list[int]] = {}
        self.team_group: dict[str, str] = {}
        for g, sub in groups.groupby("group"):
            idxs = [self.team_to_idx[t] for t in sub["team"]]
            self.group_idx[g] = idxs
            for t in sub["team"]:
                self.team_group[t] = g

        # Ratings array (+ host boost), warning for missing teams.
        self.ratings = np.full(self.n_teams, cfg.elo_initial, dtype=float)
        missing = []
        for t, i in self.team_to_idx.items():
            if t in ratings.index:
                self.ratings[i] = ratings[t]
            else:
                missing.append(t)
        if missing:
            print(
                "[warn] no Elo rating from history for: "
                + ", ".join(missing)
                + f" (defaulting to {cfg.elo_initial:.0f})"
            )
        if cfg.host_boost:
            for host in cfg.host_teams:
                if host in self.team_to_idx:
                    self.ratings[self.team_to_idx[host]] += cfg.host_boost

    # ------------------------------------------------------------------ #
    def _shootout_home_prob(self, rh: np.ndarray, ra: np.ndarray) -> np.ndarray:
        """Elo-weighted coin flip for a drawn knockout game."""
        elo_p = 1.0 / (1.0 + 10.0 ** (-(rh - ra) / 400.0))
        p = 0.5 + self.cfg.shootout_elo_weight * (elo_p - 0.5)
        return np.clip(p, 0.02, 0.98)

    # ------------------------------------------------------------------ #
    def run(self, n: int | None = None) -> "SimResults":
        n = n or self.cfg.n_sims
        rng = self.rng
        model = self.model
        arange = np.arange(n)

        # ---- group stage --------------------------------------------- #
        winners: dict[str, np.ndarray] = {}
        runners: dict[str, np.ndarray] = {}
        thirds_team = np.empty((n, 12), dtype=np.int32)
        thirds_pts = np.empty((n, 12), dtype=np.int32)
        thirds_gd = np.empty((n, 12), dtype=np.int32)
        thirds_gf = np.empty((n, 12), dtype=np.int32)

        for col, g in enumerate(_GROUP_LETTERS):
            gidx = np.array(self.group_idx[g])
            r = self.ratings[gidx]
            pts = np.zeros((n, 4), dtype=np.int32)
            gf = np.zeros((n, 4), dtype=np.int32)
            ga = np.zeros((n, 4), dtype=np.int32)
            hh = np.zeros((n, 4, 4), dtype=np.int32)

            for i, j in _FIXTURES:
                lam, mu = model.expected_goals(r[i], r[j], neutral=True)
                gh, gag = model.sample_scores_fixed(float(lam), float(mu), n, rng)
                pi = np.where(gh > gag, 3, np.where(gh == gag, 1, 0)).astype(np.int32)
                pj = np.where(gag > gh, 3, np.where(gh == gag, 1, 0)).astype(np.int32)
                pts[:, i] += pi
                pts[:, j] += pj
                gf[:, i] += gh
                ga[:, i] += gag
                gf[:, j] += gag
                ga[:, j] += gh
                hh[:, i, j] = pi
                hh[:, j, i] = pj

            gd = gf - ga
            # head-to-head: mini-league points among teams tied on the prior
            # criteria (points, GD and GF). This key only changes the order of
            # teams that are otherwise exactly level, so it is a faithful
            # (if simplified) version of FIFA's head-to-head step.
            same_tie = (
                (pts[:, :, None] == pts[:, None, :])
                & (gd[:, :, None] == gd[:, None, :])
                & (gf[:, :, None] == gf[:, None, :])
            )                                               # (n,4,4)
            h2h = (hh * same_tie).sum(axis=2)               # (n,4)
            rnd = rng.random((n, 4))

            # priority (last = primary): pts > gd > gf > h2h > random
            order = _rank_desc([rnd, h2h, gf, gd, pts])     # (n,4) best first
            w_loc, r_loc, t_loc = order[:, 0], order[:, 1], order[:, 2]

            winners[g] = gidx[w_loc]
            runners[g] = gidx[r_loc]
            thirds_team[:, col] = gidx[t_loc]
            thirds_pts[:, col] = np.take_along_axis(pts, t_loc[:, None], 1)[:, 0]
            thirds_gd[:, col] = np.take_along_axis(gd, t_loc[:, None], 1)[:, 0]
            thirds_gf[:, col] = np.take_along_axis(gf, t_loc[:, None], 1)[:, 0]

        # ---- 8 best third-placed teams ------------------------------- #
        rnd12 = rng.random((n, 12))
        # priority: pts > gd > gf > random
        order3 = _rank_desc([rnd12, thirds_gf, thirds_gd, thirds_pts])  # best first
        top8 = order3[:, :8]                                            # (n,8)
        T = [thirds_team[arange, top8[:, k]] for k in range(8)]

        # ---- assemble qualifier slots -------------------------------- #
        slot_team: dict[str, np.ndarray] = {}
        for g in _GROUP_LETTERS:
            slot_team[f"W_{g}"] = winners[g]
            slot_team[f"R_{g}"] = runners[g]
        for k in range(8):
            slot_team[f"T_{k + 1}"] = T[k]

        stage = np.zeros((n, self.n_teams), dtype=np.int8)
        for arr in slot_team.values():
            stage[arange, arr] = STAGE_R32

        # ---- knockout ------------------------------------------------ #
        match_winner: dict[str, np.ndarray] = {}

        def resolve(slot: str) -> np.ndarray:
            return slot_team[slot] if slot in slot_team else match_winner[slot]

        for rname in ROUND_ORDER:
            win_stage = ROUND_WIN_STAGE[rname]
            for m in self.bracket.rounds[rname]:
                h_idx = resolve(m["home"])
                a_idx = resolve(m["away"])
                rh = self.ratings[h_idx]
                ra = self.ratings[a_idx]
                lam, mu = model.expected_goals(rh, ra, neutral=True)
                gh, gag = model.sample_scores_array(lam, mu, rng)

                home_win = gh > gag
                draw = gh == gag
                home_so = rng.random(n) < self._shootout_home_prob(rh, ra)
                winner = np.where(home_win | (draw & home_so), h_idx, a_idx)

                match_winner[m["id"]] = winner
                stage[arange, winner] = win_stage

        return SimResults(self, stage, n)


class SimResults:
    """Aggregated per-team stage probabilities."""

    def __init__(self, sim: WorldCupSimulator, stage: np.ndarray, n: int):
        self.sim = sim
        self.stage = stage
        self.n = n

    def table(self) -> pd.DataFrame:
        sim = self.sim
        stage = self.stage
        n = self.n

        # cumulative "reached stage s" probabilities
        reach = {s: (stage >= s).mean(axis=0) for s in range(1, 7)}
        # modal finishing stage per team
        modal = np.array(
            [np.bincount(stage[:, i], minlength=7).argmax() for i in range(sim.n_teams)]
        )

        rows = []
        for i, team in enumerate(sim.teams):
            rows.append(
                {
                    "team": team,
                    "group": sim.team_group[team],
                    "elo": round(float(sim.ratings[i]), 1),
                    "R32_%": 100 * reach[1][i],
                    "R16_%": 100 * reach[2][i],
                    "QF_%": 100 * reach[3][i],
                    "SF_%": 100 * reach[4][i],
                    "Final_%": 100 * reach[5][i],
                    "Champion_%": 100 * reach[6][i],
                    "most_likely_finish": STAGE_LABELS[int(modal[i])],
                }
            )
        df = pd.DataFrame(rows).sort_values(
            ["Champion_%", "Final_%", "SF_%", "elo"], ascending=False
        ).reset_index(drop=True)
        return df
