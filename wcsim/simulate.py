"""Vectorised Monte-Carlo tournament engine.

The whole tournament is simulated for all ``N`` runs at once: arrays carry a
leading axis of length N, so a single numpy expression advances every
simulation in lock-step (20,000 tournaments in a few seconds).

Supports both the 48-team 2026 format (12 groups, top-2 + 8 best thirds into
a Round of 32) and the classic 32-team format (8 groups, top-2 into a Round
of 16) - the bracket config decides.  Everything the analysis layer needs is
recorded per simulation: group positions, points, goals, every knockout
pairing and winner, and each team's furthest stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SimConfig
from .match_model import GoalsModel
from .shootout import ShootoutModel
from .tournament import (
    Bracket,
    ROUND_ENTRY_STAGE,
    ROUND_WIN_STAGE,
    STAGE_LABELS,
    ThirdAllocator,
)

_FIXTURES = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]  # 4-team round robin


def _rank_desc(keys: list[np.ndarray]) -> np.ndarray:
    """Per-row ranking (best first); keys ordered lowest-priority first."""
    order = np.lexsort(keys, axis=1)
    return order[:, ::-1]


@dataclass
class SimRecord:
    """Everything recorded across the N simulations."""

    teams: list
    team_group: dict
    ratings: np.ndarray
    group_letters: list
    entry_stage: int
    round_names: list
    n: int
    stage: np.ndarray                 # (n, T) furthest stage code
    pos: np.ndarray                   # (n, T) group finishing position 0..3
    pts: np.ndarray                   # (n, T) group-stage points
    gf: np.ndarray                    # (n, T) group-stage goals for
    ga: np.ndarray                    # (n, T) group-stage goals against
    matches: dict = field(default_factory=dict)  # id -> {round, home, away, winner}


class WorldCupSimulator:
    def __init__(
        self,
        groups: pd.DataFrame,
        ratings: pd.Series,
        model: GoalsModel,
        bracket: Bracket,
        cfg: SimConfig,
        rng: np.random.Generator,
        shootout: ShootoutModel | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.bracket = bracket
        self.rng = rng
        self.shootout = shootout or ShootoutModel(cfg)

        self.teams = list(groups["team"])
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.n_teams = len(self.teams)
        self.group_letters = sorted(groups["group"].unique())
        self.letter_col = {g: c for c, g in enumerate(self.group_letters)}

        self.group_idx: dict[str, np.ndarray] = {}
        self.team_group: dict[str, str] = {}
        for g, sub in groups.groupby("group"):
            self.group_idx[g] = np.array([self.team_to_idx[t] for t in sub["team"]])
            for t in sub["team"]:
                self.team_group[t] = g

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
        self.hosts = set(cfg.host_teams) & set(self.teams)
        if cfg.host_boost:
            for host in self.hosts:
                self.ratings[self.team_to_idx[host]] += cfg.host_boost

        self.allocator = (
            ThirdAllocator(bracket) if bracket.third_slots else None
        )

    # ------------------------------------------------------------------ #
    def run(self, n: int | None = None) -> "SimResults":
        n = n or self.cfg.n_sims
        cfg, model, rng = self.cfg, self.model, self.rng
        arange = np.arange(n)
        T = self.n_teams
        G = len(self.group_letters)
        entry_stage = ROUND_ENTRY_STAGE[self.bracket.entry_round]

        stage = np.zeros((n, T), dtype=np.int8)
        pos_all = np.zeros((n, T), dtype=np.int8)
        pts_all = np.zeros((n, T), dtype=np.int16)
        gf_all = np.zeros((n, T), dtype=np.int16)
        ga_all = np.zeros((n, T), dtype=np.int16)

        winners: dict[str, np.ndarray] = {}
        runners: dict[str, np.ndarray] = {}
        thirds_team = np.empty((n, G), dtype=np.int32)
        thirds_pts = np.empty((n, G), dtype=np.int16)
        thirds_gd = np.empty((n, G), dtype=np.int16)
        thirds_gf = np.empty((n, G), dtype=np.int16)

        # ---- group stage --------------------------------------------- #
        for col, g in enumerate(self.group_letters):
            gidx = self.group_idx[g]
            r = self.ratings[gidx]
            pts = np.zeros((n, 4), dtype=np.int16)
            gf = np.zeros((n, 4), dtype=np.int16)
            ga = np.zeros((n, 4), dtype=np.int16)
            hh = np.zeros((n, 4, 4), dtype=np.int16)

            for i, j in _FIXTURES:
                home_loc, away_loc, neutral = i, j, True
                if cfg.host_group_home:
                    i_host = self.teams[gidx[i]] in self.hosts
                    j_host = self.teams[gidx[j]] in self.hosts
                    if i_host != j_host:
                        neutral = False
                        if j_host:
                            home_loc, away_loc = j, i
                gh, gag = model.sample_scores_fixed(
                    float(r[home_loc]), float(r[away_loc]), neutral, n, rng
                )
                ph = np.where(gh > gag, 3, np.where(gh == gag, 1, 0)).astype(np.int16)
                pa = np.where(gag > gh, 3, np.where(gh == gag, 1, 0)).astype(np.int16)
                pts[:, home_loc] += ph
                pts[:, away_loc] += pa
                gf[:, home_loc] += gh
                ga[:, home_loc] += gag
                gf[:, away_loc] += gag
                ga[:, away_loc] += gh
                hh[:, home_loc, away_loc] = ph
                hh[:, away_loc, home_loc] = pa

            gd = gf - ga
            # head-to-head: mini-league points among teams tied on the prior
            # criteria (points, GD and GF) - faithful, simplified FIFA step.
            same_tie = (
                (pts[:, :, None] == pts[:, None, :])
                & (gd[:, :, None] == gd[:, None, :])
                & (gf[:, :, None] == gf[:, None, :])
            )
            h2h = (hh * same_tie).sum(axis=2)
            rnd = rng.random((n, 4))

            order = _rank_desc([rnd, h2h, gf, gd, pts])  # best first
            for k in range(4):
                pos_all[arange, gidx[order[:, k]]] = k
            pts_all[:, gidx] = pts
            gf_all[:, gidx] = gf
            ga_all[:, gidx] = ga

            winners[g] = gidx[order[:, 0]]
            runners[g] = gidx[order[:, 1]]
            t_loc = order[:, 2]
            thirds_team[:, col] = gidx[t_loc]
            thirds_pts[:, col] = np.take_along_axis(pts, t_loc[:, None], 1)[:, 0]
            thirds_gd[:, col] = np.take_along_axis(gd, t_loc[:, None], 1)[:, 0]
            thirds_gf[:, col] = np.take_along_axis(gf, t_loc[:, None], 1)[:, 0]

        # ---- qualifier slots ------------------------------------------ #
        slot_team: dict[str, np.ndarray] = {}
        for g in self.group_letters:
            slot_team[f"W_{g}"] = winners[g]
            slot_team[f"R_{g}"] = runners[g]

        n_thirds = self.bracket.n_third_slots
        if n_thirds:
            rndg = rng.random((n, G))
            order3 = _rank_desc([rndg, thirds_gf, thirds_gd, thirds_pts])
            topq = order3[:, :n_thirds]            # qualifying group columns

            if self.bracket.ranked_third_slots:    # legacy rank-based slots
                for k, slot in enumerate(self.bracket.ranked_third_slots):
                    slot_team[slot] = thirds_team[arange, topq[:, k]]
            else:                                   # official constrained slots
                masks = np.bitwise_or.reduce(1 << topq.astype(np.int32), axis=1)
                for slot in self.allocator.slot_names:
                    slot_team[slot] = np.empty(n, dtype=np.int32)
                for mask in np.unique(masks):
                    sel = masks == mask
                    qual = frozenset(
                        self.group_letters[c] for c in range(G) if mask >> c & 1
                    )
                    assign = self.allocator.assign(qual)
                    for slot, letter in assign.items():
                        col = self.letter_col[letter]
                        slot_team[slot][sel] = thirds_team[sel, col]

        for arr in slot_team.values():
            stage[arange, arr] = np.maximum(stage[arange, arr], entry_stage)

        # ---- knockout rounds ------------------------------------------ #
        matches: dict[str, dict] = {}
        match_winner: dict[str, np.ndarray] = {}

        def resolve(slot: str) -> np.ndarray:
            return slot_team[slot] if slot in slot_team else match_winner[slot]

        for rname in self.bracket.round_names:
            win_stage = ROUND_WIN_STAGE[rname]
            for m in self.bracket.rounds[rname]:
                h_idx = resolve(m["home"])
                a_idx = resolve(m["away"])
                rh = self.ratings[h_idx]
                ra = self.ratings[a_idx]
                gh, gag = model.sample_scores_array(rh, ra, True, rng)
                home_win = gh > gag
                draw = gh == gag
                home_so = rng.random(n) < self.shootout.home_win_prob(rh, ra)
                winner = np.where(home_win | (draw & home_so), h_idx, a_idx)

                match_winner[m["id"]] = winner
                stage[arange, winner] = win_stage
                matches[m["id"]] = {
                    "round": rname,
                    "home": h_idx.astype(np.int16),
                    "away": a_idx.astype(np.int16),
                    "winner": winner.astype(np.int16),
                }

        rec = SimRecord(
            teams=self.teams,
            team_group=self.team_group,
            ratings=self.ratings,
            group_letters=self.group_letters,
            entry_stage=entry_stage,
            round_names=self.bracket.round_names,
            n=n,
            stage=stage,
            pos=pos_all,
            pts=pts_all,
            gf=gf_all,
            ga=ga_all,
            matches=matches,
        )
        return SimResults(rec)


class SimResults:
    """Aggregated per-team stage probabilities."""

    def __init__(self, rec: SimRecord):
        self.rec = rec

    def table(self) -> pd.DataFrame:
        rec = self.rec
        stage, n = rec.stage, rec.n
        names = {1: "R32_%", 2: "R16_%", 3: "QF_%", 4: "SF_%", 5: "Final_%", 6: "Champion_%"}
        stage_cols = [(s, names[s]) for s in range(rec.entry_stage, 7)]

        reach = {s: (stage >= s).mean(axis=0) for s, _ in stage_cols}
        modal = np.array(
            [np.bincount(stage[:, i], minlength=7).argmax() for i in range(len(rec.teams))]
        )
        champ = (stage >= 6).mean(axis=0)
        champ_se = np.sqrt(champ * (1 - champ) / n)

        rows = []
        for i, team in enumerate(rec.teams):
            row = {
                "team": team,
                "group": rec.team_group[team],
                "elo": round(float(rec.ratings[i]), 1),
            }
            for s, cname in stage_cols:
                row[cname] = 100 * reach[s][i]
            row["Champion_se"] = 100 * champ_se[i]
            row["most_likely_finish"] = STAGE_LABELS[int(modal[i])]
            rows.append(row)
        sort_cols = [c for _, c in stage_cols[::-1]] + ["elo"]
        return (
            pd.DataFrame(rows)
            .sort_values(sort_cols, ascending=False)
            .reset_index(drop=True)
        )
