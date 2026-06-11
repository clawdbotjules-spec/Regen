"""Round-by-round analysis of the simulation record.

Everything here is pure aggregation over the arrays recorded by the
simulator - no additional modeling.  Outputs:

    * per-group forecast tables (position probabilities, expected pts/goals)
    * exact finishing-stage distribution per team ("every possibility")
    * most likely pairings for every knockout match slot
    * per-team most likely opponents by round (conditional on playing)
    * pairwise P(meet at some point) matrix
    * most likely Finals and Champion/runner-up pairs
    * conditional advancement (win rate given the round is reached)
    * headline joint probabilities (hosts, confederations, seeds)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .simulate import SimRecord
from .tournament import STAGE_LABELS


# --------------------------------------------------------------------------- #
def group_stage_table(rec: SimRecord) -> pd.DataFrame:
    """Per team: position probabilities, advancement, expected pts/goals."""
    n = rec.n
    rows = []
    for i, team in enumerate(rec.teams):
        pos_counts = np.bincount(rec.pos[:, i], minlength=4) / n
        advanced = rec.stage[:, i] >= rec.entry_stage
        third = rec.pos[:, i] == 2
        third_n = int(third.sum())
        rows.append(
            {
                "team": team,
                "group": rec.team_group[team],
                "elo": round(float(rec.ratings[i]), 1),
                "P_1st_%": 100 * pos_counts[0],
                "P_2nd_%": 100 * pos_counts[1],
                "P_3rd_%": 100 * pos_counts[2],
                "P_4th_%": 100 * pos_counts[3],
                "P_advance_%": 100 * advanced.mean(),
                "P_adv_as_3rd_%": 100 * (third & advanced).mean(),
                "P_qual_if_3rd_%": (
                    100 * (third & advanced).sum() / third_n if third_n else np.nan
                ),
                "exp_pts": rec.pts[:, i].mean(),
                "exp_gf": rec.gf[:, i].mean(),
                "exp_ga": rec.ga[:, i].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["group", "P_advance_%"], ascending=[True, False]
    ).reset_index(drop=True)


def stage_distribution(rec: SimRecord) -> pd.DataFrame:
    """Exact P(finishing at each stage) per team - sums to 100% per row."""
    n = rec.n
    rows = []
    for i, team in enumerate(rec.teams):
        counts = np.bincount(rec.stage[:, i], minlength=7) / n
        row = {"team": team, "group": rec.team_group[team]}
        for s in range(7):
            if s == 0 or s >= rec.entry_stage:
                row[f"out_{STAGE_LABELS[s]}_%"] = 100 * counts[s]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(f"out_{STAGE_LABELS[6]}_%", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def matchup_table(rec: SimRecord, round_name: str, top_k: int = 5) -> pd.DataFrame:
    """Most likely pairings for every match slot of one knockout round."""
    n = rec.n
    rows = []
    for mid, m in rec.matches.items():
        if m["round"] != round_name:
            continue
        code = m["home"].astype(np.int32) * len(rec.teams) + m["away"]
        vals, counts = np.unique(code, return_counts=True)
        order = np.argsort(counts)[::-1][:top_k]
        for rank, oi in enumerate(order, 1):
            h, a = divmod(int(vals[oi]), len(rec.teams))
            rows.append(
                {
                    "match": mid,
                    "rank": rank,
                    "pairing": f"{rec.teams[h]} vs {rec.teams[a]}",
                    "prob_%": 100 * counts[oi] / n,
                }
            )
    return pd.DataFrame(rows)


def opponents_by_round(rec: SimRecord, top_k: int = 3) -> pd.DataFrame:
    """Per team and round: P(play that round) and likeliest opponents."""
    n, T = rec.n, len(rec.teams)
    rows = []
    for rname in rec.round_names:
        # opponent matrix for this round: counts[t, o]
        counts = np.zeros((T, T), dtype=np.int64)
        for m in rec.matches.values():
            if m["round"] != rname:
                continue
            np.add.at(counts, (m["home"], m["away"]), 1)
            np.add.at(counts, (m["away"], m["home"]), 1)
        played = counts.sum(axis=1)
        for i, team in enumerate(rec.teams):
            if played[i] == 0:
                continue
            top = np.argsort(counts[i])[::-1][:top_k]
            opps = "; ".join(
                f"{rec.teams[o]} ({100 * counts[i, o] / played[i]:.0f}%)"
                for o in top
                if counts[i, o] > 0
            )
            rows.append(
                {
                    "team": team,
                    "round": rname,
                    "P_play_round_%": 100 * played[i] / n,
                    "likely_opponents (given played)": opps,
                }
            )
    return pd.DataFrame(rows)


def meet_matrix(rec: SimRecord) -> pd.DataFrame:
    """P(team A and team B meet in any knockout match)."""
    T = len(rec.teams)
    counts = np.zeros((T, T), dtype=np.int64)
    for m in rec.matches.values():
        np.add.at(counts, (m["home"], m["away"]), 1)
        np.add.at(counts, (m["away"], m["home"]), 1)
    return pd.DataFrame(
        100 * counts / rec.n, index=rec.teams, columns=rec.teams
    ).round(2)


def finals_table(rec: SimRecord, top_k: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(most likely Finals pairings, most likely Champion/runner-up pairs)."""
    final_id = [mid for mid, m in rec.matches.items() if m["round"] == "F"][0]
    m = rec.matches[final_id]
    T = len(rec.teams)

    lo = np.minimum(m["home"], m["away"]).astype(np.int32)
    hi = np.maximum(m["home"], m["away"]).astype(np.int32)
    vals, counts = np.unique(lo * T + hi, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    pairs = pd.DataFrame(
        [
            {
                "final": f"{rec.teams[v // T]} vs {rec.teams[v % T]}",
                "prob_%": 100 * c / rec.n,
            }
            for v, c in ((int(vals[o]), counts[o]) for o in order)
        ]
    )

    loser = np.where(m["winner"] == m["home"], m["away"], m["home"]).astype(np.int32)
    vals2, counts2 = np.unique(m["winner"].astype(np.int32) * T + loser, return_counts=True)
    order2 = np.argsort(counts2)[::-1][:top_k]
    outcomes = pd.DataFrame(
        [
            {
                "outcome": f"{rec.teams[v // T]} beat {rec.teams[v % T]}",
                "prob_%": 100 * c / rec.n,
            }
            for v, c in ((int(vals2[o]), counts2[o]) for o in order2)
        ]
    )
    return pairs, outcomes


def conditional_advancement(rec: SimRecord) -> pd.DataFrame:
    """P(win the round | reached the round), per team and knockout round."""
    from .tournament import ROUND_ENTRY_STAGE, ROUND_WIN_STAGE

    rows = []
    for i, team in enumerate(rec.teams):
        row = {"team": team}
        for rname in rec.round_names:
            entry = ROUND_WIN_STAGE[rname] - 1  # stage required to play this round
            reached = (rec.stage[:, i] >= entry).sum()
            won = (rec.stage[:, i] >= ROUND_WIN_STAGE[rname]).sum()
            row[f"P_win_{rname}_given_reach_%"] = (
                100 * won / reached if reached else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def load_confederations(data_dir: str) -> dict[str, str]:
    path = os.path.join(data_dir, "confederations.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["confederation"]))


def headline_stats(rec: SimRecord, confeds: dict[str, str], hosts: set[str]) -> pd.DataFrame:
    """Named joint probabilities computed across simulations."""
    n = rec.n
    idx = {t: i for i, t in enumerate(rec.teams)}
    out: list[tuple[str, float]] = []

    champion_of = np.argmax(rec.stage == 6, axis=1)  # exactly one 6 per sim

    if confeds:
        conf_arr = np.array([confeds.get(t, "?") for t in rec.teams])
        for conf in sorted(set(conf_arr)):
            p = (conf_arr[champion_of] == conf).mean()
            out.append((f"Champion from {conf}", p))

    host_idx = [idx[h] for h in sorted(hosts) if h in idx]
    if host_idx:
        h = rec.stage[:, host_idx]
        out.append(("All hosts reach the knockouts", (h >= rec.entry_stage).all(axis=1).mean()))
        out.append(("At least one host reaches the QF", (h >= 3).any(axis=1).mean()))
        out.append(("At least one host reaches the SF", (h >= 4).any(axis=1).mean()))
        out.append(("A host reaches the Final", (h >= 5).any(axis=1).mean()))
        out.append(("A host wins the World Cup", (h >= 6).any(axis=1).mean()))

    top2 = np.argsort(rec.ratings)[::-1][:2]
    finalists = rec.stage >= 5
    out.append(
        (
            f"Final is {rec.teams[top2[0]]} vs {rec.teams[top2[1]]} (top-2 Elo)",
            (finalists[:, top2[0]] & finalists[:, top2[1]]).mean(),
        )
    )
    top10 = set(np.argsort(rec.ratings)[::-1][:10])
    outside = np.array([i not in top10 for i in range(len(rec.teams))])
    out.append(
        ("A team outside the Elo top 10 reaches the SF",
         (rec.stage[:, outside] >= 4).any(axis=1).mean())
    )
    out.append(
        ("The Elo favourite wins the title",
         (champion_of == int(np.argmax(rec.ratings))).mean())
    )
    return pd.DataFrame(
        [{"event": e, "prob_%": 100 * p} for e, p in out]
    )
