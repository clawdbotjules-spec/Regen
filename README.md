# 2026 FIFA World Cup Simulator

A Monte-Carlo simulator that estimates how far each of the 48 teams is likely
to advance at the 2026 World Cup. It walks ~150 years of international results
to build **World-Football-Elo** ratings, fits a **Dixon-Coles** Poisson goals
model on top of those ratings, and then simulates the full 48-team / 12-group
tournament tens of thousands of times.

For every team it reports the probability of reaching each stage — **Round of
32, Round of 16, Quarter-final, Semi-final, Final, Champion** — plus the single
most likely stage at which they bow out.

> ⚠️ **This is for analysis and entertainment.** International football is
> noisy, a World Cup is a tiny sample, and (as the backtest below shows) this
> model is roughly on par with a plain Elo baseline and **will not reliably
> beat sharp betting markets.** Don't bet your house on it.

---

## Quick start

```bash
python -m pip install -r requirements.txt   # pandas, numpy, scipy
python scripts/download_data.py             # fetch results.csv + shootouts.csv
python main.py                              # 20,000 simulations, seed 42
```

`results.csv` and `shootouts.csv` are also committed, so `python main.py` works
out of the box. `groups.csv` (the 2026 draw) lives in `data/`.

### Useful flags

```bash
python main.py -n 50000 --seed 7      # more simulations, different seed
python main.py --validate             # run the 2018/2022 backtest first
python main.py --show-ratings 30      # print the top-30 Elo table
python main.py --half-life 540        # shorter goals-model memory (~1.5 yrs)
python main.py --friendly-weight 0.25 # trust friendlies even less
python main.py --host-boost 30        # give USA/Canada/Mexico an Elo bump
python main.py --shootout-weight 0    # pure coin-flip penalty shootouts
python main.py --help                 # all options
```

Output: a sorted table to the console (Champion % descending) and the full
table to `output/predictions.csv`.

---

## How it works (step by step)

### Data (`wcsim/data.py`)
Source: the **martj42 "International football results 1872→2026"** dataset
(`results.csv`, `shootouts.csv`, optional `goalscorers.csv`). Cleaning:

- parse dates; **drop fixtures with no recorded score** (future/scheduled rows),
- cast scores to int, coerce `neutral` to a real boolean,
- **normalise team names** via an alias map so the draw matches the history
  (e.g. `Curacao → Curaçao`, `USA → United States`, `Türkiye → Turkey`),
- expose two weighting controls used by the goals model:
  - **exponential time decay** with a configurable half-life in days
    (`weight = 0.5 ** (age_days / half_life)`), so recent matches count more,
  - **competition weighting** that down-weights friendlies vs. competitive games.

### Step 1 — Team strength: Elo (`wcsim/elo.py`)
A standard [World Football Elo](https://en.wikipedia.org/wiki/World_Football_Elo_Ratings)
walk over the full match history in chronological order:

- expected score `We = 1 / (1 + 10**(-dr/400))`, where `dr = R_home − R_away`
  plus a **+100 home advantage** unless the match is at a neutral venue,
- K-factor `K = importance(tournament) × goal_difference_multiplier(margin)`
  (importance: 60 World Cup, 50 continental finals, 40 qualifiers/Nations
  League, 30 other, 20 friendlies; the margin multiplier is the usual
  1.5 for a 2-goal win, 1.75 for 3, etc.),
- update `R' = R + K·(W − We)` for both teams.

The walk also records each match's **pre-game ratings**, which become the
strength feature for the goals model — so the goals model is trained on exactly
the ratings the simulator later uses. Run with `--show-ratings` to sanity-check;
the top of the table (Spain, Argentina, France, England, Brazil…) looks right.

### Step 2 — Match model: Dixon-Coles (`wcsim/match_model.py`)
Goals for each side are modelled as Poisson with a low-score correction
([Dixon & Coles 1997](https://doi.org/10.1111/1467-9876.00065)):

```
log λ_home = c0 + c1·(R_home − R_away)/100 + c2·home_flag
log μ_away = c0 − c1·(R_home − R_away)/100
P(x,y)     = τ(x,y; λ,μ,ρ) · Poisson(x; λ) · Poisson(y; μ)
```

`c0` is the baseline scoring rate, `c1` turns an Elo edge into goal supremacy,
`c2` is the home effect (applied only at non-neutral venues), and the
Dixon-Coles `τ(·)` term with parameter `ρ` corrects the dependence in the
0-0 / 1-0 / 0-1 / 1-1 cells. All four parameters are fit jointly by **weighted
maximum likelihood** (time-decay × competition weight) with `scipy.optimize`.

The model returns both **win/draw/loss probabilities** (for the backtest) and a
**sampled scoreline** (for the simulator), because the group stage needs goal
difference and goals scored for tiebreakers. A typical fit:

```
c0 = 0.107  -> ~1.11 goals/side in a neutral, even game
c1 = 0.184  per 100 Elo points
c2 = 0.228  -> home teams score ~1.26x
ρ  = -0.038 (mild, the usual negative Dixon-Coles value)
```

### Step 3 — Tournament structure (`wcsim/tournament.py`, `wcsim/simulate.py`)
The **2026 format**: 48 teams in 12 groups of 4; each team plays its 3
group-mates once. The **top 2 of every group (24) plus the 8 best 3rd-placed
teams = 32** advance to a Round of 32, then single elimination
R32 → R16 → QF → SF → Final.

- **Group ranking tiebreakers**, in order: points → goal difference →
  goals scored → **head-to-head** (mini-league points among teams tied on
  points, GD and GF) → random.
- **8 best third-placed teams** ranked across all 12 groups by points →
  goal difference → goals scored.
- **Knockouts**: a drawn game goes to a **penalty shootout** modelled as a
  coin flip nudged slightly by Elo (`--shootout-weight`, 0 = pure coin flip).
- **Bracket wiring is configurable** in [`config/bracket.json`](config/bracket.json):
  every R32 → Final match is a slot reference (`W_A`, `R_B`, `T_3`, or the
  winner of an earlier match), so you can swap in different pairings without
  touching code.

> **Caveat on the bracket.** The *official* R32 pairings depend on **which** 8
> groups the third-placed teams come from (a 495-row FIFA lookup table). This
> repo ships a **balanced, representative** bracket and assigns the qualifying
> thirds in **rank order** (`T_1` = best third … `T_8` = 8th). The shipped
> bracket spreads the group winners across the draw and keeps the two halves
> apart until the Final, but it is **not** the official seeding. Drop the real
> pairings into `config/bracket.json` once finalised. Aggregate "reach stage X"
> probabilities are fairly robust to this; exact opponent paths are not.

The whole tournament is **vectorised across the N simulations** (numpy arrays
with a leading axis of length N), so 20,000 tournaments — including the
150-year Elo walk and the model fit — run in **~4 seconds**.

### Step 4 — Groups input (`data/groups.csv`)
The 48 teams and their group assignments are read from `data/groups.csv`
(columns `group, team`); nothing is hard-coded. Any team in `groups.csv` with
no rating from the history triggers a warning and falls back to the initial
rating (1500). With the current draw, all 48 teams match the history.

### Step 5 — Simulate (`wcsim/simulate.py`)
Runs `N` simulations (default 20,000, `-n`), aggregates each team's frequency
of reaching each stage, prints a table sorted by Champion % and writes
`output/predictions.csv`. The `most_likely_finish` column is the modal stage at
which a team is eliminated (e.g. "Round of 32" = usually qualifies from the
group but loses its first knockout game; "Group stage" = usually eliminated in
the group). A fixed `--seed` makes runs reproducible.

### Step 6 — Validation / backtest (`wcsim/validate.py`)
Backtests the goals model on the **2018 and 2022** World Cups: for each, it
trains **only on matches before the tournament**, predicts every actual finals
match, and reports **multiclass log-loss** and **Brier score** against a
**Davidson Elo-only baseline** (a one-parameter Elo → win/draw/loss map, fit by
MLE). Knockout games decided on penalties are scored as draws (their 90'/120'
result), i.e. we grade the pre-shootout 3-way prediction.

**Results (`python main.py --validate`):**

| Tournament | n | Dixon-Coles log-loss | DC Brier | Elo-only log-loss | Elo Brier |
|---|---|---|---|---|---|
| WC 2018 | 64 | 0.991 | 0.588 | 0.982 | 0.582 |
| WC 2022 | 64 | 1.044 | 0.609 | 1.042 | 0.612 |
| **Pooled** | **128** | **1.017** | **0.599** | **1.012** | **0.597** |

**Honest read:** on these 128 matches the Dixon-Coles model and a well-tuned
Elo-only baseline are **effectively tied — Elo is marginally ahead here.** That
is a common, expected outcome: for 3-way match results a calibrated Elo is very
hard to beat, and 64-match samples carry wide error bars. The goals model still
earns its place because it produces **full scorelines**, which the group stage
needs for goal-difference / goals-scored tiebreakers (Elo alone can't). The
calibration table printed by `--validate` shows the mid-range probabilities are
reasonably calibrated. Bottom line: useful for analysis, **not** an edge over
the market.

---

## Project layout

```
main.py                 CLI entry point
requirements.txt
config/bracket.json     configurable R32 -> Final bracket
data/
  groups.csv            the 2026 draw (committed)
  results.csv           historical results (committed; refresh via script)
  shootouts.csv         penalty-shootout history (committed)
scripts/download_data.py
wcsim/
  config.py             all tunable parameters + Elo K tiers + name aliases
  data.py               load / clean / normalise / weight
  elo.py                World-Football-Elo chronological walk
  match_model.py        Dixon-Coles Poisson goals model
  tournament.py         bracket loader + stage definitions
  simulate.py           vectorised Monte-Carlo engine
  validate.py           2018/2022 backtest vs Elo baseline
output/predictions.csv  generated
```

## Assumptions & limitations

- **Men's senior international results only**; no club form, injuries,
  suspensions, squad changes, travel, weather, or in-tournament momentum.
- **Elo never forgets**, but the goals model uses a time-decay half-life
  (default ~3 years) so recent form dominates the scoring estimates.
- **Neutral venues** are assumed for all 2026 matches by default; host nations
  get no edge unless you pass `--host-boost`.
- **Shootouts** are a lightly Elo-weighted coin flip, not a penalty model.
- **Head-to-head** tiebreaks use mini-league points among teams level on
  points (a reasonable simplification of the full FIFA criteria).
- The **bracket seeding is representative, not official** (see Step 3).
- Goals are modelled as (corrected) **Poisson**; real scorelines have fatter
  tails and correlations this only partly captures.

Everything is configurable from the CLI or `wcsim/config.py` — tune the
half-life, friendly weight, home advantage, shootout weight, host boost, seed,
and number of simulations to explore the sensitivity of the results.
