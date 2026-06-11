# 2026 FIFA World Cup Simulator

A Monte-Carlo simulator that estimates how far each of the 48 teams is likely
to advance at the 2026 World Cup. It walks ~150 years of international results
to build **World-Football-Elo** ratings, fits a **suite of goal-scoring models**
(Dixon-Coles, bivariate Poisson, negative binomial, and an ensemble of the
three) on top of those ratings, and simulates the full 48-team tournament
through the **official post-draw knockout bracket** (FIFA matches 73–104,
including the constrained third-place allocation) tens of thousands of times.

For every team it reports the probability of reaching each stage — **Round of
32, Round of 16, Quarter-final, Semi-final, Final, Champion** — plus
round-by-round matchup probabilities, likely opponents, most likely Finals,
group-position distributions, headline joint probabilities, sensitivity bands
and a two-tournament backtest.

> ⚠️ **This is for analysis and entertainment.** International football is
> noisy and a World Cup is a tiny sample. As the backtests below show, the
> goal models are statistically tied with a plain Elo baseline at match level,
> and nothing here will reliably beat sharp betting markets.

---

## Quick start

```bash
python -m pip install -r requirements.txt   # pandas, numpy, scipy
python main.py                              # 20,000 sims, ensemble model, seed 42
```

The data (`results.csv`, `shootouts.csv`, `groups.csv`) is committed, so this
works out of the box; refresh the history any time with
`python scripts/download_data.py`.

### Useful flags

```bash
python main.py -n 50000 --seed 7        # more sims, different seed
python main.py --model dc               # Dixon-Coles only (or bp / nb / ensemble)
python main.py --validate               # match- AND tournament-level backtests
python main.py --sensitivity            # knob + parameter uncertainty report
python main.py --half-life 540          # shorter goals-model memory (~1.5 yrs)
python main.py --friendly-weight 0.25   # trust friendlies even less
python main.py --no-host-home           # treat host group matches as neutral
python main.py --shootout-model weighted --shootout-weight 0   # coin-flip pens
python main.py --help                   # everything else
```

### Outputs (written to `output/`)

| file | contents |
|---|---|
| `predictions.csv` | per team: P(reach each stage), MC standard error, modal finish |
| `group_stats.csv` | P(1st/2nd/3rd/4th), P(advance), P(advance as 3rd), expected pts/GF/GA |
| `stage_distribution.csv` | exact elimination-stage distribution (rows sum to 100%) |
| `r32_matchups.csv` | most likely pairings for each official R32 match |
| `opponents_by_round.csv` | per team & round: P(play) and likeliest opponents |
| `meet_matrix.csv` | 48x48 P(the two teams meet in a knockout match) |
| `finals.csv` | most likely Finals pairings |
| `headline_stats.csv` | host/confederation/seed joint probabilities |
| `conditional_advancement.csv` | P(win the round \| reached it), per team |
| `sensitivity.csv` | champion % under every modeling variation (with `--sensitivity`) |

---

## How it works (step by step)

### Data (`wcsim/data.py`)
Source: the **martj42 "International football results 1872→2026"** dataset
(49,400 cleaned matches). Cleaning: parse dates, drop unplayed fixtures,
coerce types, **normalise team names** via an alias map (`Curacao → Curaçao`,
`Türkiye → Turkey`, …). Two weighting controls feed the goals fit:
exponential **time decay** (configurable half-life, default ~3 years) and a
**friendly down-weight** (default 0.5).

### Step 1 — Team strength: Elo (`wcsim/elo.py`)
A standard World-Football-Elo walk over the full history in chronological
order: expected score `1/(1+10^(-dr/400))`, +100 home advantage at non-neutral
venues, K scaled by competition importance (60 World Cup … 20 friendly) × the
usual goal-difference multiplier. The walk records each match's **pre-game
ratings**, which become the strength feature for every goals model — so the
models are trained on exactly the ratings the simulator later uses. Top of the
2026 table: Spain, Argentina, France, England, Brazil — sane.

### Step 2 — Match models (`wcsim/match_model.py`)
All models share the same log-linear link from Elo to expected goals
(`log λ = c0 + c1·ΔElo/100 + c2·home`), are fit by **weighted MLE**, and
return both W/D/L probabilities and sampled scorelines:

- **`dc` Dixon-Coles** — independent Poissons with the τ low-score correction
  (fitted ρ ≈ −0.04, the textbook mild negative value).
- **`bp` bivariate Poisson** (Karlis-Ntzoufras) — a shared Poisson component
  creates positive score correlation. Honest finding: the fitted shared
  component is ≈ 0 on this data — once Elo sets the means, scores show no
  extra positive correlation, so BP nearly degenerates to a double Poisson.
- **`nb` negative binomial** — over-dispersed goals (fitted dispersion r ≈ 9,
  i.e. variance ≈ 1.14× mean at typical scoring rates: mild fat tails).
- **`ensemble`** (default) — equal-weight mixture of the three. Sampling draws
  each scoreline from a randomly chosen member, which is exactly sampling
  from the averaged distribution.

### Step 2b — Penalty shootouts (`wcsim/shootout.py`)
Instead of a hand-tuned "coin flip weighted slightly by strength", the default
model is **fit on the actual shootout history** (677 shootouts joined to their
pre-match Elo): a one-parameter logistic on the Elo difference. The data says
shootouts are nearly coin flips — a +100-Elo side wins just **53.5%** — and
that's what the simulator uses. `--shootout-model weighted` restores the
manual interpolation if you want to play with it.

### Step 3 — Tournament structure (`wcsim/tournament.py`, `wcsim/simulate.py`)
2026 format: 12 groups of 4 → top 2 plus the **8 best third-placed teams**
(ranked across groups by points → GD → GF) → Round of 32 → … → Final. Group
tiebreakers: points → GD → GF → head-to-head mini-league among tied teams →
random.

**The bracket is the official one** (`config/bracket.json`, FIFA match numbers
73–104, verified against the published FIFA schedule): e.g. winner of Group J
meets the Group H runner-up (an Argentina–Uruguay R32 collision in 44% of
sims), and each of the 8 third-place R32 slots carries FIFA's exact 5-group
candidate set (e.g. M74: Winner E vs 3rd of A/B/C/D/F). Third-placed teams are
assigned to slots by a **constraint-respecting deterministic matching**
(validated against all C(12,8) = 495 qualification combinations): like FIFA's
Annex C table, the assignment depends only on *which* groups qualify. The one
approximation: where FIFA's table picks a specific row, we use an
alphabetical-first feasible assignment — same constraints, possibly different
slot order within them. The third-place playoff (M103) is omitted (it doesn't
affect how far anyone advances). Hosts (Mexico, Canada, USA) play their
**group matches at home** by default (`--no-host-home` to disable); knockout
venues are treated as neutral.

The engine is **fully vectorised across simulations** — 20,000 tournaments
including the Elo walk, three model fits and all analysis take ~6 seconds.

### Step 4 — Groups input (`data/groups.csv`)
The 48 teams / 12 groups are read from `data/groups.csv` — which matches the
real December 5, 2025 draw team-for-team. Teams missing from the history get
a warning and the initial 1500 rating (currently: none).

### Step 5 — Simulate & analyse (`wcsim/simulate.py`, `wcsim/analysis.py`)
Default 20,000 sims (`-n`), fixed seed (`--seed`), bitwise-reproducible.
Beyond the headline table, the analysis layer aggregates **every recorded
possibility**: group position distributions, exact elimination stages, every
knockout slot's likely pairings, per-team likely opponents per round, the
48×48 "probability we ever meet" matrix, most likely Finals, conditional
win-rates per round, and joint headline events (host runs, confederation
champions, top-seed outcomes).

### Step 6 — Validation (`wcsim/validate.py`), run with `--validate`

**Match-level** (train strictly pre-tournament, predict all 64 matches,
multiclass log-loss / Brier / RPS, vs a Davidson Elo-only baseline fit by MLE):

| pooled 2018+2022 (128) | log-loss | Brier | RPS |
|---|---|---|---|
| Elo-only (Davidson) | **1.0119** | 0.5972 | 0.2137 |
| Negative binomial | 1.0122 | 0.5976 | 0.2138 |
| Ensemble | 1.0144 | 0.5978 | 0.2140 |
| Bivariate Poisson | 1.0147 | 0.5976 | 0.2140 |
| Dixon-Coles | 1.0171 | 0.5986 | 0.2142 |

**Honest read:** all five are within 0.005 log-loss — statistically
indistinguishable on 128 matches, with the Elo baseline nominally first. The
goal models earn their place not by beating Elo on W/D/L but by producing
**scorelines** (needed for group tiebreakers) and slightly better draw rates.

**Tournament-level** (re-simulate 2018/2022 with their real groups and
brackets, 10,000 sims each):

- **WC 2018:** actual champion France was predicted 5.6% (rank 6 of 32);
  champion log-loss 2.89 vs 3.47 uniform.
- **WC 2022:** actual champion Argentina was predicted **21.4% (rank 2)**;
  log-loss 1.54 vs 3.47 uniform.
- Pooled stage-level Brier skill vs a "everyone equal" baseline: **+22% at
  R16, +21% at QF, +3% SF, +7% Final, +12% Champion** — real but modest skill,
  strongest where the field is wide.

### Step 7 — Sensitivity & uncertainty (`wcsim/sensitivity.py`), `--sensitivity`
Re-runs the simulation under 12 modeling variations (half-life, friendly
weight, model family, shootout treatment, host assumptions) and reports each
top team's champion-probability spread vs Monte-Carlo noise — e.g. Spain
ranges ~19–27% across knobs, with the shootout treatment the biggest lever.
It also propagates **goals-model parameter uncertainty** (sampling from the
asymptotic normal around the Dixon-Coles MLE) into champion-probability bands
(Spain ≈ 14–34% at the 5th–95th percentile). Elo-rating uncertainty itself is
not modeled — the true bands are wider still.

---

## Headline results (20,000 sims, ensemble, seed 42)

Spain ~21% champion, Argentina ~17%, France ~9%, England ~6%, Colombia and
Brazil ~5–6%. UEFA takes the title in ~54% of sims, CONMEBOL ~35%. A host
nation wins it all in ~3.5% of sims. Most likely Final: Spain vs Argentina
(~7%). Full numbers regenerate with `python main.py`.

## Project layout

```
main.py                    CLI entry point
requirements.txt
config/bracket.json        OFFICIAL 2026 bracket (editable)
config/bracket_32.json     classic 32-team chart (backtests)
data/
  groups.csv               the real Dec 2025 draw
  results.csv              historical results (refresh via script)
  shootouts.csv            shootout history (drives the empirical pens model)
  confederations.csv       team -> confederation (headline stats)
  wc2018_groups.csv        2018 groups + actual finishes (backtest)
  wc2022_groups.csv        2022 groups + actual finishes (backtest)
scripts/download_data.py
wcsim/
  config.py                all tunable parameters, Elo K tiers, name aliases
  data.py                  load / clean / normalise / weight
  elo.py                   World-Football-Elo chronological walk
  match_model.py           DC + bivariate Poisson + negbin + ensemble
  shootout.py              empirical penalty-shootout model
  tournament.py            bracket loader + third-place allocation (Kuhn matching)
  simulate.py              vectorised Monte-Carlo engine (48- and 32-team)
  analysis.py              round-by-round aggregation layer
  validate.py              match- and tournament-level backtests
  sensitivity.py           knob sensitivity + parameter uncertainty
output/                    generated CSVs
```

## Assumptions & limitations

- Men's senior international results only; no club form, injuries, squads,
  travel, weather, or in-tournament momentum.
- Elo never forgets; the goals models use a ~3-year half-life so recent form
  dominates scoring estimates.
- Host home advantage applies to hosts' group matches only; knockout venues
  are treated as neutral even when a host would effectively be at home.
- The third-place slot assignment respects FIFA's constraint sets but may
  order teams across slots differently than FIFA's Annex C table; aggregate
  stage probabilities are insensitive to this, exact opponent paths less so.
- Shootouts are a one-parameter logistic on Elo — no penalty-taker data.
- Head-to-head tiebreaks use a mini-league among tied teams (a faithful
  simplification of the full FIFA criteria; fair-play points and disciplinary
  tiebreakers are replaced by a random draw).
- All models share the single Elo strength feature; they cannot express
  attack-vs-defense style differences between equally-rated teams.
- Parameter uncertainty bands exclude Elo-rating uncertainty, so true
  uncertainty is wider than reported.
