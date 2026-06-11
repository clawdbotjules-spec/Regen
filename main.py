"""2026 FIFA World Cup simulator - command-line entry point.

Examples
--------
    python main.py                          # 100,000 sims, ensemble model
    python main.py -n 20000 --seed 7        # quicker run, different seed
    python main.py --model dc               # Dixon-Coles only
    python main.py --validate               # match + tournament backtests first
    python main.py --sensitivity            # knob/parameter uncertainty report
    python main.py --tune                   # hyperparameter grid (1998-2014 WCs)
    python main.py --show-ratings 25        # print the top-25 Elo table
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from wcsim.config import SimConfig
from wcsim import data as datamod
from wcsim import analysis
from wcsim.elo import compute_elo
from wcsim.match_model import MODEL_NAMES, make_model
from wcsim.shootout import ShootoutModel
from wcsim.tournament import load_bracket
from wcsim.simulate import WorldCupSimulator
from wcsim.validate import match_backtest, tournament_backtest
from wcsim.sensitivity import parameter_uncertainty, run_sensitivity


def build_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        data_dir=args.data_dir,
        output_path=args.output,
        output_dir=os.path.dirname(args.output) or "output",
        bracket_path=args.bracket,
        seed=args.seed,
        n_sims=args.sims,
        elo_home_advantage=args.home_advantage,
        elo_cutoff=args.elo_cutoff,
        model_name=args.model,
        half_life_days=args.half_life,
        friendly_weight=args.friendly_weight,
        slope_scale=args.slope_scale,
        max_goals=args.max_goals,
        host_boost=args.host_boost,
        host_group_home=not args.no_host_home,
        shootout_model=args.shootout_model,
        shootout_elo_weight=args.shootout_weight,
        backtest_sims=args.backtest_sims,
        sens_sims=args.sens_sims,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2026 FIFA World Cup Monte-Carlo simulator")
    p.add_argument("-n", "--sims", type=int, default=SimConfig.n_sims,
                   help="number of simulations (default 100k, ~15s; "
                        "use 20k for quick iteration)")
    p.add_argument("--seed", type=int, default=42, help="random seed (reproducibility)")
    p.add_argument("--data-dir", default="data", help="directory holding the CSVs")
    p.add_argument("--output", default="output/predictions.csv", help="predictions CSV path")
    p.add_argument("--bracket", default="config/bracket.json", help="bracket config JSON")
    p.add_argument("--model", default="ensemble", choices=MODEL_NAMES,
                   help="goals model: dc (Dixon-Coles), bp (bivariate Poisson), "
                        "nb (negative binomial), ensemble (mixture of all three)")
    p.add_argument("--half-life", type=float, default=SimConfig.half_life_days,
                   help="time-decay half-life in days for the goals fit")
    p.add_argument("--friendly-weight", type=float, default=SimConfig.friendly_weight,
                   help="weight of friendlies in the goals fit (1.0 = same as competitive)")
    p.add_argument("--slope-scale", type=float, default=SimConfig.slope_scale,
                   help="calibration multiplier on the Elo->goals slope "
                        "(default tuned on the 1998-2014 World Cups)")
    p.add_argument("--tune", action="store_true",
                   help="re-run the hyperparameter grid search on the 1998-2014 "
                        "validation World Cups (slow; prints the winning combo)")
    p.add_argument("--home-advantage", type=float, default=100.0, help="Elo home advantage")
    p.add_argument("--host-boost", type=float, default=0.0,
                   help="extra Elo for host nations (USA/Canada/Mexico) everywhere")
    p.add_argument("--no-host-home", action="store_true",
                   help="treat hosts' group matches as neutral (default: they are home)")
    p.add_argument("--shootout-model", default="empirical", choices=("empirical", "weighted"),
                   help="empirical = logistic fit on shootout history; weighted = manual")
    p.add_argument("--shootout-weight", type=float, default=0.5,
                   help="for --shootout-model weighted: 0 = coin flip, 1 = full Elo")
    p.add_argument("--max-goals", type=int, default=12, help="score-grid truncation per side")
    p.add_argument("--elo-cutoff", default=None,
                   help="ISO date; use only matches before it (default: all data)")
    p.add_argument("--validate", action="store_true",
                   help="run the 2018/2022 match- and tournament-level backtests")
    p.add_argument("--backtest-sims", type=int, default=10_000,
                   help="simulations per tournament-level backtest")
    p.add_argument("--sensitivity", action="store_true",
                   help="run the sensitivity / parameter-uncertainty report")
    p.add_argument("--sens-sims", type=int, default=4_000,
                   help="simulations per sensitivity variation")
    p.add_argument("--no-sim", action="store_true", help="skip the tournament simulation")
    p.add_argument("--no-analysis", action="store_true",
                   help="skip the round-by-round analysis outputs")
    p.add_argument("--show-ratings", type=int, default=20,
                   help="how many top Elo ratings to print (0 to hide)")
    p.add_argument("--top", type=int, default=48,
                   help="how many teams to print in the predictions table")
    return p.parse_args(argv)


def _header(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def _print_group_forecast(gtable: pd.DataFrame) -> None:
    for g, sub in gtable.groupby("group"):
        print(f"\nGroup {g}")
        print(f"  {'team':<24}{'elo':>7}{'1st%':>7}{'2nd%':>7}{'3rd%':>7}"
              f"{'4th%':>7}{'adv%':>7}{'xPts':>6}")
        for _, r in sub.iterrows():
            print(
                f"  {r['team']:<24}{r['elo']:>7.0f}{r['P_1st_%']:>7.1f}"
                f"{r['P_2nd_%']:>7.1f}{r['P_3rd_%']:>7.1f}{r['P_4th_%']:>7.1f}"
                f"{r['P_advance_%']:>7.1f}{r['exp_pts']:>6.2f}"
            )


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_rows", 200)

    cutoff = pd.to_datetime(cfg.elo_cutoff) if cfg.elo_cutoff else None

    print("Loading data ...")
    results = datamod.load_results(cfg.data_dir)
    groups = datamod.load_groups(cfg.data_dir)
    shootouts = datamod.load_shootouts(cfg.data_dir)
    print(f"  {len(results):,} cleaned matches, {results['date'].min().date()} "
          f"-> {results['date'].max().date()}")

    print("\nComputing Elo ratings (chronological walk) ...")
    ratings, prematch = compute_elo(results, cfg, cutoff=cutoff)

    if args.show_ratings > 0:
        print(f"\nTop {args.show_ratings} Elo ratings (sanity check):")
        wc_teams = set(groups["team"])
        for rank, (team, val) in enumerate(ratings.head(args.show_ratings).items(), 1):
            print(f"  {rank:>2}. {team:<22} {val:7.1f}{' *' if team in wc_teams else ''}")
        print("  (* = qualified for the 2026 tournament)")

    fit_cutoff = cutoff if cutoff is not None else results["date"].max() + pd.Timedelta(days=1)

    if args.tune:
        _header("HYPERPARAMETER TUNING (validation: 1998-2014 World Cups)")
        from wcsim.tune import tune as run_tune
        best, tune_table = run_tune(results, cfg)
        os.makedirs(cfg.output_dir, exist_ok=True)
        tune_table.to_csv(os.path.join(cfg.output_dir, "tuning_grid.csv"), index=False)
        import dataclasses
        cfg = dataclasses.replace(cfg, **best)
        print(f"\nUsing tuned parameters for this run: {best}")

    print(f"\nFitting goals model ({cfg.model_name}) ...")
    model = make_model(cfg.model_name, cfg).fit(prematch, cutoff=fit_cutoff)
    print(model.summary())

    shootout = ShootoutModel(cfg).fit(shootouts, prematch, cutoff=cutoff)
    print(shootout.summary())

    if args.validate:
        _header("VALIDATION 1/2: MATCH-LEVEL BACKTEST (all models)")
        match_backtest(results, cfg)
        _header("VALIDATION 2/2: TOURNAMENT-LEVEL BACKTEST")
        tournament_backtest(results, shootouts, cfg)

    if args.sensitivity:
        _header("SENSITIVITY ANALYSIS")
        sens = run_sensitivity(groups, ratings, prematch, shootouts, cfg, fit_cutoff)
        os.makedirs(cfg.output_dir, exist_ok=True)
        sens.to_csv(os.path.join(cfg.output_dir, "sensitivity.csv"))
        parameter_uncertainty(groups, ratings, prematch, shootouts, cfg, fit_cutoff)

    if args.no_sim:
        return

    _header(f"SIMULATING {cfg.n_sims:,} TOURNAMENTS "
            f"(model={cfg.model_name}, seed={cfg.seed})")
    bracket = load_bracket(cfg.bracket_path, set(groups["group"]))
    rng = np.random.default_rng(cfg.seed)
    sim = WorldCupSimulator(groups, ratings, model, bracket, cfg, rng, shootout=shootout)
    res = sim.run(cfg.n_sims)
    rec = res.rec
    table = res.table()

    os.makedirs(cfg.output_dir, exist_ok=True)
    pct_cols = [c for c in table.columns if c.endswith("_%")]

    if not args.no_analysis:
        _header("GROUP-STAGE FORECAST")
        gtable = analysis.group_stage_table(rec)
        _print_group_forecast(gtable)
        gtable.round(2).to_csv(os.path.join(cfg.output_dir, "group_stats.csv"), index=False)

        _header("ROUND OF 32: MOST LIKELY MATCHUPS (official bracket)")
        r32 = analysis.matchup_table(rec, "R32", top_k=5)
        for mid, sub in r32.groupby("match", sort=False):
            tops = "  |  ".join(
                f"{r['pairing']} {r['prob_%']:.0f}%" for _, r in sub.head(3).iterrows()
            )
            print(f"  {mid:>4}: {tops}")
        r32.round(2).to_csv(os.path.join(cfg.output_dir, "r32_matchups.csv"), index=False)

        opp = analysis.opponents_by_round(rec)
        opp.round(2).to_csv(os.path.join(cfg.output_dir, "opponents_by_round.csv"), index=False)
        analysis.meet_matrix(rec).to_csv(os.path.join(cfg.output_dir, "meet_matrix.csv"))
        analysis.stage_distribution(rec).round(2).to_csv(
            os.path.join(cfg.output_dir, "stage_distribution.csv"), index=False
        )
        analysis.conditional_advancement(rec).round(1).to_csv(
            os.path.join(cfg.output_dir, "conditional_advancement.csv"), index=False
        )

        _header("MOST LIKELY FINALS")
        pairs, outcomes = analysis.finals_table(rec, top_k=10)
        pairs.round(2).to_csv(os.path.join(cfg.output_dir, "finals.csv"), index=False)
        for _, r in pairs.head(8).iterrows():
            print(f"  {r['final']:<42} {r['prob_%']:5.2f}%")
        print("\n  Most likely title outcomes:")
        for _, r in outcomes.head(5).iterrows():
            print(f"  {r['outcome']:<42} {r['prob_%']:5.2f}%")

        _header("HEADLINE PROBABILITIES")
        confeds = analysis.load_confederations(cfg.data_dir)
        heads = analysis.headline_stats(rec, confeds, sim.hosts)
        for _, r in heads.iterrows():
            print(f"  {r['event']:<52} {r['prob_%']:6.2f}%")
        heads.round(2).to_csv(os.path.join(cfg.output_dir, "headline_stats.csv"), index=False)

    _header("ADVANCEMENT PROBABILITIES (sorted by Champion %)")
    show = table.head(args.top).copy()
    for c in pct_cols:
        show[c] = show[c].map(lambda v: f"{v:5.1f}")
    show["Champion_se"] = table.head(args.top)["Champion_se"].map(lambda v: f"{v:.2f}")
    print(show.to_string(index=False))

    out = table.copy()
    for c in pct_cols + ["Champion_se"]:
        out[c] = out[c].round(2)
    out.to_csv(cfg.output_path, index=False)
    print(f"\nWrote predictions to {cfg.output_path}")
    if not args.no_analysis:
        print(
            f"Wrote analysis CSVs to {cfg.output_dir}/: group_stats, r32_matchups,\n"
            "  opponents_by_round, meet_matrix, stage_distribution,\n"
            "  conditional_advancement, finals, headline_stats"
        )


if __name__ == "__main__":
    main()
