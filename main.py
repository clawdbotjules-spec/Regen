"""2026 FIFA World Cup simulator - command-line entry point.

Examples
--------
    python main.py                       # 20,000 sims with defaults
    python main.py -n 50000 --seed 7     # more sims, different seed
    python main.py --validate            # backtest 2018 & 2022 first
    python main.py --show-ratings 25     # print the top-25 Elo table
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from wcsim.config import SimConfig
from wcsim import data as datamod
from wcsim.elo import compute_elo
from wcsim.match_model import DixonColesModel
from wcsim.tournament import load_bracket
from wcsim.simulate import WorldCupSimulator
from wcsim.validate import backtest


def build_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        data_dir=args.data_dir,
        output_path=args.output,
        bracket_path=args.bracket,
        seed=args.seed,
        n_sims=args.sims,
        elo_home_advantage=args.home_advantage,
        elo_cutoff=args.elo_cutoff,
        half_life_days=args.half_life,
        friendly_weight=args.friendly_weight,
        max_goals=args.max_goals,
        host_boost=args.host_boost,
        shootout_elo_weight=args.shootout_weight,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2026 FIFA World Cup Monte-Carlo simulator")
    p.add_argument("-n", "--sims", type=int, default=20_000, help="number of simulations")
    p.add_argument("--seed", type=int, default=42, help="random seed (reproducibility)")
    p.add_argument("--data-dir", default="data", help="directory holding the CSVs")
    p.add_argument("--output", default="output/predictions.csv", help="predictions CSV path")
    p.add_argument("--bracket", default="config/bracket.json", help="bracket config JSON")
    p.add_argument("--half-life", type=float, default=1095.0,
                   help="time-decay half-life in days for the goals fit")
    p.add_argument("--friendly-weight", type=float, default=0.5,
                   help="weight of friendlies in the goals fit (1.0 = same as competitive)")
    p.add_argument("--home-advantage", type=float, default=100.0, help="Elo home advantage")
    p.add_argument("--host-boost", type=float, default=0.0,
                   help="extra Elo for host nations (USA/Canada/Mexico) in the sim")
    p.add_argument("--shootout-weight", type=float, default=0.5,
                   help="0 = coin-flip shootouts, 1 = full Elo-implied")
    p.add_argument("--max-goals", type=int, default=12, help="score-grid truncation per side")
    p.add_argument("--elo-cutoff", default=None,
                   help="ISO date; use only matches before it (default: all data)")
    p.add_argument("--validate", action="store_true",
                   help="run the 2018/2022 backtest before simulating")
    p.add_argument("--no-sim", action="store_true", help="skip the tournament simulation")
    p.add_argument("--show-ratings", type=int, default=20,
                   help="how many top Elo ratings to print (0 to hide)")
    p.add_argument("--top", type=int, default=48,
                   help="how many teams to print in the predictions table")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 100)

    cutoff = pd.to_datetime(cfg.elo_cutoff) if cfg.elo_cutoff else None

    print("Loading data ...")
    results = datamod.load_results(cfg.data_dir)
    groups = datamod.load_groups(cfg.data_dir)
    print(f"  {len(results):,} cleaned matches, {results['date'].min().date()} "
          f"-> {results['date'].max().date()}")

    print("\nComputing Elo ratings (chronological walk) ...")
    ratings, prematch = compute_elo(results, cfg, cutoff=cutoff)

    if args.show_ratings > 0:
        print(f"\nTop {args.show_ratings} Elo ratings (sanity check):")
        top = ratings.head(args.show_ratings)
        for rank, (team, val) in enumerate(top.items(), 1):
            in_wc = " *" if team in set(groups["team"]) else ""
            print(f"  {rank:>2}. {team:<22} {val:7.1f}{in_wc}")
        print("  (* = qualified for the 2026 tournament)")

    print("\nFitting Dixon-Coles goals model ...")
    fit_cutoff = cutoff if cutoff is not None else results["date"].max() + pd.Timedelta(days=1)
    model = DixonColesModel(cfg).fit(prematch, cutoff=fit_cutoff)
    print(model.summary())

    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION / BACKTEST")
        print("=" * 60)
        backtest(results, cfg)

    if args.no_sim:
        return

    print("\n" + "=" * 60)
    print(f"SIMULATING {cfg.n_sims:,} TOURNAMENTS (seed={cfg.seed})")
    print("=" * 60)
    bracket = load_bracket(cfg.bracket_path)
    rng = np.random.default_rng(cfg.seed)
    sim = WorldCupSimulator(groups, ratings, model, bracket, cfg, rng)
    res = sim.run(cfg.n_sims)
    table = res.table()

    # console output
    show = table.head(args.top).copy()
    pct_cols = ["R32_%", "R16_%", "QF_%", "SF_%", "Final_%", "Champion_%"]
    for c in pct_cols:
        show[c] = show[c].map(lambda v: f"{v:5.1f}")
    print("\nPredicted advancement probabilities (sorted by Champion %):\n")
    print(show.to_string(index=False))

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    out = table.copy()
    for c in pct_cols:
        out[c] = out[c].round(2)
    out.to_csv(cfg.output_path, index=False)
    print(f"\nWrote full predictions to {cfg.output_path}")


if __name__ == "__main__":
    main()
