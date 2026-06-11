"""2026 FIFA World Cup Monte-Carlo simulator.

A modular, dependency-light (pandas / numpy / scipy) toolkit that:
  * builds World-Football-Elo ratings from raw international results,
  * fits a Dixon-Coles Poisson goals model on top of those ratings,
  * simulates the full 48-team / 12-group 2026 tournament many thousands
    of times, and
  * backtests the goals model on the 2018 and 2022 World Cups.

See README.md for assumptions and limitations.
"""

__all__ = [
    "config",
    "data",
    "elo",
    "match_model",
    "tournament",
    "simulate",
    "validate",
]
