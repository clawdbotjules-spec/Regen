"""Fetch the historical international-results dataset into ./data/.

Uses the martj42 GitHub mirror (the same data as the Kaggle dataset
"International football results from 1872 to 2017", kept current), which
needs no Kaggle credentials.

    python scripts/download_data.py            # results + shootouts
    python scripts/download_data.py --all      # also goalscorers.csv

groups.csv (the 2026 draw) is committed in the repo and is NOT fetched here.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results.csv": f"{BASE}/results.csv",
    "shootouts.csv": f"{BASE}/shootouts.csv",
    "goalscorers.csv": f"{BASE}/goalscorers.csv",  # optional, large, unused by the model
}


def download(name: str, url: str, data_dir: str) -> None:
    dest = os.path.join(data_dir, name)
    print(f"  {name:16s} <- {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"     saved {os.path.getsize(dest):,} bytes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--all", action="store_true", help="also download goalscorers.csv")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    wanted = ["results.csv", "shootouts.csv"]
    if args.all:
        wanted.append("goalscorers.csv")

    print(f"Downloading into {args.data_dir}/ ...")
    for name in wanted:
        download(name, FILES[name], args.data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
