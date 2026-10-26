#!/usr/bin/env python3
"""CPU-only entry point for rerunning frozen algorithmic controls."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cof_scout.paths import data_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("output/public_baselines"))
    args = parser.parse_args()
    landscape = data_root() / "reference_landscape" / "ground_truth_mcap_v1.jsonl"
    command = [
        sys.executable, "-m", "cof_scout.baselines.reference_sensitivity",
        "--ground-truth", str(landscape),
        "--feasible-csv", str(data_root() / "search_space_v2_feasible.csv"),
        "--random-runs", str(args.runs), "--diversity-runs", str(args.runs),
        "--bo-runs", str(args.runs), "--new-baseline-runs", str(args.runs),
        "--budget", str(args.budget), "--out-dir", str(args.output),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
