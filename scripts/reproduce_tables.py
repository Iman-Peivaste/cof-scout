#!/usr/bin/env python3
"""Regenerate released CSV tables from the archived source-data bundle."""

from __future__ import annotations

import argparse
import csv
import tarfile
import tempfile
from pathlib import Path

from cof_scout.paths import data_root


def normalize_csv(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_root())
    parser.add_argument("--output", type=Path, default=Path("reproduced/tables"))
    args = parser.parse_args()
    archive = args.data_dir / "figure_source_data.tar.gz"
    if not archive.is_file():
        parser.error(f"missing {archive}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root, filter="data")
        sources = sorted((root / "figure_source_data" / "publication" / "tables").glob("*.csv"))
        for source in sources:
            normalize_csv(source, args.output / source.name)
    print(f"regenerated {len(sources)} tables in {args.output}")


if __name__ == "__main__":
    main()
