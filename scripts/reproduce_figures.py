#!/usr/bin/env python3
"""Regenerate paper figures from archived outputs without LLM calls."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

from cof_scout.figures import render_figure
from cof_scout.paths import data_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_root())
    parser.add_argument("--output", type=Path, default=Path("reproduced/figures"))
    parser.add_argument("--figure", type=int, choices=range(1, 7))
    args = parser.parse_args()
    archive = args.data_dir / "figure_source_data.tar.gz"
    if not archive.is_file():
        parser.error(f"missing {archive}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root, filter="data")
        source = root / "figure_source_data" / "publication" / "tables"
        numbers = [args.figure] if args.figure else list(range(1, 7))
        for number in numbers:
            render_figure(number, source, args.output)
    print(f"regenerated figures {numbers} in {args.output}")


if __name__ == "__main__":
    main()
