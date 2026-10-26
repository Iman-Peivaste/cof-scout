from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cof_scout.paths import data_path


def load_library(path: Path | None = None) -> dict[str, Any]:
    source = path or data_path("building_blocks_library_v2.json")
    return json.loads(source.read_text(encoding="utf-8"))


def load_feasible_candidates(path: Path | None = None) -> list[dict[str, str]]:
    source = path or data_path("search_space_v2_feasible.csv")
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
