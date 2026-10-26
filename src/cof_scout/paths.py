"""Portable data-path resolution for code and downloaded artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    configured = os.getenv("COF_SCOUT_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else repository_root() / "data" / "small"


def data_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)
