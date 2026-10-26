#!/usr/bin/env python3
"""Verify archived freeze files, including redacted-chain-of-custody hashes."""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import tempfile
from pathlib import Path

from cof_scout.paths import data_root

HASH_LINE = re.compile(r"^([0-9a-f]{64})\s+[* ]?(.+)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redaction_hashes(log: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    if not log.is_file():
        return result
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "SHA-256" in line or line.startswith("|---"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) >= 5:
            result[cells[1]] = (cells[3], cells[4])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_root())
    args = parser.parse_args()
    archive = args.data_dir / "freeze_manifests.tar.gz"
    log = args.data_dir / "REDACTION_LOG.md"
    if not archive.is_file():
        parser.error(f"missing {archive}; run scripts/download_data.py first")
    chain = redaction_hashes(log)
    checked = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root, filter="data")
        for hash_list in root.rglob("*.sha256"):
            for line in hash_list.read_text(encoding="utf-8").splitlines():
                match = HASH_LINE.match(line.strip())
                if not match:
                    continue
                expected, recorded_path = match.groups()
                path = (hash_list.parent / recorded_path).resolve()
                if not path.is_file() or not path.is_relative_to(root.resolve()):
                    continue
                actual = sha256(path)
                archive_key = path.relative_to(root).as_posix()
                if actual != expected:
                    original_redacted = chain.get(archive_key)
                    if not original_redacted or original_redacted != (expected, actual):
                        raise SystemExit(f"freeze verification failed: {archive_key}")
                checked += 1
    print(f"verified {checked} frozen hash entries")


if __name__ == "__main__":
    main()
