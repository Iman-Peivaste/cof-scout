#!/usr/bin/env python3
"""Download the COF-SCOUT Zenodo data record and verify every SHA-256."""

from __future__ import annotations

import argparse
import csv
import hashlib
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", default="ZENODO_DOI")
    parser.add_argument("--base-url", help="direct Zenodo-record file URL; overrides DOI resolution")
    parser.add_argument("--destination", type=Path, default=Path("data/zenodo"))
    args = parser.parse_args()
    if args.doi == "ZENODO_DOI" and not args.base_url:
        parser.error("replace ZENODO_DOI after deposition or provide --base-url")
    base = args.base_url or f"https://zenodo.org/records/{args.doi.rsplit('.', 1)[-1]}/files"
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = args.destination / "MANIFEST.csv"
    urllib.request.urlretrieve(f"{base}/MANIFEST.csv?download=1", manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        target = args.destination / row["file"]
        if not target.exists() or sha256(target) != row["sha256"]:
            urllib.request.urlretrieve(f"{base}/{row['file']}?download=1", target)
        actual = sha256(target)
        if actual != row["sha256"] or target.stat().st_size != int(row["bytes"]):
            raise SystemExit(f"verification failed: {target}")
        print(f"verified {target}")


if __name__ == "__main__":
    main()
