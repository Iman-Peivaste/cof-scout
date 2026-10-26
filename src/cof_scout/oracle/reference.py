from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceOracle:
    """Hash-verified lookup oracle for the frozen JSONL landscape."""

    def __init__(
        self,
        landscape: Path,
        expected_sha256_or_feasible: str | Path,
        manifest_path: Path | None = None,
    ):
        actual = sha256(landscape)
        if manifest_path is None and isinstance(expected_sha256_or_feasible, str) and len(expected_sha256_or_feasible) == 64:
            expected_sha256 = expected_sha256_or_feasible
        else:
            manifest = manifest_path or landscape.with_name("reference_manifest.json")
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            expected_sha256 = str(metadata["output_sha256"])
        if actual != expected_sha256:
            raise ValueError(f"Reference landscape hash mismatch: {actual}")
        self.landscape = landscape
        self.landscape_sha256 = actual
        self._rows: dict[str, dict[str, Any]] = {}
        with landscape.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                candidate_id = row.get("candidate_id") or row.get("key")
                if not candidate_id:
                    candidate_id = "/".join(
                        str(row[name]) for name in ("node_id", "linker_id", "linkage", "r_group_id")
                    )
                self._rows[candidate_id] = row

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "oracle_mode": "frozen_reference_lookup",
            "reference_landscape": str(self.landscape),
            "reference_landscape_sha256": self.landscape_sha256,
            "records": len(self._rows),
        }

    def evaluate(self, *candidate: Any) -> dict[str, Any]:
        candidate_id = str(candidate[0]) if len(candidate) == 1 else "/".join(map(str, candidate[:4]))
        try:
            row = self._rows[candidate_id]
            return dict(row.get("result", row))
        except KeyError as exc:
            raise KeyError(f"Candidate not in frozen landscape: {candidate_id}") from exc
