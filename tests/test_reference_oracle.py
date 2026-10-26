import hashlib
import json

import pytest

from cof_scout.oracle.reference import ReferenceOracle


def test_lookup_requires_matching_landscape_hash(tmp_path):
    landscape = tmp_path / "landscape.jsonl"
    row = {"candidate_id": "N001/L001/imine/R001", "hit": False}
    landscape.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(landscape.read_bytes()).hexdigest()
    oracle = ReferenceOracle(landscape, digest)
    assert oracle.evaluate(row["candidate_id"]) == row
    with pytest.raises(ValueError, match="hash mismatch"):
        ReferenceOracle(landscape, "0" * 64)
