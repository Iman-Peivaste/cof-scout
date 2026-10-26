from __future__ import annotations

from collections.abc import Iterable


def distinct_hit_scaffolds(rows: Iterable[dict[str, object]]) -> int:
    return len({str(row["scaffold_id"]) for row in rows if bool(row.get("hit"))})


def cumulative_hit_scaffolds(rows: Iterable[dict[str, object]]) -> list[int]:
    seen: set[str] = set()
    curve: list[int] = []
    for row in rows:
        if bool(row.get("hit")):
            seen.add(str(row["scaffold_id"]))
        curve.append(len(seen))
    return curve
