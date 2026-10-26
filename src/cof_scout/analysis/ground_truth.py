#!/usr/bin/env python3
"""Summarize a complete COF ground-truth JSONL and write reusable JSON metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# OC-004: L030 is disabled and excluded from the active feasible inventory.
EXPECTED_CANDIDATES = 7209
ROBUST_VBM_MARGIN = 0.25
GRAZER_VBM_MARGIN = 0.10


def feasible_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {(row["node_id"], row["linker_id"], row["linkage_norm"], row["r_group_id"])
                for row in csv.DictReader(handle)
                if str(row.get("is_feasible", "")).strip().lower() == "true"}


def read_ground_truth(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def key(record: dict) -> tuple[str, str, str, str]:
    return (record["node_id"], record["linker_id"], record["linkage"], record["r_group_id"])


def criterion(record: dict) -> dict | None:
    value = record.get("criterion")
    if isinstance(value, dict):
        return value
    nested = record.get("result", {})
    value = nested.get("criterion") if isinstance(nested, dict) else None
    return value if isinstance(value, dict) else None


def validate_complete(records: list[dict], expected: set[tuple[str, str, str, str]]) -> None:
    keys = [key(record) for record in records]
    duplicate_counts = Counter(keys)
    duplicates = [candidate for candidate, count in duplicate_counts.items() if count > 1]
    seen = set(keys)
    missing, extra = expected - seen, seen - expected
    problems = []
    if len(records) != len(expected):
        problems.append(f"record count {len(records)} != expected {len(expected)}")
    if duplicates:
        problems.append(f"duplicate keys: {len(duplicates)}")
    if missing:
        problems.append(f"missing feasible keys: {len(missing)}")
    if extra:
        problems.append(f"keys outside feasible CSV: {len(extra)}")
    if problems:
        raise RuntimeError("Ground truth is incomplete or inconsistent: " + "; ".join(problems))


def summarize(records: list[dict]) -> dict:
    successful = [record for record in records if record.get("result", {}).get("status") == "success"]
    failed = [record for record in records if record not in successful]
    hits = [record for record in successful if (criterion(record) or {}).get("is_hit", False)]
    robust_hits = [record for record in hits if (criterion(record) or {}).get("vbm_margin", 0.0) > ROBUST_VBM_MARGIN]
    scaffold_records: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in successful:
        scaffold_records[key(record)[:3]].append(record)
    hit_scaffolds = {key(record)[:3] for record in hits}
    robust_scaffolds = {key(record)[:3] for record in robust_hits}

    by_linkage = {}
    for linkage in sorted({key(record)[2] for record in records}):
        family = [record for record in successful if key(record)[2] == linkage]
        family_hits = [record for record in family if (criterion(record) or {}).get("is_hit", False)]
        by_linkage[linkage] = {
            "candidates": len(family), "hits": len(family_hits),
            "hit_rate": len(family_hits) / len(family) if family else 0.0,
            "scaffolds": len({key(record)[:3] for record in family}),
            "hit_scaffolds": len({key(record)[:3] for record in family_hits}),
        }

    gate_failures = Counter()
    for record in successful:
        current = criterion(record) or {}
        for gate in ("gap_pass", "cbm_pass", "vbm_pass", "stability_pass"):
            if not current.get(gate, False):
                gate_failures[gate.removesuffix("_pass")] += 1

    table = []
    for scaffold, members in scaffold_records.items():
        scaffold_hits = [record for record in members if (criterion(record) or {}).get("is_hit", False)]
        if not scaffold_hits:
            continue
        best_margin = max((criterion(record) or {}).get("vbm_margin", float("-inf")) for record in scaffold_hits)
        table.append({
            "scaffold": list(scaffold), "candidates": len(members), "hits": len(scaffold_hits),
            "robust_hits": sum((criterion(record) or {}).get("vbm_margin", 0) > ROBUST_VBM_MARGIN for record in scaffold_hits),
            "best_vbm_margin": round(best_margin, 6),
        })
    table.sort(key=lambda row: (-row["hits"], -row["best_vbm_margin"], row["scaffold"]))
    hit_counts = [row["hits"] for row in table]
    return {
        "total": len(records), "successful": len(successful), "failed": len(failed),
        "hits": len(hits), "hit_rate": len(hits) / len(successful) if successful else 0.0,
        "robust_hits": len(robust_hits), "robust_hit_rate": len(robust_hits) / len(successful) if successful else 0.0,
        "scaffolds": len(scaffold_records), "hit_bearing_scaffolds": len(hit_scaffolds),
        "robust_hit_bearing_scaffolds": len(robust_scaffolds),
        "decoration_factor": len(hits) / len(hit_scaffolds) if hit_scaffolds else 0.0,
        "hits_per_hit_scaffold": {"min": min(hit_counts, default=0), "max": max(hit_counts, default=0),
                                    "mean": sum(hit_counts) / len(hit_counts) if hit_counts else 0.0},
        "by_linkage": by_linkage, "gate_failures": dict(gate_failures),
        "scaffold_table": table,
        "dominant_basins": table[:3],
        "threshold_grazers": [row for row in table if row["best_vbm_margin"] <= GRAZER_VBM_MARGIN],
    }


def render(summary: dict) -> str:
    lines = ["GROUND-TRUTH LANDSCAPE", "======================", f"Candidates: {summary['total']}",
             f"Successful: {summary['successful']}  Failed: {summary['failed']}",
             f"Hits: {summary['hits']} ({summary['hit_rate']:.2%})", f"Robust hits: {summary['robust_hits']} ({summary['robust_hit_rate']:.2%})",
             f"Scaffolds: {summary['scaffolds']}  Hit-bearing: {summary['hit_bearing_scaffolds']}  Robust: {summary['robust_hit_bearing_scaffolds']}",
             f"Decoration factor: {summary['decoration_factor']:.2f}x", "", "LINKAGE BREAKDOWN", "-----------------"]
    for linkage, stats in summary["by_linkage"].items():
        lines.append(f"{linkage:<20} candidates={stats['candidates']:>4} hits={stats['hits']:>3} rate={stats['hit_rate']:.2%} scaffolds={stats['scaffolds']}")
    lines += ["", "GATE FAILURES (successful candidates; gates may overlap)", "--------------------------------------------------------"]
    for gate in ("gap", "cbm", "vbm", "stability"):
        lines.append(f"{gate:<12} {summary['gate_failures'].get(gate, 0)}")
    lines += ["", "HIT-BEARING SCAFFOLDS (top 10)", "--------------------------------"]
    for row in summary["scaffold_table"][:10]:
        lines.append(f"{'/'.join(row['scaffold']):<42} hits={row['hits']:>2} robust={row['robust_hits']:>2} best_VBM_margin={row['best_vbm_margin']:+.3f}")
    lines += ["", "DOMINANT BASINS", "---------------"]
    for row in summary["dominant_basins"]:
        lines.append(f"{'/'.join(row['scaffold'])}: {row['hits']} hits")
    lines += ["", "THRESHOLD GRAZERS (best VBM margin <= 0.10 V)", "-----------------------------------------------"]
    for row in summary["threshold_grazers"]:
        lines.append(f"{'/'.join(row['scaffold'])}: {row['best_vbm_margin']:+.3f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a complete ground-truth JSONL")
    parser.add_argument("--ground-truth", type=Path, default=Path("output/ground_truth.jsonl"))
    parser.add_argument("--feasible-csv", type=Path, default=Path("data/search_space_v2_feasible.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/analysis"))
    args = parser.parse_args()
    if not args.ground_truth.is_file():
        parser.error(f"ground truth not found: {args.ground_truth}")
    expected = feasible_keys(args.feasible_csv)
    if len(expected) != EXPECTED_CANDIDATES:
        print(f"Warning: feasible CSV has {len(expected)} candidates (expected historical value {EXPECTED_CANDIDATES})")
    records = read_ground_truth(args.ground_truth)
    validate_complete(records, expected)
    summary = summarize(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ground_truth_report.txt").write_text(render(summary), encoding="utf-8")
    (args.out_dir / "ground_truth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'ground_truth_report.txt'}")
    print(f"Wrote {args.out_dir / 'ground_truth_summary.json'}")


if __name__ == "__main__":
    main()
