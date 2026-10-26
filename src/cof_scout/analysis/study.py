#!/usr/bin/env python3
"""Aggregate preregistered COF-SCOUT campaigns and matched lookup controls."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cof_scout.analysis.campaign import summarize as summarize_campaign
from cof_scout.analysis.trace import load_events


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "data" / "ara5_publication_study_v1.json"
DEFAULT_RUNS_ROOT = ROOT / "output" / "runs"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "studies"
PRIMARY_METRIC = "hit_scaffolds"
SUMMARY_METRICS = (
    "experiments",
    "successful_evaluations",
    "hits",
    "robust_hits",
    "hit_scaffolds",
    "robust_hit_scaffolds",
    "first_hit_iteration",
    "scaffolds_tested",
    "scaffold_discovery_auc",
    "decoration_rate",
    "mean_brier_score",
    "mean_gate_prediction_accuracy",
    "limiting_gate_accuracy",
    "hit_classification_accuracy",
    "cross_scaffold_hypothesis_validations",
    "prediction_supported",
    "prediction_falsified",
    "high_confidence_falsifications",
    "hypothesis_updates",
    "strategy_updates",
    "accepted_actions",
    "rejected_actions",
    "tool_calls",
    "model_prompt_tokens",
    "model_completion_tokens",
    "model_usd",
    "wall_time_s",
)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1 or samples < 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[generator.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(draw))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def distribution(
    values: Iterable[int | float], *, label: str, bootstrap_samples: int = 10000
) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "n": 0,
            "values": [],
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "interquartile_range": None,
            "q1": None,
            "q3": None,
            "bootstrap_95_percent_confidence_interval": None,
        }
    seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)
    lower, upper = _bootstrap_mean_ci(
        numeric, samples=bootstrap_samples, seed=seed
    )
    q1 = _percentile(numeric, 0.25)
    q3 = _percentile(numeric, 0.75)
    return {
        "n": len(numeric),
        "values": numeric,
        "mean": statistics.fmean(numeric),
        "standard_deviation": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": statistics.median(numeric),
        "interquartile_range": q3 - q1,
        "q1": q1,
        "q3": q3,
        "bootstrap_95_percent_confidence_interval": [lower, upper],
    }


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wall_time_seconds(
    metadata: dict[str, Any], events: list[dict[str, Any]]
) -> float | None:
    started = _parse_timestamp(metadata.get("started_at"))
    completed = _parse_timestamp(metadata.get("completed_at"))
    if started is not None and completed is not None:
        return max(0.0, (completed - started).total_seconds())
    event_times = [
        parsed
        for parsed in (_parse_timestamp(event.get("timestamp")) for event in events)
        if parsed is not None
    ]
    if len(event_times) >= 2:
        return max(0.0, (max(event_times) - min(event_times)).total_seconds())
    return None


def _reasoning_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    plan = next(
        (event.get("plan", {}) for event in events if event.get("event_type") == "campaign_plan"),
        {},
    )
    hypotheses = plan.get("hypotheses", []) if isinstance(plan, dict) else []
    status_counts: Counter[str] = Counter()
    hypothesis_updates = 0
    strategy_updates = sum(
        event.get("event_type") == "strategy_replan" for event in events
    )
    high_confidence_falsifications = 0
    for event in events:
        if event.get("event_type") != "experiment":
            continue
        reflection = event.get("reflection")
        if isinstance(reflection, dict):
            updates = reflection.get("hypothesis_updates", [])
            if isinstance(updates, list):
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    hypothesis_updates += 1
                    status_counts[str(update.get("status", "unknown"))] += 1
            if reflection.get("strategy_update"):
                strategy_updates += 1
        adjudication = event.get("prediction_adjudication", {})
        if not isinstance(adjudication, dict):
            continue
        probability = adjudication.get("hit_probability")
        if (
            isinstance(probability, (int, float))
            and float(probability) >= 0.70
            and adjudication.get("observed_hit") is False
        ):
            high_confidence_falsifications += 1
    return {
        "initial_hypotheses": len(hypotheses) if isinstance(hypotheses, list) else 0,
        "hypothesis_updates": hypothesis_updates,
        "hypothesis_status_counts": dict(sorted(status_counts.items())),
        "strategy_updates": strategy_updates,
        "high_confidence_falsifications": high_confidence_falsifications,
    }


def _run_summary(
    run_id: str,
    condition: str,
    run_dir: Path,
    expected_budget: int,
    expected_model: str | None,
    expected_oracle_mode: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_path = run_dir / "trace.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing trace for {run_id}: {trace_path}")
    events = load_events(trace_path)
    campaign = summarize_campaign(events)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else {}
    )
    metrics = dict(campaign["metrics"])
    usage = metrics.get("model_usage", {})
    if not isinstance(usage, dict):
        usage = {}
    reasoning = _reasoning_metrics(events)
    metrics.update(reasoning)
    metrics.update({
        "cross_scaffold_hypothesis_validations": metrics.get(
            "validation_scope_counts", {}
        ).get("cross_scaffold_hypothesis_validation", 0),
        "model_prompt_tokens": usage.get("prompt_tokens"),
        "model_completion_tokens": usage.get("completion_tokens"),
        "model_usd": usage.get("usd"),
        "wall_time_s": _wall_time_seconds(metadata, events),
    })
    completed = metrics["experiments"] == expected_budget
    expected_architecture = {
        "adaptive_ara5": "ara5_scientist",
        "one_shot_llm_portfolio": "ara5_static_outcome_blind_portfolio",
    }.get(condition)
    quality_checks = {
        "fixed_horizon_completion": completed,
        "metadata_completed_budget": metadata.get("run_status") == "completed_budget",
        "architecture_identity": metadata.get("architecture") == expected_architecture,
        "model_identity": (
            expected_model is None or metadata.get("model") == expected_model
        ),
        "oracle_identity": (
            expected_oracle_mode is None
            or metadata.get("oracle_mode") == expected_oracle_mode
        ),
        "prediction_schema_completeness": (
            metrics.get("machine_scored_predictions") == expected_budget
        ),
        "no_campaign_failure_event": not any(
            event.get("event_type") == "campaign_failed" for event in events
        ),
    }
    run_record = {
        "run_id": run_id,
        "condition": condition,
        "trace": str(trace_path),
        "complete_fixed_horizon": completed,
        "quality_pass": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "expected_experiments": expected_budget,
        "metadata_run_status": metadata.get("run_status"),
        "architecture": metadata.get("architecture"),
        "model": metadata.get("model"),
        "stability_model": metadata.get("stability_model"),
        "reference_landscape_sha256": metadata.get("reference_landscape_sha256"),
        "scientist_system_prompt_sha256": metadata.get(
            "scientist_system_prompt_sha256"
        ),
        **{name: metrics.get(name) for name in SUMMARY_METRICS},
        "hypothesis_status_counts": reasoning["hypothesis_status_counts"],
    }
    return run_record, campaign


def _condition_chemistry(
    condition_runs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    scaffold_runs: dict[str, set[str]] = defaultdict(set)
    scaffold_hit_candidates: Counter[str] = Counter()
    linkage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    gate_failures: Counter[str] = Counter()
    hit_nodes: Counter[str] = Counter()
    hit_linkers: Counter[str] = Counter()
    pooled_hit_scaffolds: set[str] = set()

    for run_record, campaign in condition_runs:
        for row in campaign["experiments"]:
            linkage = str(row["linkage"])
            linkage_counts[linkage]["experiments"] += 1
            linkage_counts[linkage]["hits"] += int(row["is_hit"])
            for gate in ("gap", "cbm", "vbm", "stability"):
                if row.get(f"{gate}_pass") is False:
                    gate_failures[gate] += 1
        for row in campaign["scaffolds"]:
            if not row["is_hit_scaffold"]:
                continue
            scaffold = "/".join((row["node_id"], row["linker_id"], row["linkage"]))
            pooled_hit_scaffolds.add(scaffold)
            scaffold_runs[scaffold].add(run_record["run_id"])
            scaffold_hit_candidates[scaffold] += int(row["hits"])
            hit_nodes[row["node_id"]] += 1
            hit_linkers[row["linker_id"]] += 1

    repeated = [
        {
            "scaffold": scaffold,
            "run_count": len(run_ids),
            "run_ids": sorted(run_ids),
            "hit_candidates": scaffold_hit_candidates[scaffold],
        }
        for scaffold, run_ids in scaffold_runs.items()
        if len(run_ids) >= 2
    ]
    repeated.sort(key=lambda row: (-row["run_count"], row["scaffold"]))
    pooled_by_linkage = Counter(scaffold.rsplit("/", 1)[1] for scaffold in pooled_hit_scaffolds)
    return {
        "pooled_distinct_hit_scaffolds": len(pooled_hit_scaffolds),
        "hit_scaffolds_by_linkage": dict(sorted(pooled_by_linkage.items())),
        "linkage_experiment_outcomes": {
            linkage: dict(counts) for linkage, counts in sorted(linkage_counts.items())
        },
        "failed_gate_counts": dict(sorted(gate_failures.items())),
        "repeated_hit_scaffolds": repeated,
        "hit_scaffold_node_counts": dict(sorted(hit_nodes.items())),
        "hit_scaffold_linker_counts": dict(sorted(hit_linkers.items())),
        "pooled_hit_scaffolds": sorted(pooled_hit_scaffolds),
    }


def _pairwise_superiority(first: list[float], second: list[float]) -> float | None:
    if not first or not second:
        return None
    score = 0.0
    for left in first:
        for right in second:
            score += 1.0 if left > right else 0.5 if left == right else 0.0
    return score / (len(first) * len(second))


def _condition_comparison(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    first_values = first[PRIMARY_METRIC]["values"]
    second_values = second[PRIMARY_METRIC]["values"]
    if not first_values or not second_values:
        return {}
    return {
        "metric": "distinct_hit_scaffolds_at_50",
        "mean_difference": statistics.fmean(first_values) - statistics.fmean(second_values),
        "median_difference": statistics.median(first_values) - statistics.median(second_values),
        "pairwise_superiority_probability": _pairwise_superiority(
            first_values, second_values
        ),
    }


def _baseline_summaries(
    baseline_results: dict[str, Any] | None,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not baseline_results:
        return {}
    methods = baseline_results.get("methods", baseline_results)
    if not isinstance(methods, dict):
        return {}
    summaries = {}
    for name, method in sorted(methods.items()):
        if not isinstance(method, dict):
            continue
        records = method.get("per_seed", [])
        if not isinstance(records, list):
            continue
        summaries[name] = {
            "runs": len(records),
            "hit_scaffolds": distribution(
                (row.get("scaffolds", 0) for row in records),
                label=f"baseline:{name}:scaffolds",
                bootstrap_samples=bootstrap_samples,
            ),
            "raw_hits": distribution(
                (row.get("hits", 0) for row in records),
                label=f"baseline:{name}:hits",
                bootstrap_samples=bootstrap_samples,
            ),
        }
    return summaries


def aggregate_study(
    config: dict[str, Any],
    runs_root: Path,
    *,
    baseline_results: dict[str, Any] | None = None,
    require_complete: bool = True,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    budget = int(config["campaign_configuration"]["oracle_attempts"])
    condition_results: dict[str, Any] = {}
    missing_runs: list[str] = []
    incomplete_runs: list[str] = []
    failed_quality_runs: list[str] = []
    execution = config.get("execution", {})

    for condition, specification in config.get("conditions", {}).items():
        run_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for run_id in specification.get("run_ids", []):
            run_dir = runs_root / run_id
            if not (run_dir / "trace.jsonl").is_file():
                missing_runs.append(run_id)
                continue
            run_record, campaign = _run_summary(
                run_id,
                condition,
                run_dir,
                budget,
                execution.get("model"),
                execution.get("oracle_mode"),
            )
            if not run_record["complete_fixed_horizon"]:
                incomplete_runs.append(run_id)
            if not run_record["quality_pass"]:
                failed_quality_runs.append(run_id)
            run_pairs.append((run_record, campaign))

        records = [record for record, _ in run_pairs]
        summaries = {
            metric: distribution(
                (
                    record[metric] for record in records
                    if isinstance(record.get(metric), (int, float))
                    and not isinstance(record.get(metric), bool)
                ),
                label=f"{condition}:{metric}",
                bootstrap_samples=bootstrap_samples,
            )
            for metric in SUMMARY_METRICS
        }
        condition_results[condition] = {
            "expected_replicates": int(specification.get("replicates", len(records))),
            "completed_replicates": len(records),
            "runs": records,
            "summary": summaries,
            "chemistry": _condition_chemistry(run_pairs),
        }

    reference_hashes = sorted({
        str(run["reference_landscape_sha256"])
        for section in condition_results.values()
        for run in section["runs"]
        if run.get("reference_landscape_sha256")
    })
    if len(reference_hashes) > 1:
        failed_quality_runs.append("inconsistent_reference_landscape_hashes")

    if require_complete and (missing_runs or incomplete_runs or failed_quality_runs):
        problems = []
        if missing_runs:
            problems.append(f"missing runs: {', '.join(missing_runs)}")
        if incomplete_runs:
            problems.append(f"incomplete runs: {', '.join(incomplete_runs)}")
        if failed_quality_runs:
            problems.append(f"failed quality checks: {', '.join(failed_quality_runs)}")
        raise ValueError("Definitive study is incomplete; " + "; ".join(problems))

    comparisons = {}
    adaptive = condition_results.get("adaptive_ara5", {}).get("summary")
    static = condition_results.get("one_shot_llm_portfolio", {}).get("summary")
    if adaptive and static:
        comparisons["adaptive_vs_static"] = _condition_comparison(adaptive, static)

    return {
        "study_id": config.get("study_id"),
        "study_status": config.get("status"),
        "primary_endpoint": config.get("primary_endpoint"),
        "campaign_budget": budget,
        "analysis_unit": "campaign",
        "complete": not missing_runs and not incomplete_runs and not failed_quality_runs,
        "missing_runs": missing_runs,
        "incomplete_runs": incomplete_runs,
        "failed_quality_runs": failed_quality_runs,
        "reference_landscape_sha256": reference_hashes[0] if len(reference_hashes) == 1 else None,
        "conditions": condition_results,
        "primary_endpoint_comparisons": comparisons,
        "lookup_baselines": _baseline_summaries(
            baseline_results, bootstrap_samples
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                for key, value in row.items()
            })


def _format_distribution(summary: dict[str, Any]) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "not available"
    interval = summary["bootstrap_95_percent_confidence_interval"]
    return (
        f"{summary['mean']:.2f} ± {summary['standard_deviation']:.2f}; "
        f"median {summary['median']:.2f}, IQR {summary['interquartile_range']:.2f}; "
        f"bootstrap 95% CI {interval[0]:.2f}–{interval[1]:.2f}"
    )


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        f"# COF-SCOUT study results — {result.get('study_id')}",
        "",
        f"Study complete: **{result['complete']}**. Campaign budget: "
        f"{result['campaign_budget']} attempted oracle evaluations.",
        "",
        "## Primary endpoint",
        "",
    ]
    for condition, section in result["conditions"].items():
        label = condition.replace("_", " ")
        summary = section["summary"][PRIMARY_METRIC]
        lines.append(
            f"- {label}: {_format_distribution(summary)} "
            f"across {section['completed_replicates']} campaigns."
        )
    comparison = result["primary_endpoint_comparisons"].get("adaptive_vs_static")
    if comparison:
        lines.extend([
            "",
            "Adaptive minus static mean difference: "
            f"{comparison['mean_difference']:.2f} hit scaffolds; pairwise superiority "
            f"probability {comparison['pairwise_superiority_probability']:.3f}.",
        ])
    lines.extend(["", "## Chemistry-facing descriptive results", ""])
    for condition, section in result["conditions"].items():
        chemistry = section["chemistry"]
        lines.extend([
            f"### {condition.replace('_', ' ').title()}",
            "",
            f"Pooled distinct hit scaffolds: {chemistry['pooled_distinct_hit_scaffolds']}.",
            "",
            "Hit-scaffold counts by linkage: "
            f"`{json.dumps(chemistry['hit_scaffolds_by_linkage'], sort_keys=True)}`.",
            "",
            "Failed-gate counts: "
            f"`{json.dumps(chemistry['failed_gate_counts'], sort_keys=True)}`.",
            "",
            f"Repeated hit scaffolds across campaigns: "
            f"{len(chemistry['repeated_hit_scaffolds'])}.",
            "",
        ])
    lines.extend([
        "## Reporting note",
        "",
        "These are computational screening outcomes. Chemical patterns are descriptive "
        "until supported prospectively across independent campaigns and should not be "
        "presented as experimental photocatalytic validation.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "study_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "study_summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    run_rows = [
        run
        for section in result["conditions"].values()
        for run in section["runs"]
    ]
    _write_csv(output_dir / "campaign_replicates.csv", run_rows)
    repeated_rows = [
        {"condition": condition, **row}
        for condition, section in result["conditions"].items()
        for row in section["chemistry"]["repeated_hit_scaffolds"]
    ]
    _write_csv(output_dir / "repeated_hit_scaffolds.csv", repeated_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baselines = (
        json.loads(args.baselines.read_text(encoding="utf-8"))
        if args.baselines else None
    )
    result = aggregate_study(
        config,
        args.runs_root,
        baseline_results=baselines,
        require_complete=not args.allow_incomplete,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_dir = args.out_dir or DEFAULT_OUTPUT_ROOT / str(config["study_id"]) / "analysis"
    write_outputs(result, output_dir)
    print(f"Wrote COF-SCOUT study analysis to {output_dir}")


if __name__ == "__main__":
    main()
