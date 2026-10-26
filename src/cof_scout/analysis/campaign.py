#!/usr/bin/env python3
"""Generate concise, paper-facing metrics and tables from a COF-SCOUT trace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cof_scout.analysis.run import _model_usage
from cof_scout.agent.contracts import classify_validation_scope
from validate_ara5_trace import load_events, validate_events


def _experiment_candidate(event: dict[str, Any]) -> tuple[str, str, str, str]:
    experiment = event.get("action", {}).get("experiment", {})
    return tuple(str(experiment.get(field, "?")) for field in (
        "node_id", "linker_id", "linkage", "r_group_id",
    ))  # type: ignore[return-value]


def _is_hit(event: dict[str, Any]) -> bool:
    criterion = event.get("result", {}).get("criterion", {})
    return bool(criterion.get("is_hit")) if isinstance(criterion, dict) else False


def _is_robust_hit(event: dict[str, Any]) -> bool:
    criterion = event.get("result", {}).get("criterion", {})
    return (
        _is_hit(event)
        and isinstance(criterion, dict)
        and float(criterion.get("vbm_margin", float("-inf"))) > 0.25
    )


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return campaign metrics plus rows used for reproducible tables."""
    validate_events(events)
    experiments = [event for event in events if event.get("event_type") == "experiment"]
    seen_scaffolds: set[tuple[str, str, str]] = set()
    hit_scaffolds: set[tuple[str, str, str]] = set()
    robust_scaffolds: set[tuple[str, str, str]] = set()
    experiment_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    scaffold_accumulator: dict[tuple[str, str, str], dict[str, Any]] = {}
    linkage_accumulator: dict[str, Counter[str]] = defaultdict(Counter)
    decoration_experiments = 0
    decoration_hits = 0
    first_hit_iteration: int | None = None

    for event in experiments:
        iteration = int(event["iteration"])
        candidate = _experiment_candidate(event)
        scaffold = candidate[:3]
        result = event.get("result", {})
        criterion = result.get("criterion", {}) if isinstance(result, dict) else {}
        hit = _is_hit(event)
        robust = _is_robust_hit(event)
        new_scaffold = scaffold not in seen_scaffolds
        intent = event.get("action", {}).get("experiment", {}).get("intent")
        validation_scope = event.get("validation_scope")
        if validation_scope is None:
            validation_scope = classify_validation_scope(intent, new_scaffold)
        if not new_scaffold:
            decoration_experiments += 1
            decoration_hits += int(hit)
        seen_scaffolds.add(scaffold)
        if hit:
            hit_scaffolds.add(scaffold)
            if first_hit_iteration is None:
                first_hit_iteration = iteration
        if robust:
            robust_scaffolds.add(scaffold)

        prediction_supported = (event.get("reflection") or {}).get("prediction_supported")
        prediction_adjudication = event.get("prediction_adjudication", {})
        if not isinstance(prediction_adjudication, dict):
            prediction_adjudication = {}
        stability = result.get("stability", {}) if isinstance(result, dict) else {}
        components = stability.get("components", {}) if isinstance(stability, dict) else {}
        experiment_rows.append({
            "iteration": iteration,
            "node_id": candidate[0], "linker_id": candidate[1],
            "linkage": candidate[2], "r_group_id": candidate[3],
            "intent": intent,
            "validation_scope": validation_scope,
            "is_new_scaffold": new_scaffold,
            "status": result.get("status") if isinstance(result, dict) else None,
            "gap_corrected_eV": result.get("gap_corrected_eV") if isinstance(result, dict) else None,
            "cbm_vs_nhe": result.get("cbm_vs_nhe") if isinstance(result, dict) else None,
            "vbm_vs_nhe": result.get("vbm_vs_nhe") if isinstance(result, dict) else None,
            "s_linkage": result.get("s_linkage") if isinstance(result, dict) else None,
            "stability_score": result.get(
                "stability_score", result.get("s_linkage")
            ) if isinstance(result, dict) else None,
            "stability_model": stability.get("model") if isinstance(stability, dict) else None,
            "stability_mechanism": stability.get("mechanism") if isinstance(stability, dict) else None,
            "stability_shielding": components.get("shielding") if isinstance(components, dict) else None,
            "stability_local_hydrophobicity": components.get("local_hydrophobicity") if isinstance(components, dict) else None,
            "stability_electronic_deactivation": components.get("electronic_deactivation") if isinstance(components, dict) else None,
            "stability_rigidity": components.get("rigidity") if isinstance(components, dict) else None,
            "stability_local_protection": components.get("local_protection") if isinstance(components, dict) else None,
            "gap_pass": criterion.get("gap_pass") if isinstance(criterion, dict) else None,
            "cbm_pass": criterion.get("cbm_pass") if isinstance(criterion, dict) else None,
            "vbm_pass": criterion.get("vbm_pass") if isinstance(criterion, dict) else None,
            "stability_pass": criterion.get("stability_pass") if isinstance(criterion, dict) else None,
            "electronic_pass": criterion.get("electronic_pass") if isinstance(criterion, dict) else None,
            "is_hit": hit, "is_robust_hit": robust,
            "prediction_supported": prediction_supported,
            "predicted_hit_probability": prediction_adjudication.get("hit_probability"),
            "prediction_brier_score": prediction_adjudication.get("brier_score"),
            "hit_classification_correct": prediction_adjudication.get(
                "hit_classification_correct"
            ),
            "gate_prediction_accuracy": prediction_adjudication.get("gate_accuracy"),
            "limiting_gate_correct": prediction_adjudication.get("limiting_gate_correct"),
            "predicted_limiting_gate": prediction_adjudication.get(
                "predicted_limiting_gate"
            ),
            "observed_failed_gates": ";".join(
                prediction_adjudication.get("observed_failed_gates", [])
            ),
            "reflection_recorded": event.get("reflection") is not None,
        })
        curve_rows.append({
            "iteration": iteration,
            "cumulative_hits": sum(int(row["is_hit"]) for row in experiment_rows),
            "cumulative_hit_scaffolds": len(hit_scaffolds),
            "cumulative_robust_hit_scaffolds": len(robust_scaffolds),
            "cumulative_scaffolds_tested": len(seen_scaffolds),
            "cumulative_decoration_experiments": decoration_experiments,
        })

        linkage = candidate[2]
        linkage_accumulator[linkage]["experiments"] += 1
        linkage_accumulator[linkage]["hits"] += int(hit)
        linkage_accumulator[linkage]["robust_hits"] += int(robust)

        if scaffold not in scaffold_accumulator:
            scaffold_accumulator[scaffold] = {
                "node_id": scaffold[0], "linker_id": scaffold[1],
                "linkage": scaffold[2], "first_iteration": iteration,
                "first_hit_iteration": None, "experiments": 0, "hits": 0,
                "robust_hits": 0, "r_groups": set(),
                "best_vbm_margin": float("-inf"),
                "best_all_gate_margin": float("-inf"),
            }
        row = scaffold_accumulator[scaffold]
        row["experiments"] += 1
        row["hits"] += int(hit)
        row["robust_hits"] += int(robust)
        row["r_groups"].add(candidate[3])
        if hit and row["first_hit_iteration"] is None:
            row["first_hit_iteration"] = iteration
        if isinstance(criterion, dict):
            row["best_vbm_margin"] = max(
                row["best_vbm_margin"], float(criterion.get("vbm_margin", float("-inf")))
            )
            margins = [
                criterion.get("gap_margin_floor"), criterion.get("gap_margin_ceiling"),
                criterion.get("cbm_margin"), criterion.get("vbm_margin"),
                criterion.get("stability_margin"),
            ]
            numeric_margins = [float(value) for value in margins if isinstance(value, (int, float))]
            if numeric_margins:
                row["best_all_gate_margin"] = max(
                    row["best_all_gate_margin"], min(numeric_margins)
                )

    scaffold_rows = []
    for scaffold, row in sorted(scaffold_accumulator.items()):
        scaffold_rows.append({
            **{key: value for key, value in row.items() if key != "r_groups"},
            "r_groups": ";".join(sorted(row["r_groups"])),
            "is_hit_scaffold": scaffold in hit_scaffolds,
            "is_robust_hit_scaffold": scaffold in robust_scaffolds,
        })

    linkage_rows = []
    for linkage, counts in sorted(linkage_accumulator.items()):
        linkage_hit_scaffolds = {
            scaffold for scaffold in hit_scaffolds if scaffold[2] == linkage
        }
        linkage_robust_scaffolds = {
            scaffold for scaffold in robust_scaffolds if scaffold[2] == linkage
        }
        linkage_rows.append({
            "linkage": linkage,
            "experiments": counts["experiments"],
            "hits": counts["hits"],
            "hit_rate": counts["hits"] / counts["experiments"],
            "hit_scaffolds": len(linkage_hit_scaffolds),
            "robust_hit_scaffolds": len(linkage_robust_scaffolds),
        })

    rejected = [
        event for event in events
        if event.get("event_type") in {"plan_rejected", "action_rejected"}
    ]
    accepted_actions = [
        event for event in events
        if event.get("event_type") in {
            "tool_call", "strategy_replan", "experiment", "campaign_stop",
        } and isinstance(event.get("action"), dict)
    ]
    prediction_counts = Counter(
        str((event.get("reflection") or {}).get("prediction_supported")).lower()
        for event in experiments
    )
    usage = _model_usage(events)
    brier_scores = [
        float(row["prediction_brier_score"]) for row in experiment_rows
        if isinstance(row.get("prediction_brier_score"), (int, float))
    ]
    gate_accuracies = [
        float(row["gate_prediction_accuracy"]) for row in experiment_rows
        if isinstance(row.get("gate_prediction_accuracy"), (int, float))
    ]
    limiting_scores = [
        bool(row["limiting_gate_correct"]) for row in experiment_rows
        if isinstance(row.get("limiting_gate_correct"), bool)
    ]
    hit_classification_scores = [
        bool(row["hit_classification_correct"]) for row in experiment_rows
        if isinstance(row.get("hit_classification_correct"), bool)
    ]
    first_40 = [row for row in experiment_rows if row["iteration"] <= 40]
    final_10 = [row for row in experiment_rows if row["iteration"] > 40]

    metrics = {
        "experiments": len(experiments),
        "successful_evaluations": sum(row["status"] == "success" for row in experiment_rows),
        "hits": sum(row["is_hit"] for row in experiment_rows),
        "hit_rate": sum(row["is_hit"] for row in experiment_rows) / len(experiments),
        "robust_hits": sum(row["is_robust_hit"] for row in experiment_rows),
        "scaffolds_tested": len(seen_scaffolds),
        "hit_scaffolds": len(hit_scaffolds),
        "robust_hit_scaffolds": len(robust_scaffolds),
        "first_hit_iteration": first_hit_iteration,
        "scaffold_discoveries_per_experiment": len(hit_scaffolds) / len(experiments),
        "scaffold_discovery_auc": (
            sum(row["cumulative_hit_scaffolds"] for row in curve_rows)
            / len(curve_rows)
        ),
        "raw_hits_per_hit_scaffold": (
            sum(row["is_hit"] for row in experiment_rows) / len(hit_scaffolds)
            if hit_scaffolds else 0.0
        ),
        "decoration_experiments": decoration_experiments,
        "decoration_rate": decoration_experiments / len(experiments),
        "decoration_hits": decoration_hits,
        "tool_calls": sum(event.get("event_type") == "tool_call" for event in events),
        "rejected_actions": len(rejected),
        "accepted_actions": len(accepted_actions),
        "valid_action_rate": len(accepted_actions) / (len(accepted_actions) + len(rejected)),
        "reflections_recorded": sum(row["reflection_recorded"] for row in experiment_rows),
        "prediction_supported": prediction_counts.get("true", 0),
        "prediction_falsified": prediction_counts.get("false", 0),
        "prediction_unresolved": prediction_counts.get("none", 0),
        "machine_scored_predictions": len(brier_scores),
        "mean_brier_score": (
            sum(brier_scores) / len(brier_scores) if brier_scores else None
        ),
        "mean_gate_prediction_accuracy": (
            sum(gate_accuracies) / len(gate_accuracies) if gate_accuracies else None
        ),
        "hit_classification_accuracy": (
            sum(hit_classification_scores) / len(hit_classification_scores)
            if hit_classification_scores else None
        ),
        "limiting_gate_accuracy": (
            sum(limiting_scores) / len(limiting_scores) if limiting_scores else None
        ),
        "model_usage": usage,
        "rejection_reasons": dict(Counter(str(event.get("rejection")) for event in rejected)),
        "tool_usage": dict(Counter(
            str(event.get("action", {}).get("tool_name")) for event in events
            if event.get("event_type") == "tool_call"
        )),
        "intent_counts": dict(Counter(str(row["intent"]) for row in experiment_rows)),
        "validation_scope_counts": dict(Counter(
            str(row["validation_scope"]) for row in experiment_rows
            if row["validation_scope"] is not None
        )),
        "phase_summary": {
            "experiments_1_40": {
                "experiments": len(first_40),
                "new_scaffold_experiments": sum(row["is_new_scaffold"] for row in first_40),
                "hit_scaffolds_discovered": len({
                    (row["node_id"], row["linker_id"], row["linkage"])
                    for row in first_40 if row["is_hit"]
                }),
                "decoration_experiments": sum(not row["is_new_scaffold"] for row in first_40),
            },
            "experiments_41_50": {
                "experiments": len(final_10),
                "new_scaffold_experiments": sum(row["is_new_scaffold"] for row in final_10),
                "hit_scaffolds_discovered": len({
                    (row["node_id"], row["linker_id"], row["linkage"])
                    for row in final_10 if row["is_hit"] and row["is_new_scaffold"]
                }),
                "decoration_experiments": sum(not row["is_new_scaffold"] for row in final_10),
            },
        },
    }
    return {
        "metrics": metrics,
        "linkages": linkage_rows,
        "experiments": experiment_rows,
        "scaffolds": scaffold_rows,
        "discovery_curve": curve_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        "# COF-SCOUT campaign summary",
        "",
        "## Outcome",
        "",
        f"- Experiments: {metrics['experiments']}",
        f"- Hits: {metrics['hits']} ({metrics['hit_rate']:.1%})",
        f"- Distinct hit scaffolds: {metrics['hit_scaffolds']}",
        f"- Robust hit scaffolds: {metrics['robust_hit_scaffolds']}",
        f"- First hit: experiment {metrics['first_hit_iteration']}",
        f"- Scaffolds tested: {metrics['scaffolds_tested']}",
        f"- Decoration experiments: {metrics['decoration_experiments']} ({metrics['decoration_rate']:.1%})",
        f"- Raw hits per hit scaffold: {metrics['raw_hits_per_hit_scaffold']:.2f}",
        "",
        "## Linkage outcomes",
        "",
        "| Linkage | Experiments | Hits | Hit rate | Hit scaffolds | Robust hit scaffolds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["linkages"]:
        lines.append(
            f"| {row['linkage']} | {row['experiments']} | {row['hits']} | "
            f"{row['hit_rate']:.1%} | {row['hit_scaffolds']} | "
            f"{row['robust_hit_scaffolds']} |"
        )
    lines += [
        "",
        "## Decision-process diagnostics",
        "",
        f"- Deterministic tool calls: {metrics['tool_calls']}",
        f"- Accepted scientist actions: {metrics['accepted_actions']}",
        f"- Rejected scientist actions: {metrics['rejected_actions']}",
        f"- Valid-action rate: {metrics['valid_action_rate']:.1%}",
        f"- Predictions supported: {metrics['prediction_supported']}",
        f"- Predictions falsified: {metrics['prediction_falsified']}",
        f"- Machine-scored predictions: {metrics['machine_scored_predictions']}",
        f"- Mean Brier score: {metrics['mean_brier_score']}",
        f"- Mean gate-prediction accuracy: {metrics['mean_gate_prediction_accuracy']}",
        f"- Complete-screen classification accuracy: "
        f"{metrics['hit_classification_accuracy']}",
        f"- Limiting-gate accuracy: {metrics['limiting_gate_accuracy']}",
        f"- Reflections recorded: {metrics['reflections_recorded']}/{metrics['experiments']}",
        f"- Cross-scaffold hypothesis validations: "
        f"{metrics['validation_scope_counts'].get('cross_scaffold_hypothesis_validation', 0)}",
        f"- Local scaffold confirmations: "
        f"{metrics['validation_scope_counts'].get('local_scaffold_confirmation', 0)}",
        "",
        "## Phase behavior",
        "",
    ]
    for name, phase in metrics["phase_summary"].items():
        phase_label = {
            "experiments_1_40": "experiments 1–40",
            "experiments_41_50": "experiments 41–50",
        }.get(name, name.replace("_", " "))
        lines += [
            f"- {phase_label}: {phase['new_scaffold_experiments']} new-scaffold "
            f"experiments, {phase['hit_scaffolds_discovered']} hit scaffolds discovered, "
            f"{phase['decoration_experiments']} decoration experiments.",
        ]
    lines += [
        "",
        "## Rejected-action reasons",
        "",
    ]
    for reason, count in sorted(
        metrics["rejection_reasons"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- {count} × {reason}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    events = load_events(args.trace)
    result = summarize(events)
    out_dir = args.out_dir or args.trace.parent / "campaign_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "campaign_metrics.json").write_text(
        json.dumps({"metrics": result["metrics"], "linkages": result["linkages"]}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "campaign_summary.md").write_text(
        _render_summary(result), encoding="utf-8"
    )
    _write_csv(out_dir / "experiments.csv", result["experiments"])
    _write_csv(out_dir / "scaffolds.csv", result["scaffolds"])
    _write_csv(out_dir / "discovery_curve.csv", result["discovery_curve"])
    print(f"Wrote COF-SCOUT campaign analysis to {out_dir}")


if __name__ == "__main__":
    main()
