#!/usr/bin/env python3
"""Render an experimentalist-facing Markdown audit from a COF-SCOUT event trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cof_scout.agent.contracts import classify_validation_scope
from cof_scout.agent.memory import build_notebook, notebook_sha256


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Trace line {line_number} is not an object")
            events.append(value)
    return events


def _fenced(value: Any, language: str = "text") -> list[str]:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    fence = "````" if "```" in text else "```"
    return [f"{fence}{language}", text.rstrip(), fence]


def _candidate(event: dict[str, Any]) -> str:
    experiment = event.get("action", {}).get("experiment", {})
    return "/".join(str(experiment.get(field, "?")) for field in (
        "node_id", "linker_id", "linkage", "r_group_id"
    ))


def _gate_table(result: dict[str, Any]) -> list[str]:
    criterion = result.get("criterion", {}) if isinstance(result, dict) else {}
    if result.get("status") != "success" or not isinstance(criterion, dict):
        return [f"Oracle status: `{result.get('status', 'unknown')}`"]
    stability = result.get("stability")
    stability_model = (
        stability.get("model") if isinstance(stability, dict) else "legacy_categorical"
    )
    stability_value = result.get("stability_score", result.get("s_linkage", 0))
    electronic_pass = criterion.get("electronic_pass")
    if electronic_pass is None:
        electronic_pass = all(
            criterion.get(key) for key in ("gap_pass", "cbm_pass", "vbm_pass")
        )
    rows = [
        "| Gate | Value | Pass | Margin |",
        "|---|---:|:---:|---:|",
        f"| Gap | {result.get('gap_corrected_eV', 0):.3f} eV | {criterion.get('gap_pass')} | floor {criterion.get('gap_margin_floor')}; ceiling {criterion.get('gap_margin_ceiling')} |",
        f"| CBM | {result.get('cbm_vs_nhe', 0):+.3f} V | {criterion.get('cbm_pass')} | {criterion.get('cbm_margin')} |",
        f"| VBM | {result.get('vbm_vs_nhe', 0):+.3f} V | {criterion.get('vbm_pass')} | {criterion.get('vbm_margin')} |",
        f"| Stability ({stability_model}) | {stability_value:.3f} | {criterion.get('stability_pass')} | {criterion.get('stability_margin')} |",
        "",
        f"**Electronic gates only:** {'PASS' if electronic_pass else 'FAIL'}",
        f"**Four-gate outcome:** {'HIT' if criterion.get('is_hit') else 'MISS'}",
    ]
    if isinstance(stability, dict):
        rows.extend(["", "**MCAP evidence:**", *_fenced(stability, "json")])
    return rows


def _is_usage(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("prompt_tokens"), int)
        and isinstance(value.get("completion_tokens"), int)
    )


def _model_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate top-level call costs without double-counting reflection attempts."""
    total: dict[str, Any] = {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "usd": 0.0, "estimated": False,
    }

    def add(usage: dict[str, Any], calls: int = 1) -> None:
        total["calls"] += calls
        total["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        total["completion_tokens"] += int(usage.get("completion_tokens", 0))
        total["usd"] += float(usage.get("usd", 0.0))
        total["estimated"] = total["estimated"] or bool(usage.get("estimated", True))

    for event in events:
        cost = event.get("cost")
        if _is_usage(cost):
            add(cost)
            continue
        if not isinstance(cost, dict):
            continue
        action = cost.get("action")
        if _is_usage(action):
            add(action)
        reflection = cost.get("reflection")
        if _is_usage(reflection):
            reflection_calls = sum(
                1 for attempt in event.get("reflection_attempts", [])
                if isinstance(attempt, dict) and _is_usage(attempt.get("cost"))
                and (
                    int(attempt["cost"].get("prompt_tokens", 0))
                    + int(attempt["cost"].get("completion_tokens", 0)) > 0
                )
            )
            add(reflection, calls=reflection_calls)
    return total


def render(events: list[dict[str, Any]], metadata: dict[str, Any] | None,
           include_context: bool = False) -> str:
    notebook = build_notebook(events)
    facts = notebook["deterministic_facts"]
    usage = _model_usage(events)
    tool_calls = sum(event.get("event_type") == "tool_call" for event in events)
    rejected_actions = sum(event.get("event_type") == "action_rejected" for event in events)
    lines = [
        "# COF-SCOUT Scientist experimental audit",
        "",
        "This report records the scientist model’s emitted scientific rationale, every",
        "structured decision, the exact deterministic evidence returned by tools and the",
        "oracle, and all hypothesis or strategy revisions. It does not claim access to",
        "unobserved private chain-of-thought.",
        "",
    ]
    if metadata:
        lines += [
            "## Reproducibility",
            "",
            f"- Run ID: `{metadata.get('run_id', 'unknown')}`",
            f"- Architecture: `{metadata.get('architecture', 'unknown')}`",
            f"- Model: `{metadata.get('model', 'unknown')}`",
            f"- Trace schema: `{metadata.get('trace_schema_version', 'unknown')}`",
            f"- Oracle budget: `{metadata.get('max_experiments', 'unknown')}`",
            f"- Run status: `{metadata.get('run_status', 'unknown')}`",
            f"- Experiments completed: `{metadata.get('experiments_completed', 'unknown')}`",
            f"- Scientist prompt SHA-256: `{metadata.get('scientist_system_prompt_sha256', 'unknown')}`",
            "",
        ]

    lines += [
        "## Campaign outcome",
        "",
        f"- Oracle experiments: {facts['experiments']}",
        f"- Successful calculations: {facts['successful_evaluations']}",
        f"- Candidate hits: {facts['hits']}",
        f"- Distinct hit scaffolds: {facts['hit_scaffolds']}",
        f"- Robust hit scaffolds: {facts['robust_hit_scaffolds']}",
        f"- Deterministic tool calls: {tool_calls}",
        f"- Rejected scientist actions: {rejected_actions}",
        f"- Model calls with usage records: {usage['calls']}",
        f"- Model tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion",
        f"- Reported model cost: ${usage['usd']:.6f}{' (estimated)' if usage['estimated'] else ''}",
        f"- Final notebook SHA-256: `{notebook_sha256(notebook)}`",
        "",
    ]

    plan_event = next((event for event in events if event.get("event_type") == "campaign_plan"), None)
    if plan_event:
        plan = plan_event.get("plan", {})
        lines += [
            "## Initial scientific plan",
            "",
            "### Emitted rationale",
            "",
        ]
        lines += _fenced(plan_event.get("scientific_rationale", "(none)"))
        lines += [
            "",
            f"**Objective:** {plan.get('objective', 'not recorded')}",
            "",
            f"**Initial landscape assessment:** {plan.get('landscape_assessment', 'not recorded')}",
            "",
            f"**Strategy:** {plan.get('strategy', 'not recorded')}",
            "",
            "### Initial hypotheses",
            "",
        ]
        for hypothesis in plan.get("hypotheses", []):
            lines += [
                f"- **{hypothesis.get('hypothesis_id', '?')} — {hypothesis.get('statement', '')}**",
                f"  - Confidence: `{hypothesis.get('confidence', 'unknown')}`",
                f"  - Chemical rationale: {hypothesis.get('rationale', '')}",
                f"  - Discriminating test: {hypothesis.get('discriminating_test', '')}",
            ]
        lines += ["", "### Decision principles", ""]
        lines += [f"- {item}" for item in plan.get("decision_principles", [])]
        lines += ["", "### Saturation criteria", ""]
        lines += [f"- {item}" for item in plan.get("saturation_criteria", [])]
        lines += [""]
        if include_context:
            lines += ["<details><summary>Exact planning context</summary>", ""]
            lines += _fenced(plan_event.get("context", ""))
            lines += ["", "</details>", ""]

    lines += ["## Decision and evidence timeline", ""]
    timeline_scaffolds: set[tuple[str, str, str]] = set()
    for event in events:
        event_type = event.get("event_type")
        if event_type == "tool_call":
            action = event.get("action", {})
            lines += [
                f"### Before experiment {event.get('iteration', '?')} — deterministic tool",
                "",
                f"**Tool:** `{action.get('tool_name', 'unknown')}`",
                "",
                f"**Decision purpose:** {action.get('purpose', '')}",
                "",
                f"**Expected information:** {action.get('expected_information', '')}",
                "",
                "**Requested tool arguments:**",
                "",
            ]
            lines += _fenced(action.get("tool_arguments", {}), "json")
            lines += [
                "",
                "**Scientist’s emitted rationale:**",
                "",
            ]
            lines += _fenced(event.get("scientific_rationale", "(none)"))
            lines += ["", "**Returned deterministic evidence:**", ""]
            lines += _fenced(event.get("tool_evidence", {}), "json")
            lines += [""]
            if include_context:
                lines += ["<details><summary>Exact action context</summary>", ""]
                lines += _fenced(event.get("context", ""))
                lines += ["", "</details>", ""]

        elif event_type == "strategy_replan":
            lines += [
                f"### Before experiment {event.get('iteration', '?')} — strategy revision",
                "",
                f"**Revision:** {event.get('strategy_revision', '')}",
                "",
                f"**Evidence/reason:** {event.get('reason', '')}",
                "",
            ]
            lines += _fenced(event.get("scientific_rationale", "(none)"))
            lines += [""]

        elif event_type == "campaign_resume":
            lines += [
                "### Campaign resumed",
                "",
                f"- Experiments already completed: {event.get('completed_experiments', '?')}",
                f"- Request timeout: {event.get('request_timeout_s', '?')} s",
                f"- Reason: {event.get('reason', '')}",
                "",
            ]

        elif event_type == "action_rejected":
            lines += [
                f"### Rejected decision before experiment {event.get('iteration', '?')}",
                "",
                f"- Reason: `{event.get('rejection', 'unknown')}`",
                f"- Proposed structured action: `{json.dumps(event.get('action'), ensure_ascii=False)}`",
                "",
            ]
            if event.get("scientific_rationale"):
                lines += _fenced(event["scientific_rationale"])
                lines += [""]
            if event.get("repair_evidence"):
                lines += ["**Deterministic compatibility and repair evidence:**", ""]
                lines += _fenced(event["repair_evidence"], "json")
                lines += [""]

        elif event_type == "model_call_failed":
            lines += [
                f"### Model call failure — `{event.get('stage', 'unknown stage')}`",
                "",
                f"- Before experiment: `{event.get('iteration', 'planning')}`",
                f"- Error: `{event.get('error', 'unknown')}`",
                "",
            ]

        elif event_type == "experiment":
            action = event.get("action", {})
            experiment = action.get("experiment", {})
            prediction = experiment.get("prediction", {})
            scaffold = tuple(str(experiment.get(field, "?")) for field in (
                "node_id", "linker_id", "linkage",
            ))
            is_new_scaffold = scaffold not in timeline_scaffolds
            validation_scope = event.get("validation_scope")
            if validation_scope is None:
                validation_scope = classify_validation_scope(
                    experiment.get("intent"), is_new_scaffold
                )
            lines += [
                f"### Experiment {event.get('iteration', '?')} — `{_candidate(event)}`",
                "",
                f"**Intent:** `{experiment.get('intent', 'unknown')}`",
                "",
                f"**Scientific purpose:** {experiment.get('purpose', '')}",
                "",
                f"**Hypothesis tested:** `{experiment.get('hypothesis_id', 'none')}`",
                "",
                f"**Expected information:** {experiment.get('expected_information', '')}",
                "",
                "#### Pre-experiment emitted rationale",
                "",
            ]
            if validation_scope is not None:
                lines += [f"**Validation scope:** `{validation_scope}`", ""]
            lines += _fenced(event.get("scientific_rationale", "(none)"))
            lines += [
                "",
                "#### Pre-registered prediction",
                "",
                f"- Axis: `{prediction.get('axis', 'unknown')}`",
                f"- Direction: `{prediction.get('direction', 'unknown')}`",
                f"- Reference iteration: `{prediction.get('reference_iteration', 'none')}`",
                f"- Statement: {prediction.get('statement', '')}",
                f"- Prospective hit probability: "
                f"`{prediction.get('hit_probability', 'legacy/unrecorded')}`",
                f"- Predicted gate outcomes: "
                f"`{prediction.get('gate_predictions', 'legacy/unrecorded')}`",
                f"- Predicted limiting gate: "
                f"`{prediction.get('limiting_gate', 'legacy/unrecorded')}`",
                f"- Falsification rule: "
                f"{prediction.get('falsification_rule', 'legacy/unrecorded')}",
                "",
                "#### Pre-registered branching decision",
                "",
                f"- If supported: {experiment.get('if_supported', '')}",
                f"- If falsified: {experiment.get('if_falsified', '')}",
                f"- If inconclusive: {experiment.get('if_inconclusive', '')}",
                "",
                "#### Oracle observation",
                "",
            ]
            lines += _gate_table(event.get("result", {}))
            adjudication = event.get("prediction_adjudication")
            if isinstance(adjudication, dict):
                lines += [
                    "",
                    "#### Deterministic prediction adjudication",
                    "",
                    f"- Brier score: "
                    f"`{adjudication.get('brier_score', 'not scoreable')}`",
                    f"- Gate-prediction accuracy: "
                    f"`{adjudication.get('gate_accuracy', 'not scoreable')}`",
                    f"- Complete-screen classification correct: "
                    f"`{adjudication.get('hit_classification_correct', 'not scoreable')}`",
                    f"- Limiting-gate prediction correct: "
                    f"`{adjudication.get('limiting_gate_correct', 'not scoreable')}`",
                    f"- Observed failed gates: "
                    f"`{adjudication.get('observed_failed_gates', [])}`",
                ]
            lines += ["", "#### Post-experiment scientific reflection", ""]
            lines += _fenced(event.get("reflection_scientific_rationale", "(none recorded)"))
            reflection = event.get("reflection")
            if isinstance(reflection, dict):
                lines += [
                    "",
                    f"- Observation summary: {reflection.get('observation_summary', '')}",
                    f"- Prediction supported: `{reflection.get('prediction_supported')}`",
                    f"- Causal interpretation: {reflection.get('causal_interpretation', '')}",
                    f"- Strategy update: "
                    f"{reflection.get('strategy_update') or 'No material policy change.'}",
                    f"- Next priority: {reflection.get('next_priority', '')}",
                    "- Hypothesis updates:",
                ]
                for update in reflection.get("hypothesis_updates", []):
                    lines.append(
                        f"  - `{update.get('hypothesis_id')}` → `{update.get('status')}` / "
                        f"`{update.get('confidence')}`: {update.get('evidence')}"
                    )
            else:
                attempts = event.get("reflection_attempts", [])
                errors = [attempt.get("parse_error") for attempt in attempts if attempt.get("parse_error")]
                lines += ["", f"Reflection unavailable: `{'; '.join(errors) or 'not recorded'}`"]
            lines += [
                "",
                f"Notebook transition: `{event.get('notebook_sha256_before', 'unknown')}` → "
                f"`{event.get('notebook_sha256_after', 'unknown')}`",
                f"Compact decision-state transition: "
                f"`{event.get('decision_state_sha256_before', 'not recorded')}` → "
                f"`{event.get('decision_state_sha256_after', 'not recorded')}`",
                "",
            ]
            if include_context:
                lines += ["<details><summary>Exact action context</summary>", ""]
                lines += _fenced(event.get("action_context", ""))
                lines += ["", "</details>", ""]
                lines += ["<details><summary>Exact reflection context</summary>", ""]
                lines += _fenced(event.get("reflection_context", ""))
                lines += ["", "</details>", ""]
            timeline_scaffolds.add(scaffold)

        elif event_type == "campaign_stop":
            lines += [
                "### Campaign stopped",
                "",
                f"**Pre-registered saturation criterion:** "
                f"{event.get('saturation_criterion', 'not supplied')}",
                "",
                f"**Reason:** {event.get('reason', '')}",
                "",
            ]

    interpretations = notebook["scientist_interpretations"]
    lines += ["## Final hypothesis ledger", ""]
    for hypothesis_id, hypothesis in interpretations["hypotheses"].items():
        lines += [
            f"### {hypothesis_id} — `{hypothesis.get('status')}` / `{hypothesis.get('confidence')}`",
            "",
            hypothesis.get("statement") or "(statement unavailable)",
            "",
        ]
        for evidence in hypothesis.get("evidence_history", []):
            lines.append(
                f"- Experiment {evidence.get('iteration')}: `{evidence.get('status')}` — "
                f"{evidence.get('evidence')}"
            )
        lines += [""]
    lines += ["## Strategy revision history", ""]
    for revision in interpretations["strategy_revisions"]:
        lines.append(
            f"- Before/after experiment {revision.get('iteration')}: "
            f"{revision.get('revision')}"
        )
    lines += ["", f"**Final stated priority:** {interpretations.get('next_priority') or 'not recorded'}", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a COF-SCOUT experimental audit")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-context", action="store_true")
    args = parser.parse_args()
    if not args.trace.is_file():
        sys.exit(f"Trace not found: {args.trace}")
    events = _load_events(args.trace)
    metadata_path = args.trace.parent / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else None
    )
    report = render(events, metadata, args.include_context)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
