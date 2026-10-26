#!/usr/bin/env python3
"""Validate ordering, uniqueness, and notebook hashes in a COF-SCOUT trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cof_scout.agent.contracts import TRACE_SCHEMA_VERSION, classify_validation_scope
from cof_scout.agent.memory import (
    build_decision_state,
    build_notebook,
    decision_state_sha256,
    notebook_sha256,
)


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(event, dict):
                raise ValueError(f"Trace line {line_number} is not an object")
            events.append(event)
    return events


def _candidate(event: dict[str, Any]) -> tuple[str, str, str, str]:
    experiment = event.get("action", {}).get("experiment", {})
    return tuple(str(experiment.get(field, "?")) for field in (
        "node_id", "linker_id", "linkage", "r_group_id"
    ))  # type: ignore[return-value]


def validate_events(events: list[dict[str, Any]],
                    final_notebook: dict[str, Any] | None = None) -> dict[str, int]:
    """Raise on a provenance violation and return concise event counts."""
    if not events:
        raise ValueError("Trace contains no events")
    for position, event in enumerate(events, 1):
        if event.get("event_index") != position:
            raise ValueError(
                f"event_index mismatch at position {position}: {event.get('event_index')}"
            )
        if event.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"trace schema mismatch at event {position}: "
                f"{event.get('trace_schema_version')!r}"
            )

    plan_events = [event for event in events if event.get("event_type") == "campaign_plan"]
    if len(plan_events) != 1:
        raise ValueError(f"Expected exactly one accepted campaign plan, found {len(plan_events)}")

    experiments = [event for event in events if event.get("event_type") == "experiment"]
    expected_iterations = list(range(1, len(experiments) + 1))
    observed_iterations = [event.get("iteration") for event in experiments]
    if observed_iterations != expected_iterations:
        raise ValueError(
            f"Experiment iterations are not sequential: {observed_iterations}"
        )
    candidates = [_candidate(event) for event in experiments]
    if len(candidates) != len(set(candidates)):
        raise ValueError("Trace contains duplicate oracle candidate experiments")

    seen_scaffolds: set[tuple[str, str, str]] = set()
    for index, event in enumerate(events):
        if event.get("event_type") == "tool_call":
            evidence = event.get("tool_evidence", {})
            if evidence.get("evidence_scope") != "public_inventory_and_current_run":
                raise ValueError(
                    f"Tool event {event.get('event_index')} has an invalid evidence scope"
                )
        if event.get("event_type") != "experiment":
            continue
        candidate = _candidate(event)
        scaffold = candidate[:3]
        intent = event.get("action", {}).get("experiment", {}).get("intent")
        expected_scope = classify_validation_scope(intent, scaffold not in seen_scaffolds)
        if "validation_scope" in event and event.get("validation_scope") != expected_scope:
            raise ValueError(
                f"Validation-scope mismatch at experiment {event.get('iteration')}"
            )
        seen_scaffolds.add(scaffold)
        before = build_notebook(events[:index])
        after = build_notebook(events[:index + 1])
        if event.get("notebook_sha256_before") != notebook_sha256(before):
            raise ValueError(
                f"Notebook-before hash mismatch at experiment {event.get('iteration')}"
            )
        if event.get("notebook_sha256_after") != notebook_sha256(after):
            raise ValueError(
                f"Notebook-after hash mismatch at experiment {event.get('iteration')}"
            )
        if event.get("notebook_before") != before:
            raise ValueError(
                f"Stored notebook-before snapshot mismatch at experiment {event.get('iteration')}"
            )
        if event.get("notebook_after") != after:
            raise ValueError(
                f"Stored notebook-after snapshot mismatch at experiment {event.get('iteration')}"
            )
        if "decision_state_before" in event or "decision_state_after" in event:
            stored_before = event.get("decision_state_before", {})
            stored_after = event.get("decision_state_after", {})
            schema_before = stored_before.get("decision_state_schema_version")
            schema_after = stored_after.get("decision_state_schema_version")
            if schema_before != schema_after:
                raise ValueError(
                    f"Decision-state schema mismatch at experiment {event.get('iteration')}"
                )
            state_before = build_decision_state(
                events[:index], schema_version=schema_before
            )
            state_after = build_decision_state(
                events[:index + 1], schema_version=schema_after
            )
            policy = event.get("portfolio_policy")
            if isinstance(policy, dict):
                state_before["portfolio"]["policy"] = policy
                state_after["portfolio"]["policy"] = policy
            if event.get("decision_state_before") != state_before:
                raise ValueError(
                    f"Decision-state-before mismatch at experiment {event.get('iteration')}"
                )
            if event.get("decision_state_after") != state_after:
                raise ValueError(
                    f"Decision-state-after mismatch at experiment {event.get('iteration')}"
                )
            if event.get("decision_state_sha256_before") != decision_state_sha256(
                state_before
            ):
                raise ValueError(
                    f"Decision-state-before hash mismatch at experiment {event.get('iteration')}"
                )
            if event.get("decision_state_sha256_after") != decision_state_sha256(
                state_after
            ):
                raise ValueError(
                    f"Decision-state-after hash mismatch at experiment {event.get('iteration')}"
                )

    rebuilt_final = build_notebook(events)
    if final_notebook is not None and final_notebook != rebuilt_final:
        raise ValueError("final_notebook.json does not match the notebook rebuilt from trace")
    return {
        "events": len(events),
        "experiments": len(experiments),
        "tool_calls": sum(event.get("event_type") == "tool_call" for event in events),
        "rejected_actions": sum(
            event.get("event_type") in {"plan_rejected", "action_rejected"}
            for event in events
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a COF-SCOUT event trace")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--final-notebook", type=Path)
    args = parser.parse_args()
    if not args.trace.is_file():
        parser.error(f"trace not found: {args.trace}")
    final_path = args.final_notebook or args.trace.parent / "final_notebook.json"
    final_notebook = (
        json.loads(final_path.read_text(encoding="utf-8"))
        if final_path.is_file() else None
    )
    summary = validate_events(load_events(args.trace), final_notebook)
    print(
        "COF-SCOUT trace validation PASSED: "
        f"events={summary['events']} experiments={summary['experiments']} "
        f"tool_calls={summary['tool_calls']} rejected_actions={summary['rejected_actions']}"
    )


if __name__ == "__main__":
    main()
