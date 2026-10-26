"""Reconstruct the experimental facts and scientific notebook from COF-SCOUT events."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from cof_scout.agent.contracts import NOTEBOOK_SCHEMA_VERSION


LEGACY_DECISION_STATE_SCHEMA_VERSION = "ara5-decision-state-v1"
DECISION_STATE_SCHEMA_VERSION = "ara5-decision-state-v2"
MAX_DECISION_TEXT = 800
MAX_HIT_SCAFFOLDS = 16
MAX_HYPOTHESES = 12
MAX_REJECTION_CATEGORIES = 12
MAX_MATCHED_PAIR_MEMORIES = 24
MAX_COMPONENT_MEMORIES = 24


def _compact_text(value: Any, limit: int = MAX_DECISION_TEXT) -> Any:
    """Bound model-authored text in decision state without changing the full notebook."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit - 18].rstrip() + " … [truncated]"


def _experiment_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == "experiment"]


def _candidate(event: dict[str, Any]) -> dict[str, str]:
    action = event.get("action", {})
    experiment = action.get("experiment", {}) if isinstance(action, dict) else {}
    return {
        "node_id": str(experiment.get("node_id", "?")),
        "linker_id": str(experiment.get("linker_id", "?")),
        "linkage": str(experiment.get("linkage", "?")),
        "r_group_id": str(experiment.get("r_group_id", "?")),
    }


def _criterion(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result", {})
    value = result.get("criterion", {}) if isinstance(result, dict) else {}
    return value if isinstance(value, dict) else {}


def _formatted_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "?"


def build_evaluated_candidate_memory(
        events: Iterable[dict[str, Any]]) -> list[str]:
    """Return an exhaustive, compact model-facing ledger of oracle experiments."""
    records: list[str] = []
    for event in _experiment_events(events):
        iteration = event.get("iteration", "?")
        candidate = _candidate(event)
        candidate_id = "/".join(candidate[field] for field in (
            "node_id", "linker_id", "linkage", "r_group_id",
        ))
        result = event.get("result", {})
        if not isinstance(result, dict):
            result = {}
        status = str(result.get("status", "unknown"))
        criterion = _criterion(event)
        if status != "success":
            detail = _compact_text(result.get("error") or status, 180)
            records.append(
                f"E{iteration} {candidate_id} EVALUATION_FAILED reason={detail}"
            )
            continue

        if bool(criterion.get("is_hit")):
            try:
                robust = float(criterion.get("vbm_margin", float("-inf"))) > 0.25
            except (TypeError, ValueError):
                robust = False
            outcome = "ROBUST_HIT" if robust else "HIT"
            records.append(
                f"E{iteration} {candidate_id} {outcome} all_gates_passed "
                f"gap={_formatted_number(result.get('gap_corrected_eV'))} "
                f"CBM={_formatted_number(result.get('cbm_vs_nhe'))} "
                f"VBM={_formatted_number(result.get('vbm_vs_nhe'))} "
                f"stability={_formatted_number(result.get('s_linkage'))}"
            )
            continue

        failures = []
        if not criterion.get("gap_pass", False):
            gap = result.get("gap_corrected_eV")
            try:
                gap_value = float(gap)
            except (TypeError, ValueError):
                failures.append("gap_failed")
            else:
                if gap_value < 1.8:
                    failures.append(f"gap={gap_value:.3f}<1.800")
                elif gap_value > 2.2:
                    failures.append(f"gap={gap_value:.3f}>2.200")
                else:
                    failures.append(f"gap={gap_value:.3f}_gate_failed")
        if not criterion.get("cbm_pass", False):
            failures.append(
                f"CBM={_formatted_number(result.get('cbm_vs_nhe'))}>=0.000"
            )
        if not criterion.get("vbm_pass", False):
            failures.append(
                f"VBM={_formatted_number(result.get('vbm_vs_nhe'))}<1.230"
            )
        if not criterion.get("stability_pass", False):
            failures.append(
                f"stability={_formatted_number(result.get('s_linkage'))}<0.700"
            )
        reason = ";".join(failures) or "oracle_non_hit_without_gate_detail"
        records.append(f"E{iteration} {candidate_id} MISS failed={reason}")
    return records


def _result_value(event: dict[str, Any], field: str) -> float | None:
    result = event.get("result", {})
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    value = result.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _is_robust_hit(event: dict[str, Any]) -> bool:
    criterion = _criterion(event)
    if not criterion.get("is_hit"):
        return False
    try:
        return float(criterion.get("vbm_margin", float("-inf"))) > 0.25
    except (TypeError, ValueError):
        return False


def build_scientific_memory(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build bounded relational memory from observations already in the trace.

    This is not a predictive model. It records one-component experimental
    comparisons, the observed scope of linkage conclusions, and reusable
    component outcomes so the scientist can reason from prior trials without
    relying only on recent prose.
    """
    experiments = [
        event for event in _experiment_events(events)
        if isinstance(event.get("result"), dict)
        and event["result"].get("status") == "success"
    ]
    factor_names = ("node", "linker", "linkage", "r_group")
    property_fields = {
        "gap_eV": "gap_corrected_eV",
        "cbm_V": "cbm_vs_nhe",
        "vbm_V": "vbm_vs_nhe",
        "stability": "s_linkage",
    }

    matched_pairs = []
    for left_index, left in enumerate(experiments):
        left_candidate = _candidate(left)
        left_key = tuple(left_candidate[field] for field in (
            "node_id", "linker_id", "linkage", "r_group_id",
        ))
        for right in experiments[left_index + 1:]:
            right_candidate = _candidate(right)
            right_key = tuple(right_candidate[field] for field in (
                "node_id", "linker_id", "linkage", "r_group_id",
            ))
            differing = [index for index in range(4) if left_key[index] != right_key[index]]
            if len(differing) != 1:
                continue
            changed_index = differing[0]
            deltas = {}
            for label, field in property_fields.items():
                left_value = _result_value(left, field)
                right_value = _result_value(right, field)
                deltas[label] = (
                    round(right_value - left_value, 6)
                    if left_value is not None and right_value is not None else None
                )
            left_iteration = int(left.get("iteration", 0) or 0)
            right_iteration = int(right.get("iteration", 0) or 0)
            prediction = right.get("action", {}).get("experiment", {}).get(
                "prediction", {}
            )
            prediction_linked = (
                isinstance(prediction, dict)
                and prediction.get("reference_iteration") == left_iteration
            )
            fixed_scope = {
                name: left_key[index]
                for index, name in enumerate(factor_names)
                if index != changed_index
            }
            matched_pairs.append({
                "iterations": [left_iteration, right_iteration],
                "changed_factor": factor_names[changed_index],
                "change": f"{left_key[changed_index]}->{right_key[changed_index]}",
                "fixed_scope": fixed_scope,
                "property_deltas": deltas,
                "outcome_change": (
                    f"{'HIT' if _criterion(left).get('is_hit') else 'MISS'}->"
                    f"{'HIT' if _criterion(right).get('is_hit') else 'MISS'}"
                ),
                "prediction_linked": prediction_linked,
            })

    matched_pairs.sort(key=lambda row: (
        not row["prediction_linked"],
        "HIT" not in row["outcome_change"],
        -row["iterations"][1],
        -row["iterations"][0],
    ))
    selected_pairs = matched_pairs[:MAX_MATCHED_PAIR_MEMORIES]
    selected_pairs.sort(key=lambda row: tuple(row["iterations"]))

    component_rows: dict[tuple[str, str], dict[str, Any]] = {}
    linkage_scopes: dict[str, dict[str, Any]] = {}
    candidate_fields = ("node_id", "linker_id", "linkage", "r_group_id")
    for event in experiments:
        iteration = int(event.get("iteration", 0) or 0)
        candidate = _candidate(event)
        criterion = _criterion(event)
        is_hit = bool(criterion.get("is_hit"))
        is_robust = _is_robust_hit(event)
        failed_gates = [
            gate for gate in ("gap", "cbm", "vbm", "stability")
            if not criterion.get(f"{gate}_pass", False)
        ]
        for factor, field in zip(factor_names, candidate_fields):
            component_id = candidate[field]
            row = component_rows.setdefault((factor, component_id), {
                "factor": factor, "component_id": component_id,
                "experiments": 0, "hits": 0, "robust_hits": 0,
                "iterations": [], "tested_linkages": set(),
                "hit_linkages": set(), "failed_gates": Counter(),
            })
            row["experiments"] += 1
            row["hits"] += int(is_hit)
            row["robust_hits"] += int(is_robust)
            row["iterations"].append(iteration)
            row["tested_linkages"].add(candidate["linkage"])
            if is_hit:
                row["hit_linkages"].add(candidate["linkage"])
            row["failed_gates"].update(failed_gates)

        linkage = candidate["linkage"]
        scope = linkage_scopes.setdefault(linkage, {
            "linkage": linkage, "experiments": 0, "hits": 0,
            "nodes_tested": set(), "linkers_tested": set(),
            "r_groups_tested": set(), "failed_gates": Counter(),
        })
        scope["experiments"] += 1
        scope["hits"] += int(is_hit)
        scope["nodes_tested"].add(candidate["node_id"])
        scope["linkers_tested"].add(candidate["linker_id"])
        scope["r_groups_tested"].add(candidate["r_group_id"])
        scope["failed_gates"].update(failed_gates)

    compact_components = []
    for row in component_rows.values():
        if row["hits"] == 0 and row["experiments"] < 2:
            continue
        compact_components.append({
            **{key: row[key] for key in (
                "factor", "component_id", "experiments", "hits", "robust_hits",
                "iterations",
            )},
            "tested_linkages": sorted(row["tested_linkages"]),
            "hit_linkages": sorted(row["hit_linkages"]),
            "failed_gates": dict(sorted(row["failed_gates"].items())),
        })
    compact_components.sort(key=lambda row: (
        -row["hits"], -row["robust_hits"], -row["experiments"],
        row["factor"], row["component_id"],
    ))

    compact_scopes = []
    for linkage, row in sorted(linkage_scopes.items()):
        compact_scopes.append({
            "linkage": linkage,
            "experiments": row["experiments"],
            "hits": row["hits"],
            "nodes_tested": sorted(row["nodes_tested"]),
            "linkers_tested": sorted(row["linkers_tested"]),
            "r_groups_tested": sorted(row["r_groups_tested"]),
            "failed_gates": dict(sorted(row["failed_gates"].items())),
        })

    return {
        "evidence_scope": "observations_from_current_run_only",
        "matched_one_component_changes": selected_pairs,
        "matched_pairs_omitted": max(0, len(matched_pairs) - len(selected_pairs)),
        "reusable_component_outcomes": compact_components[:MAX_COMPONENT_MEMORIES],
        "component_outcomes_omitted": max(
            0, len(compact_components) - MAX_COMPONENT_MEMORIES
        ),
        "linkage_evidence_scope": compact_scopes,
    }


def build_deterministic_facts(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build Python-owned campaign facts from accepted oracle experiments."""
    experiments = _experiment_events(events)
    successful = [
        event for event in experiments
        if isinstance(event.get("result"), dict)
        and event["result"].get("status") == "success"
    ]
    hits = [event for event in successful if bool(_criterion(event).get("is_hit"))]
    robust = [
        event for event in hits
        if float(_criterion(event).get("vbm_margin", float("-inf"))) > 0.25
    ]

    scaffold_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in experiments:
        candidate = _candidate(event)
        scaffold_rows[(candidate["node_id"], candidate["linker_id"],
                       candidate["linkage"])].append(event)

    scaffold_table: dict[str, dict[str, Any]] = {}
    for scaffold, members in sorted(scaffold_rows.items()):
        member_successes = [
            event for event in members
            if event.get("result", {}).get("status") == "success"
        ]
        member_hits = [event for event in member_successes if _criterion(event).get("is_hit")]
        margins = []
        for event in member_successes:
            criterion = _criterion(event)
            try:
                margins.append(min(
                    float(criterion["gap_margin_floor"]),
                    float(criterion["gap_margin_ceiling"]),
                    float(criterion["cbm_margin"]),
                    float(criterion["vbm_margin"]),
                    float(criterion["stability_margin"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        scaffold_table["/".join(scaffold)] = {
            "scaffold": list(scaffold),
            "experiments": len(members),
            "successful": len(member_successes),
            "hits": len(member_hits),
            "robust_hits": sum(
                float(_criterion(event).get("vbm_margin", float("-inf"))) > 0.25
                for event in member_hits
            ),
            "r_groups_tested": sorted({_candidate(event)["r_group_id"] for event in members}),
            "best_all_gate_margin": round(max(margins), 6) if margins else None,
            "discovered": bool(member_hits),
        }

    linkage_table: dict[str, dict[str, Any]] = {}
    linkage_names = sorted({_candidate(event)["linkage"] for event in experiments})
    for linkage in linkage_names:
        members = [event for event in experiments if _candidate(event)["linkage"] == linkage]
        member_successes = [
            event for event in members if event.get("result", {}).get("status") == "success"
        ]
        member_hits = [event for event in member_successes if _criterion(event).get("is_hit")]
        linkage_table[linkage] = {
            "experiments": len(members),
            "successful": len(member_successes),
            "hits": len(member_hits),
            "hit_scaffolds": len({
                tuple(_candidate(event)[name] for name in ("node_id", "linker_id", "linkage"))
                for event in member_hits
            }),
        }

    gate_failures = Counter()
    for event in successful:
        criterion = _criterion(event)
        for gate in ("gap", "cbm", "vbm", "stability"):
            if not criterion.get(f"{gate}_pass", False):
                gate_failures[gate] += 1

    recent = []
    for event in experiments[-8:]:
        candidate = _candidate(event)
        result = event.get("result", {})
        criterion = _criterion(event)
        recent.append({
            "iteration": event.get("iteration"),
            "candidate": candidate,
            "status": result.get("status"),
            "gap_corrected_eV": result.get("gap_corrected_eV"),
            "cbm_vs_nhe": result.get("cbm_vs_nhe"),
            "vbm_vs_nhe": result.get("vbm_vs_nhe"),
            "is_hit": criterion.get("is_hit"),
        })

    return {
        "experiments": len(experiments),
        "successful_evaluations": len(successful),
        "evaluation_failures": len(experiments) - len(successful),
        "hits": len(hits),
        "robust_hits": len(robust),
        "scaffolds_tested": len(scaffold_table),
        "hit_scaffolds": sum(row["discovered"] for row in scaffold_table.values()),
        "robust_hit_scaffolds": sum(row["robust_hits"] > 0 for row in scaffold_table.values()),
        "gate_failures": dict(gate_failures),
        "linkage_table": linkage_table,
        "scaffold_table": scaffold_table,
        "recent_observations": recent,
    }


def _initial_plan(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("event_type") == "campaign_plan":
            value = event.get("plan")
            return value if isinstance(value, dict) else None
    return None


def build_notebook(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the complete notebook from the append-only trace."""
    events = list(events)
    plan = _initial_plan(events) or {}
    hypotheses: dict[str, dict[str, Any]] = {}
    for hypothesis in plan.get("hypotheses", []):
        hypothesis_id = str(hypothesis.get("hypothesis_id", "")).strip()
        if not hypothesis_id:
            continue
        hypotheses[hypothesis_id] = {
            "hypothesis_id": hypothesis_id,
            "statement": hypothesis.get("statement"),
            "rationale": hypothesis.get("rationale"),
            "confidence": hypothesis.get("confidence"),
            "status": "active",
            "discriminating_test": hypothesis.get("discriminating_test"),
            "evidence_history": [],
        }

    revisions: list[dict[str, Any]] = []
    next_priority = None
    for event in events:
        if event.get("event_type") == "strategy_replan":
            revisions.append({
                "iteration": event.get("iteration"),
                "source": "explicit_replan",
                "revision": event.get("strategy_revision"),
                "reason": event.get("reason"),
            })
        if event.get("event_type") != "experiment":
            continue
        reflection = event.get("reflection")
        if not isinstance(reflection, dict):
            continue
        iteration = event.get("iteration")
        raw_strategy_update = reflection.get("strategy_update")
        strategy_update = (
            str(raw_strategy_update).strip() if raw_strategy_update is not None else ""
        )
        if strategy_update:
            revisions.append({
                "iteration": iteration,
                "source": "post_experiment_reflection",
                "revision": strategy_update,
            })
        next_priority = reflection.get("next_priority") or next_priority
        for update in reflection.get("hypothesis_updates", []):
            if not isinstance(update, dict):
                continue
            hypothesis_id = str(update.get("hypothesis_id", "")).strip()
            if not hypothesis_id:
                continue
            current = hypotheses.setdefault(hypothesis_id, {
                "hypothesis_id": hypothesis_id,
                "statement": update.get("revised_statement") or "Introduced during reflection",
                "rationale": "Introduced after observing campaign evidence.",
                "confidence": update.get("confidence"),
                "status": "active",
                "discriminating_test": None,
                "evidence_history": [],
            })
            if update.get("revised_statement"):
                current["statement"] = update["revised_statement"]
            current["status"] = update.get("status", current.get("status"))
            current["confidence"] = update.get("confidence", current.get("confidence"))
            current["evidence_history"].append({
                "iteration": iteration,
                "status": update.get("status"),
                "confidence": update.get("confidence"),
                "evidence": update.get("evidence"),
            })

    return {
        "notebook_schema_version": NOTEBOOK_SCHEMA_VERSION,
        "deterministic_facts": build_deterministic_facts(events),
        "scientist_interpretations": {
            "objective": plan.get("objective"),
            "landscape_assessment": plan.get("landscape_assessment"),
            "initial_strategy": plan.get("strategy"),
            "decision_principles": list(plan.get("decision_principles", [])),
            "saturation_criteria": list(plan.get("saturation_criteria", [])),
            "hypotheses": hypotheses,
            "strategy_revisions": revisions,
            "next_priority": next_priority,
        },
    }


def notebook_sha256(notebook: dict[str, Any]) -> str:
    """Hash the canonical notebook representation for provenance checks."""
    payload = json.dumps(notebook, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_decision_state(
        events: Iterable[dict[str, Any]],
        schema_version: str | None = None) -> dict[str, Any]:
    """Build a bounded model-facing state while retaining the full notebook separately."""
    events = list(events)
    schema_version = schema_version or DECISION_STATE_SCHEMA_VERSION
    if schema_version not in {
        LEGACY_DECISION_STATE_SCHEMA_VERSION, DECISION_STATE_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported decision-state schema: {schema_version!r}")
    notebook = build_notebook(events)
    facts = notebook["deterministic_facts"]
    interpretations = notebook["scientist_interpretations"]

    hit_scaffolds = []
    near_hits = []
    repeated_scaffolds = []
    for key, row in facts.get("scaffold_table", {}).items():
        compact = {
            "scaffold": row.get("scaffold"),
            "experiments": row.get("experiments"),
            "hits": row.get("hits"),
            "robust_hits": row.get("robust_hits"),
            "r_groups_tested": row.get("r_groups_tested"),
            "best_all_gate_margin": row.get("best_all_gate_margin"),
        }
        if row.get("discovered"):
            hit_scaffolds.append(compact)
        elif isinstance(row.get("best_all_gate_margin"), (int, float)):
            near_hits.append(compact)
        if int(row.get("experiments", 0)) > 1:
            repeated_scaffolds.append({"scaffold_id": key, **compact})
    hit_scaffolds.sort(key=lambda row: (-int(row.get("robust_hits", 0)), str(row["scaffold"])))
    near_hits.sort(key=lambda row: (-float(row["best_all_gate_margin"]), str(row["scaffold"])))
    repeated_scaffolds.sort(key=lambda row: (-int(row["experiments"]), row["scaffold_id"]))

    hypothesis_rows = []
    for hypothesis_id, hypothesis in interpretations.get("hypotheses", {}).items():
        history = list(hypothesis.get("evidence_history", []))
        status_counts = Counter(str(item.get("status", "unknown")) for item in history)
        recent_evidence = [{
            **item,
            "evidence": _compact_text(item.get("evidence")),
        } for item in history[-2:]]
        hypothesis_rows.append((
            int(history[-1].get("iteration") or 0) if history else 0,
            hypothesis_id,
            {
            "hypothesis_id": hypothesis_id,
            "statement": _compact_text(hypothesis.get("statement")),
            "rationale": _compact_text(hypothesis.get("rationale")),
            "discriminating_test": _compact_text(hypothesis.get("discriminating_test")),
            "status": hypothesis.get("status"),
            "confidence": hypothesis.get("confidence"),
            "evidence_updates": len(history),
            "status_counts": dict(status_counts),
            "recent_evidence": recent_evidence,
            },
        ))
    hypothesis_rows.sort(key=lambda item: (-item[0], item[1]))
    compact_hypotheses = {
        hypothesis_id: value
        for _, hypothesis_id, value in hypothesis_rows[:MAX_HYPOTHESES]
    }

    rejection_counts = Counter()
    invalid_candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") not in {"plan_rejected", "action_rejected"}:
            continue
        rejection = str(event.get("rejection", "unknown rejection"))
        rejection_counts[rejection] += 1
        if rejection != "Candidate is absent from the exact active inventory.":
            continue
        experiment = event.get("action", {}).get("experiment", {})
        candidate = tuple(str(experiment.get(field, "?")) for field in (
            "node_id", "linker_id", "linkage", "r_group_id",
        ))
        current = invalid_candidates.setdefault(candidate, {
            "candidate": list(candidate), "attempts": 0,
            "first_iteration": event.get("iteration"), "last_iteration": None,
            "nearest_feasible": [],
        })
        current["attempts"] += 1
        current["last_iteration"] = event.get("iteration")
        repair = event.get("repair_evidence", {})
        result = repair.get("result", {}) if isinstance(repair, dict) else {}
        if isinstance(result, dict) and result.get("nearest_feasible"):
            current["nearest_feasible"] = result["nearest_feasible"][:5]

    experiments = _experiment_events(events)
    seen: set[tuple[str, str, str]] = set()
    decoration_experiments = 0
    consecutive_decorations = 0
    recent_novelty = []
    for event in experiments:
        candidate = _candidate(event)
        scaffold = (candidate["node_id"], candidate["linker_id"], candidate["linkage"])
        is_new = scaffold not in seen
        if is_new:
            consecutive_decorations = 0
        else:
            decoration_experiments += 1
            consecutive_decorations += 1
        seen.add(scaffold)
        recent_novelty.append({
            "iteration": event.get("iteration"),
            "scaffold": list(scaffold),
            "is_new_scaffold": is_new,
        })

    revisions = list(interpretations.get("strategy_revisions", []))
    current_strategy = _compact_text(
        revisions[-1].get("revision") if revisions
        else interpretations.get("initial_strategy")
    )
    compact_revisions = [{
        **revision,
        "revision": _compact_text(revision.get("revision")),
        "reason": _compact_text(revision.get("reason")),
    } for revision in revisions[-3:]]
    rejection_rows = sorted(
        rejection_counts.items(), key=lambda item: (-item[1], item[0])
    )[:MAX_REJECTION_CATEGORIES]
    state = {
        "decision_state_schema_version": schema_version,
        "deterministic_facts": {
            "experiments": facts.get("experiments"),
            "successful_evaluations": facts.get("successful_evaluations"),
            "evaluation_failures": facts.get("evaluation_failures"),
            "hits": facts.get("hits"),
            "robust_hits": facts.get("robust_hits"),
            "scaffolds_tested": facts.get("scaffolds_tested"),
            "hit_scaffolds_count": facts.get("hit_scaffolds"),
            "robust_hit_scaffolds_count": facts.get("robust_hit_scaffolds"),
            "gate_failures": facts.get("gate_failures"),
            "linkage_table": facts.get("linkage_table"),
            "hit_scaffolds": hit_scaffolds[:MAX_HIT_SCAFFOLDS],
            "hit_scaffolds_omitted": max(0, len(hit_scaffolds) - MAX_HIT_SCAFFOLDS),
            "near_hit_scaffolds": near_hits[:8],
            "recent_observations": list(facts.get("recent_observations", []))[-6:],
        },
        "portfolio": {
            "decoration_experiments": decoration_experiments,
            "decoration_rate": (
                round(decoration_experiments / len(experiments), 6) if experiments else 0.0
            ),
            "consecutive_decorations": consecutive_decorations,
            "recent_novelty": recent_novelty[-5:],
            "repeated_scaffolds": repeated_scaffolds[:8],
        },
        "rejection_memory": {
            "counts": dict(rejection_rows),
            "categories_omitted": max(
                0, len(rejection_counts) - MAX_REJECTION_CATEGORIES
            ),
            "invalid_candidates": sorted(
                invalid_candidates.values(), key=lambda row: (-row["attempts"], row["candidate"])
            )[:12],
        },
        "scientist_interpretations": {
            "objective": _compact_text(interpretations.get("objective")),
            "decision_principles": [
                _compact_text(item)
                for item in list(interpretations.get("decision_principles", []))[:12]
            ],
            "saturation_criteria": [
                _compact_text(item)
                for item in list(interpretations.get("saturation_criteria", []))[:12]
            ],
            "hypotheses": compact_hypotheses,
            "hypotheses_omitted": max(0, len(hypothesis_rows) - MAX_HYPOTHESES),
            "current_strategy": current_strategy,
            "recent_strategy_revisions": compact_revisions,
            "next_priority": _compact_text(interpretations.get("next_priority")),
        },
    }
    if schema_version == DECISION_STATE_SCHEMA_VERSION:
        state["scientific_memory"] = build_scientific_memory(events)
    return state


def decision_state_sha256(state: dict[str, Any]) -> str:
    """Hash the canonical compact decision state."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
