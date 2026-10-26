#!/usr/bin/env python3
"""COF-SCOUT Scientist: plan -> investigate -> experiment -> reflect -> replan.

The loop exposes safe deterministic analysis tools to one scientific agent.
Only accepted oracle experiments consume the experimental budget. Every
observable scientific rationale, action, evidence packet, prediction, decision
rule, result, and reflection is retained in an append-only event trace.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from cof_scout.agent.contracts import (
    ALLOWED_DIRECTIONS,
    ActionKind,
    PREDICTION_GATES,
    TRACE_SCHEMA_VERSION,
    classify_validation_scope,
)
from cof_scout.agent.memory import (
    DECISION_STATE_SCHEMA_VERSION,
    build_decision_state,
    build_evaluated_candidate_memory,
    build_notebook,
    decision_state_sha256,
    notebook_sha256,
)
from cof_scout.agent.protocol import (
    extract_scientific_plan,
    extract_scientist_action,
    extract_scientist_reflection,
)
from cof_scout.agent.tools import ScientistToolbox
from cof_scout.inventory.assembly import load_library
from cof_scout.oracle.calibration import load_cbm_calibration, load_transfer_function
from cof_scout.agent.client import make_client
from cof_scout.oracle.evaluation import CBM_MAX, GAP_MAX, GAP_MIN, STAB_MIN, VBM_MIN, evaluate_candidate
from cof_scout.agent.prompt_builder import _load_feasible_set, build_prompt, run_leakage_audit
from cof_scout.oracle.reference import ReferenceOracle
from cof_scout.oracle.stability import STABILITY_MODEL_VERSION
from cof_scout.paths import data_path, repository_root


ROOT = repository_root()
OUTPUT_ROOT = ROOT / "output" / "runs"
MAX_PLAN_ATTEMPTS = 3
MAX_REFLECTION_ATTEMPTS = 2


SCIENTIST_SYSTEM = """You are the lead computational chemist running an auditable COF
discovery campaign for experimental collaborators. Your primary objective is to discover
distinct hit scaffolds efficiently, while producing a scientifically useful record of why
each decision was made.

Act like a scientist, not a candidate generator. Form competing hypotheses, identify the
most informative evidence, use deterministic analysis tools when useful, pre-register a
falsifiable prediction before each oracle experiment, and revise strategy when evidence
contradicts expectations. Distinguish observed facts from interpretation. Never invent an
oracle value or claim to have used a tool that was not actually called.

The scientific_rationale field in each structured response is an emitted justification
intended for experimental review. It should explain the chemical considerations and decision
logic concisely. It is not a request to reveal private hidden chain-of-thought. The exact
supplied context, emitted rationale, structured decision, tool evidence, and later reflection
are all retained in the audit trace.

Only an experiment action calls the expensive oracle. Tool actions analyse public inventory
or observations already collected in this campaign and do not consume the oracle budget.
Do not farm R-group decorations after a scaffold is discovered unless a clearly stated
secondary scientific objective justifies it.
"""


def build_scientist_system_prompt() -> str:
    """Return chemistry/inventory context without Ara-3's proposal protocol."""
    base_prompt, _ = build_prompt()
    marker = "\nYOUR TASK EACH ITERATION:"
    if marker not in base_prompt:
        raise RuntimeError("Could not locate the Ara-3 procedural section in the base prompt")
    chemistry_reference = base_prompt.split(marker, 1)[0]
    run_leakage_audit(chemistry_reference)
    return chemistry_reference + "\n\n" + SCIENTIST_SYSTEM


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _usage_total(usages: list[dict[str, Any]]) -> dict[str, Any]:
    if not usages:
        return {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0, "estimated": True}
    return {
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usages),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in usages),
        "usd": sum(float(item.get("usd", 0.0)) for item in usages),
        "estimated": any(bool(item.get("estimated", True)) for item in usages),
    }


def _target_constraint_geometry() -> dict[str, Any]:
    """Return exact public target relationships for model-visible contexts."""
    lower_at_gap_min = round(VBM_MIN - GAP_MIN, 6)
    lower_at_gap_max = round(VBM_MIN - GAP_MAX, 6)
    return {
        "gap_window_eV": [GAP_MIN, GAP_MAX],
        "cbm_gate_vs_nhe_V": f"CBM < {CBM_MAX:.2f}",
        "vbm_gate_vs_nhe_V": f"VBM >= {VBM_MIN:.2f}",
        "stability_gate": f"S_MCAP >= {STAB_MIN:.2f}",
        "stability_model": (
            "candidate-specific mechanism-constrained aqueous persistence score; "
            "mechanistic linkage envelopes with local shielding, hydrophobicity, "
            "electronic deactivation, and rigidity"
        ),
        "calibration_identity": "VBM = CBM + corrected_gap",
        "coupled_feasibility": {
            "required_cbm_lower_bound_for_given_gap": (
                f"CBM >= {VBM_MIN:.2f} - corrected_gap"
            ),
            "feasible_cbm_interval_for_given_gap": (
                f"[{VBM_MIN:.2f} - corrected_gap, {CBM_MAX:.2f})"
            ),
            f"at_gap_{GAP_MIN:.1f}_eV": [lower_at_gap_min, CBM_MAX],
            f"at_gap_{GAP_MAX:.1f}_eV": [lower_at_gap_max, CBM_MAX],
            "interval_note": "The upper endpoint is open because the CBM gate is strict.",
            "design_warning": (
                "A more negative CBM is not unconditionally better: at fixed gap it lowers "
                "the derived VBM and can fail the oxidation gate."
            ),
        },
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid trace JSON at {path}:{line_number}") from error
            if not isinstance(event, dict):
                raise ValueError(f"Trace event {line_number} is not an object")
            events.append(event)
    return events


def _append_event(path: Path, events: list[dict[str, Any]], event: dict[str, Any]) -> None:
    event = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "event_index": len(events) + 1,
        "timestamp": _now(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
    events.append(event)


def _planning_context(inventory: dict[str, Any], tools: dict[str, str],
                      max_experiments: int) -> str:
    schema = {
        "objective": "scientific objective for this campaign",
        "landscape_assessment": "initial assessment of constraints and opportunities",
        "strategy": "how exploration, refinement, falsification, and tools will be used",
        "hypotheses": [{
            "hypothesis_id": "H1",
            "statement": "testable scientific statement",
            "rationale": "chemical basis",
            "confidence": "low|medium|high",
            "discriminating_test": "observation that separates this hypothesis from alternatives",
        }],
        "decision_principles": ["rules for choosing informative experiments"],
        "saturation_criteria": ["evidence required before claiming a region exhausted"],
    }
    return "\n".join([
        "Design the initial scientific campaign plan before requesting any experiment.",
        f"Oracle experiment budget: {max_experiments}.",
        "PUBLIC TARGET CONSTRAINT GEOMETRY:",
        json.dumps(_target_constraint_geometry(), indent=2, sort_keys=True),
        "ACTIVE INVENTORY SUMMARY:", json.dumps(inventory, indent=2, sort_keys=True),
        "AVAILABLE DETERMINISTIC TOOLS:", json.dumps(tools, indent=2, sort_keys=True),
        "Return exactly one JSON object and no surrounding prose. Use this envelope:",
        json.dumps({
            "scientific_rationale": "concise experimentalist-facing justification",
            "plan": schema,
        }, indent=2),
    ])


def _action_context(notebook: dict[str, Any], tools: dict[str, str],
                    max_experiments: int, completed: int, tool_calls: int,
                    max_tool_calls: int, recent_tool_evidence: list[dict[str, Any]],
                    feedback: list[str],
                    evaluated_candidate_memory: list[str] | None = None,
                    fixed_horizon: bool = False) -> str:
    candidate_memory = list(evaluated_candidate_memory or [])
    return "\n".join([
        "Choose the next scientific action. You may investigate with a deterministic tool,",
        (
            "revise strategy or register one oracle experiment."
            if fixed_horizon else
            "revise strategy, register one oracle experiment, or stop if defensible saturation"
        ),
        (
            "The campaign has a fixed experimental horizon; stop actions are unavailable."
            if fixed_horizon else "criteria are met."
        ),
        f"Oracle budget: {completed}/{max_experiments} experiments used.",
        f"Tool allowance before the next experiment: {tool_calls}/{max_tool_calls} used.",
        "PUBLIC TARGET CONSTRAINT GEOMETRY:",
        json.dumps(_target_constraint_geometry(), indent=2, sort_keys=True),
        "CURRENT COMPACT DECISION STATE:", json.dumps(notebook, indent=2, sort_keys=True),
        "SCIENTIFIC LEARNING MEMORY NOTE:",
        (
            "The scientific_memory section is deterministic and contains only observations "
            "from this campaign. Use its matched one-component changes, reusable component "
            "outcomes, and explicit linkage evidence scope. Treat every pattern as limited "
            "to the listed experiments and tested components; seek transfer or counterexample "
            "experiments before generalizing or closing a chemical family."
        ),
        "PERMANENT EVALUATED-CANDIDATE MEMORY:",
        (
            "This list is exhaustive for the current campaign. Every exact candidate "
            "listed below is permanently unavailable for another oracle experiment. "
            "Use its HIT/MISS outcome and deterministic failure reason as scientific "
            "evidence, then select a different exact candidate."
        ),
        json.dumps(candidate_memory, ensure_ascii=False, separators=(",", ":")),
        "AVAILABLE TOOLS:", json.dumps(tools, indent=2, sort_keys=True),
        "RECENT TOOL EVIDENCE SUMMARIES:", json.dumps(
            [_compact_tool_evidence(item) for item in recent_tool_evidence[-3:]],
            indent=2, sort_keys=True,
        ),
        "HARNESS FEEDBACK:", json.dumps(feedback[-6:], indent=2),
        "",
        "Return exactly one JSON object and no surrounding prose. The top-level envelope is:",
        json.dumps({
            "scientific_rationale": "concise experimentalist-facing justification",
            "action": "one action object matching exactly one schema below",
        }, indent=2),
        "Tool action object:", json.dumps({
            "action": "tool", "purpose": "why this evidence is needed",
            "expected_information": "how the result will change the decision",
            "tool": {"name": "allowlisted tool name", "arguments": {}},
        }, indent=2),
        "Experiment action object:", json.dumps({
            "action": "experiment", "purpose": "why this is the next experiment",
            "experiment": {
                "candidate": {"node_id": "N...", "linker_id": "L...",
                              "linkage": "...", "r_group_id": "R..."},
                "intent": (
                    "explore|refine|falsify|validate; the harness distinguishes "
                    "new-scaffold hypothesis validation from repeated-scaffold confirmation"
                ),
                "purpose": "scientific purpose",
                "hypothesis_id": "H1 or null",
                "expected_information": "what this experiment resolves",
                "prediction": {"axis": "gap|cbm|vbm|stability|overall",
                               "direction": "|".join(sorted(ALLOWED_DIRECTIONS)),
                               "statement": "falsifiable prediction",
                               "reference_iteration": None,
                               "hit_probability": 0.5,
                               "gate_predictions": {
                                   "gap": True, "cbm": True,
                                   "vbm": True, "stability": True,
                               },
                               "limiting_gate": "gap|cbm|vbm|stability|none",
                               "falsification_rule": (
                                   "specific oracle outcome that contradicts the prediction"
                               )},
                "decision_rule": {"if_supported": "next action", "if_falsified": "next action",
                                  "if_inconclusive": "next action"},
            },
        }, indent=2),
        "Replan action object:", json.dumps({
            "action": "replan", "purpose": "why strategy must change",
            "strategy_revision": "new strategy", "reason": "evidence requiring the change",
        }, indent=2),
        "Stop action object:", json.dumps({
            "action": "stop", "purpose": "why stopping is scientifically justified",
            "saturation_criterion": "copy one pre-registered saturation criterion exactly",
            "reason": "evidence satisfying pre-registered saturation criteria",
        }, indent=2),
    ])


def _score_preregistered_prediction(action: dict[str, Any],
                                    result: dict[str, Any]) -> dict[str, Any]:
    """Score prospective fields using deterministic oracle facts only."""
    experiment = action.get("experiment", {}) if isinstance(action, dict) else {}
    prediction = experiment.get("prediction", {}) if isinstance(experiment, dict) else {}
    criterion = result.get("criterion", {}) if isinstance(result, dict) else {}
    if result.get("status") != "success" or not isinstance(criterion, dict):
        return {"scoreable": False, "reason": "oracle_evaluation_not_successful"}

    observed_gates = {
        gate: bool(criterion.get(f"{gate}_pass")) for gate in PREDICTION_GATES
    }
    observed_hit = bool(criterion.get("is_hit"))
    failed_gates = [gate for gate in PREDICTION_GATES if not observed_gates[gate]]
    adjudication: dict[str, Any] = {
        "scoreable": True,
        "observed_hit": observed_hit,
        "observed_gates": observed_gates,
        "observed_failed_gates": failed_gates,
    }

    probability = prediction.get("hit_probability")
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        probability = float(probability)
        adjudication["hit_probability"] = probability
        adjudication["brier_score"] = round(
            (probability - float(observed_hit)) ** 2, 8
        )

    predicted_gates = prediction.get("gate_predictions")
    if isinstance(predicted_gates, dict) and set(predicted_gates) == set(PREDICTION_GATES):
        gate_correct = {
            gate: predicted_gates.get(gate) is observed_gates[gate]
            for gate in PREDICTION_GATES
        }
        adjudication["predicted_gates"] = {
            gate: bool(predicted_gates[gate]) for gate in PREDICTION_GATES
        }
        adjudication["gate_correct"] = gate_correct
        adjudication["gate_accuracy"] = round(
            sum(gate_correct.values()) / len(PREDICTION_GATES), 6
        )
        predicted_hit = all(bool(predicted_gates[gate]) for gate in PREDICTION_GATES)
        adjudication["predicted_hit_from_gates"] = predicted_hit
        adjudication["hit_classification_correct"] = predicted_hit is observed_hit

    predicted_limiting = prediction.get("limiting_gate")
    if predicted_limiting in {*PREDICTION_GATES, "none"}:
        limiting_correct = (
            predicted_limiting == "none" and not failed_gates
        ) or predicted_limiting in failed_gates
        adjudication["predicted_limiting_gate"] = predicted_limiting
        adjudication["limiting_gate_correct"] = limiting_correct

    adjudication["falsification_rule"] = prediction.get("falsification_rule")
    adjudication["complete_publication_prediction"] = all(
        key in adjudication for key in (
            "brier_score", "gate_accuracy", "limiting_gate_correct"
        )
    ) and bool(prediction.get("falsification_rule"))
    return adjudication


def _observation(iteration: int, action: dict[str, Any], result: dict[str, Any]) -> str:
    experiment = action.get("experiment", {})
    candidate = "/".join(str(experiment.get(field, "?")) for field in (
        "node_id", "linker_id", "linkage", "r_group_id"
    ))
    if result.get("status") != "success":
        return f"EXPERIMENT {iteration}: {candidate}; status={result.get('status')}; error={result.get('error', 'not supplied')}"
    criterion = result.get("criterion", {})
    return "\n".join([
        f"EXPERIMENT {iteration} OBSERVATION",
        f"Candidate: {candidate}",
        f"gap_corrected={result.get('gap_corrected_eV'):.3f} eV; "
        f"CBM={result.get('cbm_vs_nhe'):.3f} V; VBM={result.get('vbm_vs_nhe'):.3f} V; "
        f"stability_MCAP={result.get('stability_score', result.get('s_linkage')):.3f}; "
        f"model={result.get('stability', {}).get('model', 'legacy_categorical')}",
        f"gates: gap={criterion.get('gap_pass')} CBM={criterion.get('cbm_pass')} "
        f"VBM={criterion.get('vbm_pass')} stability={criterion.get('stability_pass')} "
        f"hit={criterion.get('is_hit')}",
        f"margins: gap_floor={criterion.get('gap_margin_floor')} "
        f"gap_ceiling={criterion.get('gap_margin_ceiling')} "
        f"CBM={criterion.get('cbm_margin')} VBM={criterion.get('vbm_margin')} "
        f"stability={criterion.get('stability_margin')}",
        "stability_components=" + json.dumps(
            result.get("stability", {}).get("components", {}), sort_keys=True
        ),
    ])


def _reflection_context(action_rationale: str, action: dict[str, Any], observation: str,
                        notebook_before: dict[str, Any]) -> str:
    schema = {
        "observation_summary": "what the experiment established",
        "prediction_supported": True,
        "causal_interpretation": "chemical interpretation, with uncertainty and alternatives",
        "hypothesis_updates": [{
            "hypothesis_id": "H1", "status": "active|supported|weakened|rejected|revised",
            "confidence": "low|medium|high", "evidence": "specific observed evidence",
            "revised_statement": "required full statement only for revised status; otherwise null",
        }],
        "strategy_update": "material policy change, or null when policy should not change",
        "next_priority": "highest-value unresolved scientific question",
    }
    return "\n".join([
        "Interpret the completed experiment for the experimental record. Separate direct",
        "observation from causal interpretation, adjudicate the pre-registered prediction,",
        "update hypotheses, and state the resulting policy change.",
        "PRE-EXPERIMENT SCIENTIFIC RATIONALE:", action_rationale or "(none emitted)",
        "PRE-REGISTERED ACTION:", json.dumps(action, indent=2, sort_keys=True),
        "ORACLE OBSERVATION:", observation,
        "COMPACT DECISION STATE BEFORE EXPERIMENT:",
        json.dumps(notebook_before, indent=2, sort_keys=True),
        "Return exactly one JSON object and no surrounding prose. Use this envelope:",
        json.dumps({
            "scientific_rationale": "concise observation and interpretation for reviewers",
            "reflection": schema,
        }, indent=2),
        "Use null for prediction_supported when the calculation failed or cannot resolve it.",
        "Use an empty hypothesis_updates list when no hypothesis materially changes.",
        "Do not repeat an unchanged hypothesis or strategy. Use null for strategy_update",
        "when the result does not materially change experimental policy.",
        "When status is revised, revised_statement must contain the new full testable statement;",
        "otherwise revised_statement should be null.",
    ])


def _compact_tool_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep decision-relevant tool evidence bounded in model-facing contexts."""
    if not isinstance(evidence, dict):
        return {"invalid_evidence": True}
    result = evidence.get("result", {})
    summary: dict[str, Any] = {
        "tool": evidence.get("tool"),
        "normalized_arguments": evidence.get("normalized_arguments", {}),
        "evidence_scope": evidence.get("evidence_scope"),
    }
    if not isinstance(result, dict):
        summary["result"] = result
        return summary
    if isinstance(result.get("returned"), list):
        summary["result"] = {
            key: value for key, value in result.items() if key != "returned"
        }
        summary["result"]["returned_count"] = len(result["returned"])
        summary["result"]["examples"] = result["returned"][:5]
    elif isinstance(result.get("neighbors"), list):
        factor_counts: dict[str, int] = {}
        for row in result["neighbors"]:
            factor = str(row.get("changed_factor", "unknown"))
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        summary["result"] = {
            "query": result.get("query"),
            "in_active_inventory": result.get("in_active_inventory"),
            "neighbor_count": len(result["neighbors"]),
            "changed_factor_counts": factor_counts,
            "examples": result["neighbors"][:6],
        }
    elif isinstance(result.get("nearest_feasible"), list):
        summary["result"] = {
            key: value for key, value in result.items() if key != "nearest_feasible"
        }
        summary["result"]["nearest_feasible"] = result["nearest_feasible"][:6]
    else:
        summary["result"] = result
    return summary


def _persistent_rejection_feedback(events: list[dict[str, Any]]) -> list[str]:
    """Carry candidate-specific hard rejections across experiment boundaries."""
    duplicates: Counter[tuple[str, str, str, str]] = Counter()
    for event in events:
        if (event.get("event_type") != "action_rejected"
                or event.get("rejection") !=
                "Candidate has already consumed an oracle experiment."):
            continue
        candidate = _event_candidate_from_action(event.get("action", {}))
        if "?" not in candidate:
            duplicates[candidate] += 1
    return [
        f"Permanently blocked duplicate candidate {'/'.join(candidate)} was already "
        f"evaluated and has been rejected {attempts} time(s); do not propose it again."
        for candidate, attempts in duplicates.most_common(6)
    ]


class LiveScientist:
    """Stateless calls to one model using the reconstructed scientific notebook."""

    def __init__(self, client, system_prompt: str, temperature: float):
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.last_cost: dict[str, Any] = {}

    def _call(self, context: str) -> str:
        text, self.last_cost = self.client.chat_structured(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": context}],
            temperature=self.temperature,
            json_object=True,
        )
        return text

    def plan(self, context: str) -> str:
        return self._call(context)

    def act(self, context: str) -> str:
        return self._call(context)

    def reflect(self, context: str) -> str:
        return self._call(context)


class ScriptedScientist:
    """Deterministic scientist used for protocol and end-to-end tests."""

    def __init__(self, plan_response: str, action_responses: list[str],
                 reflection_responses: list[str]):
        self.plan_response = plan_response
        self.action_responses = list(action_responses)
        self.reflection_responses = list(reflection_responses)
        self.last_cost = {
            "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0, "estimated": True,
        }

    def plan(self, _: str) -> str:
        return self.plan_response

    def act(self, _: str) -> str:
        if not self.action_responses:
            raise StopIteration("No scripted action remains")
        return self.action_responses.pop(0)

    def reflect(self, _: str) -> str:
        if not self.reflection_responses:
            raise StopIteration("No scripted reflection remains")
        return self.reflection_responses.pop(0)


Evaluator = Callable[[str, str, str, str, Path], dict[str, Any]]


def run_campaign(scientist: LiveScientist | ScriptedScientist,
                 toolbox: ScientistToolbox, evaluator: Evaluator, run_dir: Path,
                 max_experiments: int, max_tool_calls: int = 6,
                 max_decisions_per_experiment: int = 12,
                 max_decoration_fraction: float = 0.10,
                 max_candidates_per_scaffold: int = 2,
                 max_consecutive_decorations: int = 2,
                 resume: bool = False,
                 fixed_horizon: bool = False,
                 require_scored_predictions: bool = False) -> list[dict[str, Any]]:
    """Run or resume one notebook-driven scientific campaign."""
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists() and not resume:
        raise FileExistsError(
            f"Trace already exists: {trace_path}. Use a new run ID or --resume."
        )
    events = _read_events(trace_path) if resume else []
    tools = toolbox.catalog()
    inventory = toolbox.run("inventory_summary", {}, events)["result"]

    if not any(event.get("event_type") == "campaign_plan" for event in events):
        feedback: list[str] = []
        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            context = _planning_context(inventory, tools, max_experiments)
            if feedback:
                context += "\nPREVIOUS FORMAT FEEDBACK:\n" + "\n".join(feedback)
            try:
                raw = scientist.plan(context)
            except Exception as error:
                feedback.append(f"Model call failed: {error}")
                _append_event(trace_path, events, {
                    "event_type": "model_call_failed", "stage": "campaign_plan",
                    "attempt": attempt, "context": context, "error": str(error),
                })
                continue
            try:
                rationale, plan = extract_scientific_plan(raw)
                _append_event(trace_path, events, {
                    "event_type": "campaign_plan", "attempt": attempt,
                    "context": context, "raw_response": raw,
                    "scientific_rationale": rationale, "plan": plan.to_dict(),
                    "cost": dict(scientist.last_cost),
                })
                break
            except ValueError as error:
                feedback.append(str(error))
                _append_event(trace_path, events, {
                    "event_type": "plan_rejected", "attempt": attempt,
                    "context": context, "raw_response": raw,
                    "rejection": str(error), "cost": dict(scientist.last_cost),
                })
        else:
            raise RuntimeError("Scientist failed to produce a valid initial plan")

    completed = sum(event.get("event_type") == "experiment" for event in events)
    seen = {
        _event_candidate_from_action(event.get("action", {}))
        for event in events if event.get("event_type") == "experiment"
    }
    experiment_events = [
        event for event in events if event.get("event_type") == "experiment"
    ]
    scaffold_counts = Counter(
        _event_candidate_from_action(event.get("action", {}))[:3]
        for event in experiment_events
    )
    observed_scaffolds: set[tuple[str, str, str]] = set()
    decoration_experiments = 0
    consecutive_decorations = 0
    for event in experiment_events:
        scaffold = _event_candidate_from_action(event.get("action", {}))[:3]
        if scaffold in observed_scaffolds:
            decoration_experiments += 1
            consecutive_decorations += 1
        else:
            observed_scaffolds.add(scaffold)
            consecutive_decorations = 0
    decoration_budget = int(max_experiments * max_decoration_fraction)
    recent_tool_evidence = [
        event.get("tool_evidence") for event in events
        if event.get("event_type") == "tool_call" and isinstance(event.get("tool_evidence"), dict)
    ][-6:]

    while completed < max_experiments:
        tool_calls = 0
        feedback = _persistent_rejection_feedback(events)
        experiment_completed = False
        for decision_number in range(1, max_decisions_per_experiment + 1):
            notebook_before = build_notebook(events)
            decision_state_before = build_decision_state(events)
            portfolio_policy = {
                "decoration_budget": decoration_budget,
                "max_decoration_fraction": max_decoration_fraction,
                "max_candidates_per_scaffold": max_candidates_per_scaffold,
                "max_consecutive_decorations": max_consecutive_decorations,
            }
            decision_state_before["portfolio"]["policy"] = portfolio_policy
            context = _action_context(
                decision_state_before, tools, max_experiments, completed, tool_calls,
                max_tool_calls, recent_tool_evidence, feedback,
                build_evaluated_candidate_memory(events),
                fixed_horizon=fixed_horizon,
            )
            try:
                raw = scientist.act(context)
            except StopIteration:
                failure_event = {
                    "event_type": "campaign_failed" if fixed_horizon else "campaign_stop",
                    "iteration": completed + 1,
                    "reason": "Scientist action stream ended.",
                }
                _append_event(trace_path, events, failure_event)
                if fixed_horizon:
                    raise RuntimeError(failure_event["reason"])
                return events
            except Exception as error:
                reason = f"Scientist action call failed: {error}"
                feedback.append(reason)
                _append_event(trace_path, events, {
                    "event_type": "model_call_failed", "stage": "action",
                    "iteration": completed + 1, "decision_number": decision_number,
                    "context": context, "error": str(error),
                })
                continue
            action_cost = dict(scientist.last_cost)
            try:
                rationale, action = extract_scientist_action(raw)
            except ValueError as error:
                feedback.append(str(error))
                _append_event(trace_path, events, {
                    "event_type": "action_rejected", "iteration": completed + 1,
                    "decision_number": decision_number, "context": context,
                    "raw_response": raw, "rejection": str(error), "cost": action_cost,
                })
                continue

            if action.kind is ActionKind.TOOL:
                if tool_calls >= max_tool_calls:
                    reason = "Tool-call allowance exhausted; register an experiment, replan, or stop."
                    feedback.append(reason)
                    _append_event(trace_path, events, {
                        "event_type": "action_rejected", "iteration": completed + 1,
                        "decision_number": decision_number, "context": context,
                        "raw_response": raw, "scientific_rationale": rationale,
                        "action": action.to_dict(), "rejection": reason, "cost": action_cost,
                    })
                    continue
                try:
                    evidence = toolbox.run(
                        action.tool_name or "", action.tool_arguments or {}, events
                    )
                except (ValueError, TypeError) as error:
                    reason = f"Tool request failed: {error}"
                    feedback.append(reason)
                    _append_event(trace_path, events, {
                        "event_type": "action_rejected", "iteration": completed + 1,
                        "decision_number": decision_number, "context": context,
                        "raw_response": raw, "scientific_rationale": rationale,
                        "action": action.to_dict(), "rejection": reason, "cost": action_cost,
                    })
                    continue
                tool_calls += 1
                recent_tool_evidence.append(evidence)
                recent_tool_evidence = recent_tool_evidence[-6:]
                _append_event(trace_path, events, {
                    "event_type": "tool_call", "iteration": completed + 1,
                    "decision_number": decision_number, "context": context,
                    "raw_response": raw, "scientific_rationale": rationale,
                    "action": action.to_dict(), "tool_evidence": evidence,
                    "notebook_sha256": notebook_sha256(notebook_before),
                    "cost": action_cost,
                })
                continue

            if action.kind is ActionKind.REPLAN:
                _append_event(trace_path, events, {
                    "event_type": "strategy_replan", "iteration": completed + 1,
                    "decision_number": decision_number, "context": context,
                    "raw_response": raw, "scientific_rationale": rationale,
                    "action": action.to_dict(),
                    "strategy_revision": action.strategy_revision, "reason": action.reason,
                    "notebook_sha256": notebook_sha256(notebook_before),
                    "cost": action_cost,
                })
                continue

            if action.kind is ActionKind.STOP:
                if fixed_horizon:
                    rejection = (
                        "Fixed-horizon publication mode does not permit early scientific "
                        "stopping; continue until the registered oracle budget is complete."
                    )
                    feedback.append(rejection)
                    _append_event(trace_path, events, {
                        "event_type": "action_rejected", "iteration": completed + 1,
                        "decision_number": decision_number, "context": context,
                        "raw_response": raw, "scientific_rationale": rationale,
                        "action": action.to_dict(), "rejection": rejection,
                        "cost": action_cost,
                    })
                    continue
                criteria = notebook_before["scientist_interpretations"]["saturation_criteria"]
                if action.saturation_criterion not in criteria:
                    rejection = (
                        "Stop action must copy one pre-registered saturation criterion exactly."
                    )
                    feedback.append(rejection)
                    _append_event(trace_path, events, {
                        "event_type": "action_rejected", "iteration": completed + 1,
                        "decision_number": decision_number, "context": context,
                        "raw_response": raw, "scientific_rationale": rationale,
                        "action": action.to_dict(), "rejection": rejection, "cost": action_cost,
                    })
                    continue
                _append_event(trace_path, events, {
                    "event_type": "campaign_stop", "iteration": completed + 1,
                    "decision_number": decision_number, "context": context,
                    "raw_response": raw, "scientific_rationale": rationale,
                    "action": action.to_dict(), "reason": action.reason,
                    "saturation_criterion": action.saturation_criterion,
                    "notebook_sha256": notebook_sha256(notebook_before),
                    "cost": action_cost,
                })
                return events

            experiment = action.experiment
            assert experiment is not None
            candidate_key = experiment.candidate_key
            rejection = None
            repair_evidence = None
            if require_scored_predictions and not experiment.prediction.is_machine_scoreable:
                rejection = (
                    "Publication mode requires hit_probability, four gate_predictions, "
                    "limiting_gate, and falsification_rule in every prediction."
                )
            if rejection is None and candidate_key not in toolbox.feasible_set:
                rejection = "Candidate is absent from the exact active inventory."
                repair_evidence = toolbox.run(
                    "resolve_candidate",
                    {"candidate": {
                        "node_id": candidate_key[0], "linker_id": candidate_key[1],
                        "linkage": candidate_key[2], "r_group_id": candidate_key[3],
                    }, "limit": 8},
                    events,
                )
            elif rejection is None and candidate_key in seen:
                rejection = "Candidate has already consumed an oracle experiment."
                repair_evidence = toolbox.run(
                    "scaffold_frontier",
                    {
                        "linkage": candidate_key[2],
                        "preferred_r_group": candidate_key[3],
                        "limit": 8,
                    },
                    events,
                )
            elif rejection is None and experiment.hypothesis_id is not None:
                hypotheses = notebook_before["scientist_interpretations"]["hypotheses"]
                if experiment.hypothesis_id not in hypotheses:
                    rejection = (
                        f"Unknown hypothesis_id {experiment.hypothesis_id!r}; use an existing "
                        "hypothesis or null."
                    )
            reference = experiment.prediction.reference_iteration
            if rejection is None and reference is not None and reference > completed:
                rejection = (
                    f"Prediction references unobserved experiment {reference}; only experiments "
                    f"1 through {completed} are available."
                )
            scaffold_key = experiment.scaffold_key
            is_decoration = scaffold_counts.get(scaffold_key, 0) > 0
            validation_scope = classify_validation_scope(
                experiment.intent, not is_decoration
            )
            if rejection is None and is_decoration:
                if experiment.intent not in {"refine", "validate"}:
                    rejection = (
                        "A previously tested scaffold requires refine or validate intent."
                    )
                elif decoration_experiments >= decoration_budget:
                    rejection = (
                        "Decoration budget exhausted; select a previously untested scaffold."
                    )
                elif scaffold_counts[scaffold_key] >= max_candidates_per_scaffold:
                    rejection = (
                        "Per-scaffold candidate limit reached; select a distinct scaffold."
                    )
                elif consecutive_decorations >= max_consecutive_decorations:
                    rejection = (
                        "Consecutive-decoration limit reached; select a distinct scaffold."
                    )
            if rejection:
                feedback_message = rejection
                if repair_evidence is not None:
                    result = repair_evidence.get("result", {})
                    if repair_evidence.get("tool") == "resolve_candidate":
                        feedback_message += (
                            " Component compatibility and nearest feasible alternatives: "
                            + json.dumps({
                                "component_compatibility": result.get(
                                    "component_compatibility", {}
                                ),
                                "nearest_feasible": result.get(
                                    "nearest_feasible", []
                                )[:5],
                            }, sort_keys=True)
                        )
                    elif repair_evidence.get("tool") == "scaffold_frontier":
                        feedback_message += (
                            f" The exact candidate {'/'.join(candidate_key)} is permanently "
                            "blocked. Select a different candidate. Unevaluated scaffold "
                            "representatives: "
                            + json.dumps(result.get("returned", [])[:8], sort_keys=True)
                        )
                feedback.append(feedback_message)
                _append_event(trace_path, events, {
                    "event_type": "action_rejected", "iteration": completed + 1,
                    "decision_number": decision_number, "context": context,
                    "raw_response": raw, "scientific_rationale": rationale,
                    "action": action.to_dict(), "rejection": rejection,
                    "repair_evidence": repair_evidence, "cost": action_cost,
                })
                continue

            iteration = completed + 1
            try:
                result = evaluator(*candidate_key, run_dir / "xtb")
            except Exception as error:  # preserve the campaign even if an evaluator raises
                result = {
                    "node": candidate_key[0], "linker": candidate_key[1],
                    "linkage": candidate_key[2], "r_group": candidate_key[3],
                    "status": "evaluator_exception", "error": str(error),
                }
            observation = _observation(iteration, action.to_dict(), result)
            prediction_adjudication = _score_preregistered_prediction(
                action.to_dict(), result
            )
            reflection_context = _reflection_context(
                rationale, action.to_dict(), observation, decision_state_before
            )
            reflection_attempts = []
            reflection = reflection_rationale = None
            reflection_usages: list[dict[str, Any]] = []
            reflection_feedback = ""
            for reflection_attempt in range(1, MAX_REFLECTION_ATTEMPTS + 1):
                current_context = reflection_context
                if reflection_feedback:
                    current_context += "\nFORMAT FEEDBACK:\n" + reflection_feedback
                try:
                    reflection_raw = scientist.reflect(current_context)
                except StopIteration:
                    reflection_attempts.append({
                        "attempt": reflection_attempt, "context": current_context,
                        "raw_response": "", "parse_error": "Scientist reflection stream ended.",
                        "cost": dict(scientist.last_cost),
                    })
                    break
                except Exception as error:
                    reflection_feedback = f"Scientist reflection call failed: {error}"
                    reflection_attempts.append({
                        "attempt": reflection_attempt, "context": current_context,
                        "raw_response": "", "call_error": str(error),
                        "cost": dict(scientist.last_cost),
                    })
                    continue
                reflection_cost = dict(scientist.last_cost)
                reflection_usages.append(reflection_cost)
                try:
                    reflection_rationale, parsed_reflection = extract_scientist_reflection(
                        reflection_raw
                    )
                    known_hypotheses = notebook_before[
                        "scientist_interpretations"
                    ]["hypotheses"]
                    for update in parsed_reflection.hypothesis_updates:
                        if (update.hypothesis_id not in known_hypotheses
                                and update.revised_statement is None):
                            raise ValueError(
                                "A newly introduced hypothesis requires revised_statement "
                                "to provide its full testable statement."
                            )
                    reflection = parsed_reflection.to_dict()
                    reflection_attempts.append({
                        "attempt": reflection_attempt, "context": current_context,
                        "raw_response": reflection_raw, "accepted": True,
                        "cost": reflection_cost,
                    })
                    break
                except ValueError as error:
                    reflection_feedback = str(error)
                    reflection_attempts.append({
                        "attempt": reflection_attempt, "context": current_context,
                        "raw_response": reflection_raw, "accepted": False,
                        "parse_error": str(error), "cost": reflection_cost,
                    })

            event: dict[str, Any] = {
                "event_type": "experiment", "iteration": iteration,
                "decision_number": decision_number, "action_context": context,
                "raw_action_response": raw, "scientific_rationale": rationale,
                "action": action.to_dict(), "notebook_before": notebook_before,
                "notebook_sha256_before": notebook_sha256(notebook_before),
                "decision_state_before": decision_state_before,
                "decision_state_sha256_before": decision_state_sha256(
                    decision_state_before
                ),
                "portfolio_policy": portfolio_policy,
                "validation_scope": validation_scope,
                "prediction_adjudication": prediction_adjudication,
                "result": result, "oracle_observation": observation,
                "reflection_context": reflection_context,
                "reflection_attempts": reflection_attempts,
                "reflection_scientific_rationale": reflection_rationale,
                "reflection": reflection,
                "cost": {"action": action_cost, "reflection": _usage_total(reflection_usages)},
            }
            provisional_events = events + [{
                "event_type": "experiment", "iteration": iteration,
                "action": action.to_dict(), "result": result, "reflection": reflection,
            }]
            notebook_after = build_notebook(provisional_events)
            decision_state_after = build_decision_state(provisional_events)
            decision_state_after["portfolio"]["policy"] = portfolio_policy
            event["notebook_after"] = notebook_after
            event["notebook_sha256_after"] = notebook_sha256(notebook_after)
            event["decision_state_after"] = decision_state_after
            event["decision_state_sha256_after"] = decision_state_sha256(
                decision_state_after
            )
            _append_event(trace_path, events, event)
            seen.add(candidate_key)
            scaffold_counts[scaffold_key] += 1
            if is_decoration:
                decoration_experiments += 1
                consecutive_decorations += 1
            else:
                consecutive_decorations = 0
            completed += 1
            experiment_completed = True
            print(
                f"experiment {iteration}: {'/'.join(candidate_key)} -> "
                f"{result.get('status')} hit={(result.get('criterion') or {}).get('is_hit')}"
            )
            break

        if not experiment_completed:
            failure_event = {
                "event_type": "campaign_failed" if fixed_horizon else "campaign_stop",
                "iteration": completed + 1,
                "reason": (
                    "Maximum scientist decisions were exhausted without registering a valid "
                    "oracle experiment."
                ),
            }
            _append_event(trace_path, events, failure_event)
            if fixed_horizon:
                raise RuntimeError(failure_event["reason"])
            return events
    return events


def _event_candidate_from_action(action: dict[str, Any]) -> tuple[str, str, str, str]:
    experiment = action.get("experiment", {}) if isinstance(action, dict) else {}
    return tuple(str(experiment.get(name, "?")) for name in (
        "node_id", "linker_id", "linkage", "r_group_id"
    ))  # type: ignore[return-value]


def _mock_result(node_id: str, linker_id: str, linkage: str,
                 r_group_id: str, _: Path) -> dict[str, Any]:
    if linker_id == "L001":
        gap, cbm, vbm = 2.238, -0.172, 2.066
    else:
        gap, cbm, vbm = 2.124, -0.836, 1.288
    stability = 0.80
    criterion = {
        "gap_pass": 1.8 <= gap <= 2.2,
        "gap_margin_floor": round(gap - 1.8, 6),
        "gap_margin_ceiling": round(2.2 - gap, 6),
        "cbm_pass": cbm < 0.0, "cbm_margin": round(-cbm, 6),
        "vbm_pass": vbm >= 1.23, "vbm_margin": round(vbm - 1.23, 6),
        "stability_pass": stability >= 0.70,
        "stability_margin": round(stability - 0.70, 6),
    }
    criterion["is_hit"] = all(criterion[name] for name in (
        "gap_pass", "cbm_pass", "vbm_pass", "stability_pass"
    ))
    return {
        "node": node_id, "linker": linker_id, "linkage": linkage,
        "r_group": r_group_id, "status": "success", "ip_eV": 7.0, "ea_eV": 3.0,
        "gap_xtb_eV": 4.0, "gap_corrected_eV": gap, "cbm_vs_nhe": cbm,
        "vbm_vs_nhe": vbm, "s_linkage": stability, "criterion": criterion,
        "is_hit": criterion["is_hit"],
    }


def _mock_scientist() -> ScriptedScientist:
    plan = """The campaign should establish a chemically robust reference before changing one structural variable at a time. I will use tools to verify the inventory and retain explicit alternatives.

SCIENTIST_PLAN:
{"objective":"Discover distinct four-gate COF scaffolds with interpretable evidence.","landscape_assessment":"The gap window and VBM requirement are coupled, while linkage chemistry controls a separate stability gate.","strategy":"Begin with a robust beta-ketoenamine reference, then change the linker while preserving node, linkage, and R-group so the frame effect is interpretable.","hypotheses":[{"hypothesis_id":"H1","statement":"Changing the diamine linker within a Tp beta-ketoenamine scaffold can lower an overly wide corrected gap while retaining the stability and band-edge gates.","rationale":"The linkage chemistry remains fixed while conjugation length and torsion change with the linker.","confidence":"medium","discriminating_test":"Compare two Tp beta-ketoenamine scaffolds at R001."}],"decision_principles":["Prefer one-variable comparisons when testing a mechanism.","Move to a new scaffold after a hit unless confirmation serves a stated purpose."],"saturation_criteria":["Do not declare a family exhausted without multiple structurally distinct observations and a consistent limiting-gate mechanism."]}"""
    actions = [
        """Before spending the first oracle call I will verify the active inventory composition and linkage options.

SCIENTIST_ACTION:
{"action":"tool","purpose":"Establish the exact search-space composition.","expected_information":"The counts will determine whether the first experiments should be linkage-stratified.","tool":{"name":"inventory_summary","arguments":{}}}""",
        """Tp beta-ketoenamine provides a mechanistically robust starting point. The unsubstituted candidate is a clean reference for a later linker change.

SCIENTIST_ACTION:
{"action":"experiment","purpose":"Establish an interpretable beta-ketoenamine reference.","experiment":{"candidate":{"node_id":"N002","linker_id":"L001","linkage":"beta_ketoenamine","r_group_id":"R001"},"intent":"explore","purpose":"Measure the baseline gap and band-edge margins without an R-group perturbation.","hypothesis_id":"H1","expected_information":"Whether this frame is close enough to the gap window for a linker-level comparison.","prediction":{"axis":"gap","direction":"wider","statement":"The compact phenylene linker will give a corrected gap near or above the upper boundary while the beta-ketoenamine stability gate passes.","reference_iteration":null,"hit_probability":0.35,"gate_predictions":{"gap":false,"cbm":true,"vbm":true,"stability":true},"limiting_gate":"gap","falsification_rule":"A corrected gap at or below 2.10 eV would falsify the proposed wide-gap limitation."},"decision_rule":{"if_supported":"Test a more extended diamine linker at fixed node, linkage, and R-group.","if_falsified":"Reassess whether linker extension is the correct control variable.","if_inconclusive":"Inspect the individual gate margins before selecting the next experiment."}}}""",
        """The reference was slightly too wide, so I will inspect the factual campaign summary before selecting the controlled linker change.

SCIENTIST_ACTION:
{"action":"tool","purpose":"Verify the limiting gate and current scaffold status.","expected_information":"The summary should confirm whether gap refinement, rather than a band-edge or stability repair, is justified.","tool":{"name":"campaign_summary","arguments":{}}}""",
        """The first result supports a controlled linker intervention. I will change only L001 to L002 and retain the Tp node, linkage, and R001 baseline.

SCIENTIST_ACTION:
{"action":"experiment","purpose":"Test the linker-extension hypothesis with a matched structural comparison.","experiment":{"candidate":{"node_id":"N002","linker_id":"L002","linkage":"beta_ketoenamine","r_group_id":"R001"},"intent":"falsify","purpose":"Determine whether changing the diamine frame moves the gap into the target window.","hypothesis_id":"H1","expected_information":"A direct test of whether linker identity can repair the gap without losing other gates.","prediction":{"axis":"gap","direction":"decrease","statement":"Relative to iteration 1, L002 will lower the corrected gap into the target window while retaining passing stability and band-edge gates.","reference_iteration":1,"hit_probability":0.70,"gate_predictions":{"gap":true,"cbm":true,"vbm":true,"stability":true},"limiting_gate":"none","falsification_rule":"Failure of any gate or a corrected gap not lower than iteration 1 would falsify the matched-linker prediction."},"decision_rule":{"if_supported":"Treat linker identity as a productive scaffold-level control and explore a new linker scaffold.","if_falsified":"Reject the simple linker-extension rule and test an alternative structural axis.","if_inconclusive":"Use a matched-pair analysis before another oracle call."}}}""",
    ]
    reflections = [
        """The reference passes stability and band-edge gates but misses narrowly above the gap ceiling. This isolates the immediate design problem to gap reduction and supports a controlled linker comparison.

SCIENTIST_REFLECTION:
{"observation_summary":"N002/L001 beta-ketoenamine passes CBM, VBM, and stability but has a corrected gap above 2.2 eV.","prediction_supported":true,"causal_interpretation":"The robust linkage behaves as expected; the unresolved question is whether linker conjugation can narrow the gap without sacrificing VBM.","hypothesis_updates":[{"hypothesis_id":"H1","status":"active","confidence":"medium","evidence":"The L001 reference is an electronically near-target, stability-passing scaffold suitable for the proposed matched linker test.","revised_statement":null}],"strategy_update":"Keep node, linkage, and R-group fixed and change the linker next.","next_priority":"Measure N002/L002/beta_ketoenamine/R001 as a matched scaffold comparison."}""",
        """The linker change moved the corrected gap into the target window and all four gates pass. The experiment therefore supports linker identity as a productive scaffold-level design axis.

SCIENTIST_REFLECTION:
{"observation_summary":"N002/L002 beta-ketoenamine is a four-gate hit with a lower corrected gap than iteration 1.","prediction_supported":true,"causal_interpretation":"At fixed node, linkage, and R-group, the linker change produced the intended gap reduction while preserving favorable absolute band edges and linkage stability.","hypothesis_updates":[{"hypothesis_id":"H1","status":"supported","confidence":"high","evidence":"The matched comparison lowered the corrected gap from above the ceiling into the target window and produced a hit.","revised_statement":null}],"strategy_update":"Move to a distinct linker scaffold rather than decorating the discovered scaffold.","next_priority":"Test whether the linker trend transfers to another chemically distinct diamine."}""",
    ]
    def envelope(raw: str, marker: str, key: str) -> str:
        rationale, payload = raw.split(marker, 1)
        return json.dumps({
            "scientific_rationale": rationale.strip(),
            key: json.loads(payload.strip()),
        }, ensure_ascii=False)

    return ScriptedScientist(
        envelope(plan, "SCIENTIST_PLAN:", "plan"),
        [envelope(action, "SCIENTIST_ACTION:", "action") for action in actions],
        [
            envelope(reflection, "SCIENTIST_REFLECTION:", "reflection")
            for reflection in reflections
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the auditable COF-SCOUT scientist")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=5)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument("--max-decoration-fraction", type=float, default=0.10)
    parser.add_argument("--max-candidates-per-scaffold", type=int, default=2)
    parser.add_argument("--max-consecutive-decorations", type=int, default=2)
    parser.add_argument("--run-id", default="ara5_mock")
    parser.add_argument("--provider", choices=["openai_compat"], default="openai_compat")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--reference-landscape", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument(
        "--feasible-csv",
        type=Path,
        default=data_path("search_space_v2_feasible.csv"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--publication-mode", action="store_true",
        help=(
            "Enforce a fixed oracle horizon and require machine-scoreable prospective "
            "predictions for every experiment."
        ),
    )
    args = parser.parse_args()
    if (args.max_experiments < 1 or args.max_tool_calls < 0
            or args.max_decisions < 1 or args.request_timeout <= 0
            or not 0 <= args.max_decoration_fraction <= 1
            or args.max_candidates_per_scaffold < 1
            or args.max_consecutive_decorations < 0):
        parser.error(
            "budgets must be positive; tool calls/consecutive decorations may be zero, "
            "and decoration fraction must be between zero and one"
        )

    library = load_library()
    feasible_set = _load_feasible_set()
    toolbox = ScientistToolbox(library, feasible_set)
    system_prompt = build_scientist_system_prompt()
    run_dir = OUTPUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        if args.reference_landscape is not None:
            parser.error("--mock and --reference-landscape cannot be combined")
        scientist: LiveScientist | ScriptedScientist = _mock_scientist()
        evaluator = _mock_result
        model, endpoint, provider_name = "mock", "", "mock"
        oracle_provenance = {
            "oracle_mode": "mock",
            "stability_model": "mock_fixed",
        }
    else:
        key = os.getenv("OPENAI_API_KEY")
        if not key or not args.base_url:
            raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required")
        client = make_client(
            args.provider, args.model, api_key=key, base_url=args.base_url,
            timeout_s=args.request_timeout,
        )
        scientist = LiveScientist(client, system_prompt, args.temperature)
        if args.reference_landscape is not None:
            reference_oracle = ReferenceOracle(
                args.reference_landscape,
                args.feasible_csv,
                args.reference_manifest,
            )
            evaluator = reference_oracle.evaluate
            oracle_provenance = reference_oracle.provenance
        else:
            transfer_fn = load_transfer_function()
            cbm_cal = load_cbm_calibration()

            def evaluator(node_id: str, linker_id: str, linkage: str,
                          r_group_id: str, work_dir: Path) -> dict[str, Any]:
                return evaluate_candidate(
                    node_id, linker_id, linkage, r_group_id, library=library,
                    transfer_fn=transfer_fn, cbm_cal=cbm_cal, work_dir=work_dir,
                )

            oracle_provenance = {
                "oracle_mode": "live_xtb",
                "stability_model": STABILITY_MODEL_VERSION,
            }

        model, endpoint, provider_name = args.model, args.base_url, args.provider

    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        (run_dir / "scientist_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        metadata_path.write_text(json.dumps({
            "run_id": args.run_id, "architecture": "ara5_scientist",
            "model": model, "endpoint": endpoint, "provider": provider_name,
            "temperature": (
                args.temperature if model not in {"mock", "gpt-5.6-luna"} else None
            ),
            "requested_temperature": args.temperature,
            "fixed_sampling": model in {"mock", "gpt-5.6-luna"},
            "started_at": _now(),
            "max_experiments": args.max_experiments,
            "max_tool_calls_per_experiment": args.max_tool_calls,
            "max_decisions_per_experiment": args.max_decisions,
            "max_decoration_fraction": args.max_decoration_fraction,
            "max_candidates_per_scaffold": args.max_candidates_per_scaffold,
            "max_consecutive_decorations": args.max_consecutive_decorations,
            "publication_mode": args.publication_mode,
            "fixed_horizon": args.publication_mode,
            "machine_scored_predictions_required": args.publication_mode,
            "request_timeout_s": args.request_timeout,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "response_protocol": "json_envelope_v1",
            "decision_state_schema_version": DECISION_STATE_SCHEMA_VERSION,
            **oracle_provenance,
            "scientist_system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "observable_reasoning_note": (
                "The trace records emitted scientific rationale and all decision evidence; "
                "it does not claim access to hidden chain-of-thought."
            ),
        }, indent=2), encoding="utf-8")

    if args.resume:
        trace_path = run_dir / "trace.jsonl"
        existing_events = _read_events(trace_path)
        if not existing_events:
            raise RuntimeError(f"Cannot resume because the trace is empty: {trace_path}")
        _append_event(trace_path, existing_events, {
            "event_type": "campaign_resume",
            "completed_experiments": sum(
                event.get("event_type") == "experiment" for event in existing_events
            ),
            "request_timeout_s": args.request_timeout,
            "reason": (
                "Explicit operator-authorized resume after an interrupted or "
                "harness-stopped process."
            ),
        })
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("resume_history", []).append({
            "resumed_at": _now(),
            "completed_experiments": sum(
                event.get("event_type") == "experiment" for event in existing_events
            ),
            "request_timeout_s": args.request_timeout,
        })
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    events = run_campaign(
        scientist, toolbox, evaluator, run_dir, args.max_experiments,
        max_tool_calls=args.max_tool_calls,
        max_decisions_per_experiment=args.max_decisions,
        max_decoration_fraction=args.max_decoration_fraction,
        max_candidates_per_scaffold=args.max_candidates_per_scaffold,
        max_consecutive_decorations=args.max_consecutive_decorations,
        resume=args.resume,
        fixed_horizon=args.publication_mode,
        require_scored_predictions=args.publication_mode,
    )
    notebook = build_notebook(events)
    (run_dir / "final_notebook.json").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    experiments_completed = notebook["deterministic_facts"]["experiments"]
    metadata.update({
        "completed_at": _now(),
        "experiments_completed": experiments_completed,
        "event_count": len(events),
        "run_status": (
            "completed_budget" if experiments_completed >= args.max_experiments
            else "stopped_early"
        ),
        "final_notebook_sha256": notebook_sha256(notebook),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Trace: {run_dir / 'trace.jsonl'}")
    print(f"Notebook: {run_dir / 'final_notebook.json'}")


if __name__ == "__main__":
    main()


def public_main(*, config_path: str, replicate: int) -> None:
    """Run one public campaign from a frozen config through the CLI implementation."""
    import sys

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = config.get("model") or config.get("conditions", {}).get(config.get("condition"), {}).get("model")
    if not model:
        raise ValueError("config must identify a model")
    sys.argv = [
        "cof-scout-campaign", "--publication-mode", "--max-experiments", "50",
        "--run-id", f"public_{Path(config_path).stem}_rep{replicate:02d}",
        "--model", str(model),
    ]
    main()
