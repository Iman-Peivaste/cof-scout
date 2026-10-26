#!/usr/bin/env python3
"""Select a complete outcome-blind LLM portfolio, then evaluate it in fixed order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cof_scout.agent.contracts import PREDICTION_GATES, Prediction, TRACE_SCHEMA_VERSION
from cof_scout.agent.loop import (
    OUTPUT_ROOT,
    _append_event,
    _now,
    _observation,
    _read_events,
    _score_preregistered_prediction,
    build_scientist_system_prompt,
)
from cof_scout.agent.memory import build_notebook, notebook_sha256
from cof_scout.inventory.assembly import load_library
from cof_scout.oracle.calibration import load_cbm_calibration, load_transfer_function
from cof_scout.agent.client import make_client
from cof_scout.oracle.evaluation import evaluate_candidate
from cof_scout.agent.prompt_builder import _load_feasible_set
from cof_scout.oracle.reference import ReferenceOracle
from cof_scout.oracle.stability import STABILITY_MODEL_VERSION


MAX_BATCH_ATTEMPTS = 3


def _portfolio_sha256(entries: list["StaticPortfolioEntry"]) -> str:
    serialized = json.dumps(
        [asdict(entry) for entry in entries], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StaticPortfolioEntry:
    node_id: str
    linker_id: str
    linkage: str
    r_group_id: str
    purpose: str
    chemical_rationale: str
    prediction: Prediction

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "StaticPortfolioEntry":
        if not isinstance(value, dict):
            raise ValueError("each portfolio entry must be an object")
        candidate = value.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("portfolio entry candidate must be an object")

        def required(field: Any, name: str) -> str:
            text = str(field or "").strip()
            if not text:
                raise ValueError(f"{name} must be nonempty")
            return text

        linkage = required(candidate.get("linkage"), "candidate.linkage")
        linkage = linkage.lower().replace("-", "_").replace(" ", "_")
        prediction = Prediction.from_mapping(value.get("prediction"))
        if not prediction.is_machine_scoreable:
            raise ValueError(
                "every static entry requires hit_probability, four gate_predictions, "
                "limiting_gate, and falsification_rule"
            )
        return cls(
            node_id=required(candidate.get("node_id"), "candidate.node_id").upper(),
            linker_id=required(candidate.get("linker_id"), "candidate.linker_id").upper(),
            linkage=linkage,
            r_group_id=required(
                candidate.get("r_group_id"), "candidate.r_group_id"
            ).upper(),
            purpose=required(value.get("purpose"), "portfolio purpose"),
            chemical_rationale=required(
                value.get("chemical_rationale"), "chemical_rationale"
            ),
            prediction=prediction,
        )

    @property
    def candidate_key(self) -> tuple[str, str, str, str]:
        return self.node_id, self.linker_id, self.linkage, self.r_group_id

    @property
    def scaffold_key(self) -> tuple[str, str, str]:
        return self.candidate_key[:3]

    def to_action(self) -> dict[str, Any]:
        return {
            "kind": "experiment",
            "purpose": self.purpose,
            "tool_name": None,
            "tool_arguments": None,
            "expected_information": None,
            "experiment": {
                "node_id": self.node_id,
                "linker_id": self.linker_id,
                "linkage": self.linkage,
                "r_group_id": self.r_group_id,
                "intent": "explore",
                "purpose": self.purpose,
                "hypothesis_id": "H_STATIC",
                "expected_information": (
                    "Outcome-blind test of the preregistered static portfolio prediction."
                ),
                "prediction": asdict(self.prediction),
                "if_supported": "No adaptation; retain the precommitted portfolio order.",
                "if_falsified": "No adaptation; retain the precommitted portfolio order.",
                "if_inconclusive": "No adaptation; retain the precommitted portfolio order.",
            },
            "strategy_revision": None,
            "reason": None,
            "saturation_criterion": None,
        }

    @classmethod
    def from_locked_mapping(cls, value: dict[str, Any]) -> "StaticPortfolioEntry":
        """Rebuild one already locked entry without asking the model again."""
        return cls(
            node_id=str(value["node_id"]),
            linker_id=str(value["linker_id"]),
            linkage=str(value["linkage"]),
            r_group_id=str(value["r_group_id"]),
            purpose=str(value["purpose"]),
            chemical_rationale=str(value["chemical_rationale"]),
            prediction=Prediction.from_mapping(value["prediction"]),
        )


def _parse_batch(raw: str, expected_count: int) -> tuple[str, list[StaticPortfolioEntry]]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid static portfolio JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("static portfolio response must be one JSON object")
    rationale = str(payload.get("scientific_rationale") or "").strip()
    if not rationale:
        raise ValueError("scientific_rationale must be nonempty")
    entries_raw = payload.get("portfolio_entries")
    if not isinstance(entries_raw, list) or len(entries_raw) != expected_count:
        raise ValueError(
            f"portfolio_entries must contain exactly {expected_count} entries"
        )
    return rationale, [StaticPortfolioEntry.from_mapping(item) for item in entries_raw]


def _portfolio_policy_error(entries: list[StaticPortfolioEntry], budget: int,
                            max_decoration_fraction: float,
                            max_candidates_per_scaffold: int,
                            max_consecutive_decorations: int) -> str | None:
    candidate_keys = [entry.candidate_key for entry in entries]
    if len(candidate_keys) != len(set(candidate_keys)):
        return "The static portfolio contains an exact duplicate candidate."
    counts = Counter(entry.scaffold_key for entry in entries)
    if any(count > max_candidates_per_scaffold for count in counts.values()):
        return "The static portfolio exceeds the per-scaffold candidate limit."
    seen: set[tuple[str, str, str]] = set()
    decorations = 0
    consecutive = 0
    for entry in entries:
        if entry.scaffold_key in seen:
            decorations += 1
            consecutive += 1
            if consecutive > max_consecutive_decorations:
                return "The static portfolio exceeds the consecutive-decoration limit."
        else:
            seen.add(entry.scaffold_key)
            consecutive = 0
    if decorations > int(budget * max_decoration_fraction):
        return "The static portfolio exceeds the total decoration budget."
    return None


def _batch_context(batch_number: int, batch_size: int, budget: int,
                   selected: list[StaticPortfolioEntry], feedback: list[str]) -> str:
    schema = {
        "scientific_rationale": "portfolio-level chemical selection rationale",
        "portfolio_entries": [{
            "candidate": {
                "node_id": "N...", "linker_id": "L...",
                "linkage": "...", "r_group_id": "R...",
            },
            "purpose": "why this candidate belongs in the outcome-blind portfolio",
            "chemical_rationale": "concise structure-property rationale",
            "prediction": {
                "axis": "gap|cbm|vbm|stability|overall",
                "direction": "increase|decrease|narrower|wider|improve|worsen|pass|fail|unchanged|positive|negative|uncertain",
                "statement": "prospective falsifiable chemical prediction",
                "reference_iteration": None,
                "hit_probability": 0.5,
                "gate_predictions": {gate: True for gate in PREDICTION_GATES},
                "limiting_gate": "gap|cbm|vbm|stability|none",
                "falsification_rule": "specific oracle outcome that contradicts the prediction",
            },
        }],
    }
    return "\n".join([
        "Select the next batch of an outcome-blind 50-candidate COF portfolio.",
        "All portfolio batches are locked before any oracle calculation is revealed.",
        "Prioritize distinct scaffolds and discriminating chemical coverage. Do not assume",
        "that any previously selected candidate is a hit or miss.",
        f"Batch: {batch_number}; entries required now: {batch_size}; total budget: {budget}.",
        "ALREADY LOCKED CANDIDATES (identities only; no outcomes exist):",
        json.dumps([entry.candidate_key for entry in selected], indent=2),
        "VALIDATION FEEDBACK FROM PRIOR FORMAT/INVENTORY ATTEMPTS:",
        json.dumps(feedback[-5:], indent=2),
        "Return exactly one JSON object and no surrounding prose using this schema:",
        json.dumps(schema, indent=2),
    ])


def select_static_portfolio(
    request_batch: Callable[[str], tuple[str, dict[str, Any]]],
    feasible_set: frozenset[tuple[str, str, str, str]],
    trace_path: Path,
    events: list[dict[str, Any]],
    budget: int = 50,
    batch_size: int = 10,
    max_decoration_fraction: float = 0.10,
    max_candidates_per_scaffold: int = 2,
    max_consecutive_decorations: int = 2,
    initial_entries: list[StaticPortfolioEntry] | None = None,
) -> list[StaticPortfolioEntry]:
    selected = list(initial_entries or [])
    if len(selected) > budget:
        raise ValueError("Resumed static portfolio exceeds the fixed budget")
    initial_policy_error = _portfolio_policy_error(
        selected, budget, max_decoration_fraction,
        max_candidates_per_scaffold, max_consecutive_decorations,
    )
    if initial_policy_error:
        raise ValueError(f"Invalid resumed static portfolio: {initial_policy_error}")
    batch_number = max(
        (
            int(event.get("batch_number", 0)) for event in events
            if event.get("event_type") == "static_portfolio_batch"
        ),
        default=0,
    )
    while len(selected) < budget:
        batch_number += 1
        required = min(batch_size, budget - len(selected))
        feedback: list[str] = []
        for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
            context = _batch_context(
                batch_number, required, budget, selected, feedback
            )
            raw, cost = request_batch(context)
            try:
                rationale, batch = _parse_batch(raw, required)
                unavailable = [
                    entry.candidate_key for entry in batch
                    if entry.candidate_key not in feasible_set
                ]
                if unavailable:
                    raise ValueError(
                        "Candidates absent from the exact active inventory: "
                        + json.dumps(unavailable[:10])
                    )
                combined = selected + batch
                policy_error = _portfolio_policy_error(
                    combined, budget, max_decoration_fraction,
                    max_candidates_per_scaffold, max_consecutive_decorations,
                )
                if policy_error:
                    raise ValueError(policy_error)
                _append_event(trace_path, events, {
                    "event_type": "static_portfolio_batch",
                    "batch_number": batch_number,
                    "attempt": attempt,
                    "context": context,
                    "raw_response": raw,
                    "scientific_rationale": rationale,
                    "portfolio_entries": [asdict(entry) for entry in batch],
                    "cost": cost,
                })
                selected = combined
                break
            except ValueError as error:
                feedback.append(str(error))
                _append_event(trace_path, events, {
                    "event_type": "static_portfolio_rejected",
                    "batch_number": batch_number,
                    "attempt": attempt,
                    "context": context,
                    "raw_response": raw,
                    "rejection": str(error),
                    "cost": cost,
                })
        else:
            raise RuntimeError(
                f"Static portfolio batch {batch_number} failed validation"
            )
    _append_event(trace_path, events, {
        "event_type": "static_portfolio_locked",
        "candidate_count": len(selected),
        "portfolio_sha256": _portfolio_sha256(selected),
        "portfolio": [asdict(entry) for entry in selected],
    })
    return selected


def evaluate_static_portfolio(
    entries: list[StaticPortfolioEntry], evaluator: Callable[..., dict[str, Any]],
    run_dir: Path, trace_path: Path, events: list[dict[str, Any]],
) -> None:
    completed_events = [
        event for event in events if event.get("event_type") == "experiment"
    ]
    if len(completed_events) > len(entries):
        raise ValueError("Static trace contains more experiments than locked entries")
    seen_scaffolds: set[tuple[str, str, str]] = set()
    for iteration, event in enumerate(completed_events, 1):
        experiment = event.get("action", {}).get("experiment", {})
        observed = tuple(str(experiment.get(field)) for field in (
            "node_id", "linker_id", "linkage", "r_group_id",
        ))
        if observed != entries[iteration - 1].candidate_key:
            raise ValueError(
                f"Static trace experiment {iteration} does not match the locked portfolio"
            )
        seen_scaffolds.add(entries[iteration - 1].scaffold_key)
    for iteration, entry in enumerate(entries, 1):
        if iteration <= len(completed_events):
            continue
        action = entry.to_action()
        notebook_before = build_notebook(events)
        try:
            result = evaluator(*entry.candidate_key, run_dir / "xtb")
        except Exception as error:
            result = {
                "node": entry.node_id, "linker": entry.linker_id,
                "linkage": entry.linkage, "r_group": entry.r_group_id,
                "status": "evaluator_exception", "error": str(error),
            }
        observation = _observation(iteration, action, result)
        adjudication = _score_preregistered_prediction(action, result)
        event: dict[str, Any] = {
            "event_type": "experiment",
            "iteration": iteration,
            "decision_number": 0,
            "scientific_rationale": entry.chemical_rationale,
            "action": action,
            "notebook_before": notebook_before,
            "notebook_sha256_before": notebook_sha256(notebook_before),
            "portfolio_policy": {"condition": "one_shot_outcome_blind"},
            "validation_scope": None,
            "prediction_adjudication": adjudication,
            "result": result,
            "oracle_observation": observation,
            "reflection_attempts": [],
            "reflection_scientific_rationale": None,
            "reflection": None,
            "cost": {"action": {}, "reflection": {}},
        }
        provisional = events + [{
            "event_type": "experiment", "iteration": iteration,
            "action": action, "result": result, "reflection": None,
        }]
        notebook_after = build_notebook(provisional)
        event["notebook_after"] = notebook_after
        event["notebook_sha256_after"] = notebook_sha256(notebook_after)
        _append_event(trace_path, events, event)
        seen_scaffolds.add(entry.scaffold_key)
        print(
            f"static experiment {iteration}: {'/'.join(entry.candidate_key)} -> "
            f"{result.get('status')} hit={(result.get('criterion') or {}).get('is_hit')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
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
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.budget < 1 or args.batch_size < 1 or args.batch_size > args.budget:
        parser.error("budget and batch size must be positive, with batch size <= budget")

    run_dir = OUTPUT_ROOT / args.run_id
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists() and not args.resume:
        parser.error(f"trace already exists: {trace_path}; use --resume")
    if args.resume and not trace_path.is_file():
        parser.error(f"cannot resume because the trace does not exist: {trace_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    events = _read_events(trace_path) if args.resume else []
    system_prompt = build_scientist_system_prompt() + "\n\n" + (
        "You are selecting a complete outcome-blind portfolio. No oracle result will be "
        "shown until every candidate is locked. Return only the requested JSON envelope."
    )
    if args.reference_landscape is not None:
        reference_oracle = ReferenceOracle(
            args.reference_landscape,
            args.feasible_csv,
            args.reference_manifest,
        )
        evaluator = reference_oracle.evaluate
        oracle_provenance = reference_oracle.provenance
    else:
        library = load_library()
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

    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("oracle_mode") != oracle_provenance["oracle_mode"]:
            raise ValueError("Resume requested with a different oracle mode")
        reference_hash = oracle_provenance.get("reference_landscape_sha256")
        if reference_hash and metadata.get("reference_landscape_sha256") != reference_hash:
            raise ValueError("Resume requested with a different reference landscape")
    else:
        metadata = {
            "run_id": args.run_id,
            "architecture": "ara5_static_outcome_blind_portfolio",
            "model": args.model,
            "endpoint": args.base_url,
            "provider": args.provider,
            "temperature": None if args.model == "gpt-5.6-luna" else args.temperature,
            "requested_temperature": args.temperature,
            "fixed_sampling": args.model == "gpt-5.6-luna",
            "started_at": _now(),
            "max_experiments": args.budget,
            "portfolio_batch_size": args.batch_size,
            "request_timeout_s": args.request_timeout,
            "fixed_horizon": True,
            "outcomes_hidden_until_portfolio_lock": True,
            **oracle_provenance,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "scientist_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (run_dir / "scientist_system_prompt.txt").write_text(
            system_prompt, encoding="utf-8"
        )

    if args.resume:
        _append_event(trace_path, events, {
            "event_type": "campaign_resume",
            "completed_experiments": sum(
                event.get("event_type") == "experiment" for event in events
            ),
            "request_timeout_s": args.request_timeout,
            "reason": "Operator-authorized resume of the locked static condition.",
        })

    if not any(event.get("event_type") == "campaign_plan" for event in events):
        plan = {
            "objective": "Select a diverse 50-candidate portfolio before observing outcomes.",
            "landscape_assessment": "Outcome-blind static chemical selection control.",
            "strategy": "Commit all candidates prospectively; do not adapt to oracle evidence.",
            "hypotheses": [{
                "hypothesis_id": "H_STATIC",
                "statement": "Static chemical knowledge can prioritize promising diverse scaffolds.",
                "rationale": "The language model can reason from public structures without outcomes.",
                "confidence": "medium",
                "discriminating_test": "Evaluate the locked portfolio at the same oracle budget.",
            }],
            "decision_principles": ["Prioritize distinct scaffolds and chemical coverage."],
            "saturation_criteria": ["Not applicable: the condition has a fixed horizon."],
        }
        _append_event(trace_path, events, {
            "event_type": "campaign_plan", "attempt": 1,
            "scientific_rationale": "Outcome-blind static portfolio control.",
            "plan": plan, "cost": {},
        })

    locked_event = next(
        (
            event for event in events
            if event.get("event_type") == "static_portfolio_locked"
        ),
        None,
    )
    if locked_event is not None:
        entries = [
            StaticPortfolioEntry.from_locked_mapping(value)
            for value in locked_event.get("portfolio", [])
        ]
        if len(entries) != args.budget:
            raise ValueError("Locked static portfolio does not match the fixed budget")
        if locked_event.get("portfolio_sha256") != _portfolio_sha256(entries):
            raise ValueError("Locked static portfolio hash mismatch")
    else:
        initial_entries = [
            StaticPortfolioEntry.from_locked_mapping(value)
            for event in events
            if event.get("event_type") == "static_portfolio_batch"
            for value in event.get("portfolio_entries", [])
        ]
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not args.base_url:
            raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required")
        client = make_client(
            args.provider, args.model, api_key=api_key, base_url=args.base_url,
            timeout_s=args.request_timeout,
        )

        def request_batch(context: str) -> tuple[str, dict[str, Any]]:
            return client.chat_structured(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": context}],
                temperature=args.temperature, max_tokens=12000, json_object=True,
            )

        entries = select_static_portfolio(
            request_batch, _load_feasible_set(), trace_path, events,
            budget=args.budget, batch_size=args.batch_size,
            initial_entries=initial_entries,
        )

    evaluate_static_portfolio(entries, evaluator, run_dir, trace_path, events)
    notebook = build_notebook(events)
    (run_dir / "final_notebook.json").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata.update({
        "completed_at": _now(),
        "experiments_completed": sum(
            event.get("event_type") == "experiment" for event in events
        ),
        "event_count": len(events),
        "run_status": "completed_budget",
        "final_notebook_sha256": notebook_sha256(notebook),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Trace: {trace_path}")


if __name__ == "__main__":
    main()
