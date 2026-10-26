"""Versioned contracts for the experimentalist-facing COF-SCOUT scientist.

The contracts distinguish model-authored scientific interpretation from facts
computed by Python.  Raw model responses are always stored separately; these
objects are only the validated control fields used by the campaign harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


TRACE_SCHEMA_VERSION = "ara5-v1"
NOTEBOOK_SCHEMA_VERSION = "ara5-notebook-v1"

ALLOWED_AXES = frozenset({"gap", "cbm", "vbm", "stability", "overall"})
ALLOWED_DIRECTIONS = frozenset({
    "increase", "decrease", "narrower", "wider", "improve", "worsen",
    "pass", "fail", "unchanged", "positive", "negative", "uncertain",
})
PREDICTION_GATES = ("gap", "cbm", "vbm", "stability")
ALLOWED_LIMITING_GATES = frozenset({*PREDICTION_GATES, "none"})
ALLOWED_CONFIDENCE = frozenset({"low", "medium", "high"})
ALLOWED_INTENTS = frozenset({"explore", "refine", "falsify", "validate"})
ALLOWED_HYPOTHESIS_STATUS = frozenset({
    "active", "supported", "weakened", "rejected", "revised",
})
VALIDATION_SCOPE_CROSS_SCAFFOLD = "cross_scaffold_hypothesis_validation"
VALIDATION_SCOPE_LOCAL = "local_scaffold_confirmation"


def classify_validation_scope(intent: str | None,
                              is_new_scaffold: bool) -> str | None:
    """Classify validation provenance without asking the model to self-label it."""
    if intent != "validate":
        return None
    return (
        VALIDATION_SCOPE_CROSS_SCAFFOLD
        if is_new_scaffold else VALIDATION_SCOPE_LOCAL
    )


class ActionKind(StrEnum):
    """Actions available to the scientist between oracle experiments."""

    TOOL = "tool"
    EXPERIMENT = "experiment"
    REPLAN = "replan"
    STOP = "stop"


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonempty")
    return text


def _confidence(value: Any, field: str = "confidence") -> str:
    confidence = str(value or "").strip().lower()
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"{field} must be low, medium, or high")
    return confidence


@dataclass(frozen=True)
class InitialHypothesis:
    """One testable hypothesis in the initial scientific plan."""

    hypothesis_id: str
    statement: str
    rationale: str
    confidence: str
    discriminating_test: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "InitialHypothesis":
        if not isinstance(value, dict):
            raise ValueError("each initial hypothesis must be an object")
        return cls(
            hypothesis_id=_required_text(value.get("hypothesis_id"), "hypothesis_id"),
            statement=_required_text(value.get("statement"), "hypothesis statement"),
            rationale=_required_text(value.get("rationale"), "hypothesis rationale"),
            confidence=_confidence(value.get("confidence")),
            discriminating_test=_required_text(
                value.get("discriminating_test"), "discriminating_test"
            ),
        )


@dataclass(frozen=True)
class ScientificPlan:
    """Model-authored campaign strategy produced before the first experiment."""

    objective: str
    landscape_assessment: str
    strategy: str
    hypotheses: tuple[InitialHypothesis, ...]
    decision_principles: tuple[str, ...]
    saturation_criteria: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ScientificPlan":
        if not isinstance(value, dict):
            raise ValueError("SCIENTIST_PLAN must be a JSON object")
        hypotheses_raw = value.get("hypotheses")
        if not isinstance(hypotheses_raw, list) or not hypotheses_raw:
            raise ValueError("hypotheses must be a nonempty list")
        principles = value.get("decision_principles")
        saturation = value.get("saturation_criteria")
        if not isinstance(principles, list) or not principles:
            raise ValueError("decision_principles must be a nonempty list")
        if not isinstance(saturation, list) or not saturation:
            raise ValueError("saturation_criteria must be a nonempty list")
        hypotheses = tuple(InitialHypothesis.from_mapping(item) for item in hypotheses_raw)
        hypothesis_ids = [item.hypothesis_id for item in hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("initial hypothesis IDs must be unique")
        return cls(
            objective=_required_text(value.get("objective"), "objective"),
            landscape_assessment=_required_text(
                value.get("landscape_assessment"), "landscape_assessment"
            ),
            strategy=_required_text(value.get("strategy"), "strategy"),
            hypotheses=hypotheses,
            decision_principles=tuple(
                _required_text(item, "decision principle") for item in principles
            ),
            saturation_criteria=tuple(
                _required_text(item, "saturation criterion") for item in saturation
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Prediction:
    """A falsifiable prediction registered before an oracle call."""

    axis: str
    direction: str
    statement: str
    reference_iteration: int | None
    hit_probability: float | None = None
    gate_predictions: dict[str, bool] | None = None
    limiting_gate: str | None = None
    falsification_rule: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Prediction":
        if not isinstance(value, dict):
            raise ValueError("prediction must be an object")
        axis = str(value.get("axis", "")).strip().lower()
        direction = str(value.get("direction", "")).strip().lower()
        if axis not in ALLOWED_AXES:
            raise ValueError(f"prediction axis must be one of {sorted(ALLOWED_AXES)}")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(
                f"prediction direction must be one of {sorted(ALLOWED_DIRECTIONS)}"
            )
        reference = value.get("reference_iteration")
        if reference is not None and (not isinstance(reference, int) or reference < 1):
            raise ValueError("reference_iteration must be a positive integer or null")

        hit_probability = value.get("hit_probability")
        if hit_probability is not None:
            if isinstance(hit_probability, bool) or not isinstance(
                hit_probability, (int, float)
            ):
                raise ValueError("prediction hit_probability must be a number from 0 to 1")
            hit_probability = float(hit_probability)
            if not 0.0 <= hit_probability <= 1.0:
                raise ValueError("prediction hit_probability must be between 0 and 1")

        gate_predictions = value.get("gate_predictions")
        if gate_predictions is not None:
            if not isinstance(gate_predictions, dict):
                raise ValueError("prediction gate_predictions must be an object")
            if set(gate_predictions) != set(PREDICTION_GATES):
                raise ValueError(
                    "prediction gate_predictions must contain exactly gap, cbm, vbm, "
                    "and stability"
                )
            if not all(isinstance(gate_predictions[name], bool) for name in PREDICTION_GATES):
                raise ValueError("every prediction gate prediction must be boolean")
            gate_predictions = {
                name: gate_predictions[name] for name in PREDICTION_GATES
            }

        limiting_gate = value.get("limiting_gate")
        if limiting_gate is not None:
            limiting_gate = str(limiting_gate).strip().lower()
            if limiting_gate not in ALLOWED_LIMITING_GATES:
                raise ValueError(
                    "prediction limiting_gate must be gap, cbm, vbm, stability, or none"
                )

        falsification_rule = value.get("falsification_rule")
        if falsification_rule is not None:
            falsification_rule = _required_text(
                falsification_rule, "prediction falsification_rule"
            )
        return cls(
            axis=axis,
            direction=direction,
            statement=_required_text(value.get("statement"), "prediction statement"),
            reference_iteration=reference,
            hit_probability=hit_probability,
            gate_predictions=gate_predictions,
            limiting_gate=limiting_gate,
            falsification_rule=falsification_rule,
        )

    @property
    def is_machine_scoreable(self) -> bool:
        """Return whether every publication-facing prospective field is present."""
        return (
            self.hit_probability is not None
            and self.gate_predictions is not None
            and self.limiting_gate is not None
            and self.falsification_rule is not None
        )


@dataclass(frozen=True)
class ExperimentProposal:
    """A complete, pre-registered experimental decision."""

    node_id: str
    linker_id: str
    linkage: str
    r_group_id: str
    intent: str
    purpose: str
    hypothesis_id: str | None
    expected_information: str
    prediction: Prediction
    if_supported: str
    if_falsified: str
    if_inconclusive: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ExperimentProposal":
        if not isinstance(value, dict):
            raise ValueError("experiment must be an object")
        candidate = value.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("experiment.candidate must be an object")
        intent = str(value.get("intent", "")).strip().lower()
        if intent not in ALLOWED_INTENTS:
            raise ValueError(f"experiment intent must be one of {sorted(ALLOWED_INTENTS)}")
        decision_rule = value.get("decision_rule")
        if not isinstance(decision_rule, dict):
            raise ValueError("decision_rule must be an object")
        hypothesis_id = value.get("hypothesis_id")
        if hypothesis_id is not None:
            hypothesis_id = _required_text(hypothesis_id, "hypothesis_id")
        linkage = str(candidate.get("linkage", "")).strip().lower()
        linkage = linkage.replace("-", "_").replace(" ", "_")
        return cls(
            node_id=_required_text(candidate.get("node_id"), "candidate.node_id").upper(),
            linker_id=_required_text(candidate.get("linker_id"), "candidate.linker_id").upper(),
            linkage=linkage,
            r_group_id=_required_text(
                candidate.get("r_group_id"), "candidate.r_group_id"
            ).upper(),
            intent=intent,
            purpose=_required_text(value.get("purpose"), "experiment purpose"),
            hypothesis_id=hypothesis_id,
            expected_information=_required_text(
                value.get("expected_information"), "expected_information"
            ),
            prediction=Prediction.from_mapping(value.get("prediction")),
            if_supported=_required_text(
                decision_rule.get("if_supported"), "decision_rule.if_supported"
            ),
            if_falsified=_required_text(
                decision_rule.get("if_falsified"), "decision_rule.if_falsified"
            ),
            if_inconclusive=_required_text(
                decision_rule.get("if_inconclusive"), "decision_rule.if_inconclusive"
            ),
        )

    @property
    def candidate_key(self) -> tuple[str, str, str, str]:
        return self.node_id, self.linker_id, self.linkage, self.r_group_id

    @property
    def scaffold_key(self) -> tuple[str, str, str]:
        return self.candidate_key[:3]


@dataclass(frozen=True)
class ScientistAction:
    """One validated action requested by the scientist."""

    kind: ActionKind
    purpose: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    expected_information: str | None = None
    experiment: ExperimentProposal | None = None
    strategy_revision: str | None = None
    reason: str | None = None
    saturation_criterion: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ScientistAction":
        if not isinstance(value, dict):
            raise ValueError("SCIENTIST_ACTION must be a JSON object")
        try:
            kind = ActionKind(str(value.get("action", "")).strip().lower())
        except ValueError as error:
            raise ValueError("action must be tool, experiment, replan, or stop") from error
        purpose = _required_text(value.get("purpose"), "action purpose")
        if kind is ActionKind.TOOL:
            tool = value.get("tool")
            if not isinstance(tool, dict):
                raise ValueError("tool action requires a tool object")
            arguments = tool.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("tool.arguments must be an object")
            return cls(
                kind=kind,
                purpose=purpose,
                tool_name=_required_text(tool.get("name"), "tool.name"),
                tool_arguments=arguments,
                expected_information=_required_text(
                    value.get("expected_information"), "expected_information"
                ),
            )
        if kind is ActionKind.EXPERIMENT:
            return cls(
                kind=kind,
                purpose=purpose,
                experiment=ExperimentProposal.from_mapping(value.get("experiment")),
            )
        if kind is ActionKind.REPLAN:
            return cls(
                kind=kind,
                purpose=purpose,
                strategy_revision=_required_text(
                    value.get("strategy_revision"), "strategy_revision"
                ),
                reason=_required_text(value.get("reason"), "replan reason"),
            )
        return cls(
            kind=kind,
            purpose=purpose,
            reason=_required_text(value.get("reason"), "stop reason"),
            saturation_criterion=_required_text(
                value.get("saturation_criterion"), "saturation_criterion"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        return value


@dataclass(frozen=True)
class HypothesisUpdate:
    """Evidence-linked update emitted after an experiment."""

    hypothesis_id: str
    status: str
    confidence: str
    evidence: str
    revised_statement: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "HypothesisUpdate":
        if not isinstance(value, dict):
            raise ValueError("each hypothesis update must be an object")
        status = str(value.get("status", "")).strip().lower()
        if status not in ALLOWED_HYPOTHESIS_STATUS:
            raise ValueError(
                f"hypothesis status must be one of {sorted(ALLOWED_HYPOTHESIS_STATUS)}"
            )
        revised = value.get("revised_statement")
        if revised is not None:
            revised = _required_text(revised, "revised_statement")
        if status == "revised" and revised is None:
            raise ValueError("revised hypotheses require revised_statement")
        return cls(
            hypothesis_id=_required_text(value.get("hypothesis_id"), "hypothesis_id"),
            status=status,
            confidence=_confidence(value.get("confidence")),
            evidence=_required_text(value.get("evidence"), "hypothesis evidence"),
            revised_statement=revised,
        )


@dataclass(frozen=True)
class ScientistReflection:
    """Structured interpretation produced after observing an oracle result."""

    observation_summary: str
    prediction_supported: bool | None
    causal_interpretation: str
    hypothesis_updates: tuple[HypothesisUpdate, ...]
    strategy_update: str | None
    next_priority: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ScientistReflection":
        if not isinstance(value, dict):
            raise ValueError("SCIENTIST_REFLECTION must be a JSON object")
        supported = value.get("prediction_supported")
        if supported is not None and not isinstance(supported, bool):
            raise ValueError("prediction_supported must be boolean or null")
        updates = value.get("hypothesis_updates", [])
        if not isinstance(updates, list):
            raise ValueError("hypothesis_updates must be a list")
        return cls(
            observation_summary=_required_text(
                value.get("observation_summary"), "observation_summary"
            ),
            prediction_supported=supported,
            causal_interpretation=_required_text(
                value.get("causal_interpretation"), "causal_interpretation"
            ),
            hypothesis_updates=tuple(
                HypothesisUpdate.from_mapping(item) for item in updates
            ),
            strategy_update=(
                _required_text(value.get("strategy_update"), "strategy_update")
                if value.get("strategy_update") is not None else None
            ),
            next_priority=_required_text(value.get("next_priority"), "next_priority"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
