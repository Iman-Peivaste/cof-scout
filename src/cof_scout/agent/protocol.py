"""Free-prose plus structured-suffix protocol for the COF-SCOUT scientist."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Callable, TypeVar

from cof_scout.agent.contracts import ScientificPlan, ScientistAction, ScientistReflection


PLAN_SENTINEL = "SCIENTIST_PLAN:"
ACTION_SENTINEL = "SCIENTIST_ACTION:"
REFLECTION_SENTINEL = "SCIENTIST_REFLECTION:"

T = TypeVar("T")


_SENTINEL_ALIASES = {
    PLAN_SENTINEL: {"SCIENTIST_PLAN", "SCIENCE_PLAN"},
    ACTION_SENTINEL: {"SCIENTIST_ACTION", "SCIENCE_ACTION"},
    REFLECTION_SENTINEL: {"SCIENTIST_REFLECTION", "SCIENCE_REFLECTION"},
}

_CONFUSABLE_LATIN = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041d": "H",
    "\u0406": "I", "\u041a": "K", "\u041c": "M", "\u041e": "O", "\u0420": "P",
    "\u0422": "T", "\u0425": "X", "\u0423": "Y",
    "\u0430": "a", "\u0432": "b", "\u0441": "c", "\u0435": "e", "\u043d": "h",
    "\u0456": "i", "\u043a": "k", "\u043c": "m", "\u043e": "o", "\u0440": "p",
    "\u0442": "t", "\u0445": "x", "\u0443": "y",
})


def _edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance for short protocol labels."""
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def _normalize_sentinel_label(line: str) -> str:
    """Normalize harmless model formatting noise in a sentinel-only line."""
    visible = "".join(
        character for character in line
        if unicodedata.category(character) not in {"Cf", "Co"}
    )
    visible = unicodedata.normalize("NFKC", visible).translate(_CONFUSABLE_LATIN)
    label = visible.strip().removesuffix(":").strip().upper()
    label = re.sub(r"[^A-Z0-9\s_-]+", "", label)
    return re.sub(r"[\s_-]+", "_", label)


def _is_sentinel_label(line: str, sentinel: str) -> bool:
    label = _normalize_sentinel_label(line)
    if label in _SENTINEL_ALIASES[sentinel]:
        return True
    suffix = "_" + sentinel.removesuffix(":").removeprefix("SCIENTIST_")
    if not label.endswith(suffix):
        return False
    prefix = label[:-len(suffix)]
    # Four edits covers the most severe observed deletion (``SCI<private>ST``)
    # while the exact PLAN/ACTION/REFLECTION suffix and full payload contract
    # still prevent unrelated labels from being accepted.
    return _edit_distance(prefix, "SCIENTIST") <= 4


def _find_sentinel_line(raw_response: str, sentinel: str) -> tuple[int, int]:
    """Locate a canonical or narrowly defined alias on its own line."""
    offset = 0
    match: tuple[int, int] | None = None
    for line in raw_response.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if _is_sentinel_label(content, sentinel):
            match = (offset, offset + len(line))
        offset += len(line)
    if match is not None:
        return match

    # Retain compatibility with the original same-line canonical protocol.
    marker = raw_response.rfind(sentinel)
    if marker >= 0:
        return marker, marker + len(sentinel)
    raise ValueError(f"Response must end with {sentinel} followed by JSON")


def _strip_fenced_json(value: str) -> str:
    value = value.strip()
    if value.startswith("```json") and value.endswith("```"):
        return value[len("```json"): -3].strip()
    if value.startswith("```") and value.endswith("```"):
        return value[3: -3].strip()
    return value


def _trailing_json_object(raw_response: str) -> tuple[str, dict[str, Any]] | None:
    """Extract an existing trailing JSON object without repairing its content."""
    text = raw_response.strip()
    decoder = json.JSONDecoder()
    match: tuple[int, int, dict[str, Any]] | None = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        tail = text[end:].strip()
        if tail not in {"", "```"}:
            continue
        match = (index, end, value)
    if match is None:
        return None
    start, _, value = match
    prefix = text[:start].rstrip()
    if prefix.endswith("```json"):
        prefix = prefix[:-len("```json")].rstrip()
    elif prefix.endswith("```"):
        prefix = prefix[:-3].rstrip()
    return prefix, value


def _split_legacy_prefix(prefix: str, sentinel: str) -> str:
    """Validate and remove a recognized legacy marker immediately before JSON."""
    prefix = prefix.rstrip()
    line_start = prefix.rfind("\n") + 1
    last_line = prefix[line_start:]
    if _is_sentinel_label(last_line, sentinel):
        return prefix[:line_start].rstrip()

    # Some live responses placed a mildly corrupted marker after rationale on the
    # same line. Only accept a suffix that itself normalizes to the expected marker.
    for offset in range(len(last_line)):
        if offset > 0 and not last_line[offset - 1].isspace():
            continue
        if _is_sentinel_label(last_line[offset:], sentinel):
            return (prefix[:line_start] + last_line[:offset]).rstrip()
    raise ValueError(f"Response must end with {sentinel} followed by JSON")


def _extract_structured_response(raw_response: str, sentinel: str, envelope_key: str,
                                 factory: Callable[[dict[str, Any]], T]) -> tuple[str, T]:
    cleaned = _strip_fenced_json(raw_response)
    if cleaned.lstrip().startswith("{") and "\"scientific_rationale\"" in cleaned:
        try:
            envelope = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON envelope for {envelope_key}; return exactly one complete "
                f"JSON object with scientific_rationale and {envelope_key}, with no "
                "surrounding prose"
            ) from error
        if not isinstance(envelope, dict):
            raise ValueError(f"{envelope_key} envelope must be an object")
        payload = envelope.get(envelope_key)
        if not isinstance(payload, dict):
            raise ValueError(f"{envelope_key} envelope field must be an object")
        rationale = str(envelope.get("scientific_rationale", "")).strip()
        if not rationale:
            raise ValueError("JSON envelope requires scientific_rationale")
        return rationale, factory(payload)

    extracted = _trailing_json_object(cleaned)
    if extracted is not None:
        prefix, value = extracted
        if "scientific_rationale" in value:
            payload = value.get(envelope_key)
            if not isinstance(payload, dict):
                raise ValueError(f"{envelope_key} envelope field must be an object")
            rationale = str(value.get("scientific_rationale", "")).strip()
            if not rationale:
                raise ValueError("JSON envelope requires scientific_rationale")
            return rationale, factory(payload)
        rationale = _split_legacy_prefix(prefix, sentinel)
        return rationale, factory(value)

    marker_start, suffix_start = _find_sentinel_line(raw_response, sentinel)
    rationale = raw_response[:marker_start].rstrip()
    suffix = raw_response[suffix_start:].strip()
    if suffix.startswith("```json") and suffix.endswith("```"):
        suffix = suffix[len("```json"): -3].strip()
    elif suffix.startswith("```") and suffix.endswith("```"):
        suffix = suffix[3: -3].strip()
    try:
        payload = json.loads(suffix)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON after {sentinel}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{sentinel} payload must be an object")
    return rationale, factory(payload)


def extract_scientific_plan(raw_response: str) -> tuple[str, ScientificPlan]:
    """Extract the initial plan while preserving the preceding scientific rationale."""
    return _extract_structured_response(
        raw_response, PLAN_SENTINEL, "plan", ScientificPlan.from_mapping
    )


def extract_scientist_action(raw_response: str) -> tuple[str, ScientistAction]:
    """Extract one tool, experiment, replanning, or stop action."""
    return _extract_structured_response(
        raw_response, ACTION_SENTINEL, "action", ScientistAction.from_mapping
    )


def extract_scientist_reflection(raw_response: str) -> tuple[str, ScientistReflection]:
    """Extract the post-experiment evidence interpretation."""
    return _extract_structured_response(
        raw_response, REFLECTION_SENTINEL, "reflection",
        ScientistReflection.from_mapping,
    )
