from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardState:
    evaluated: set[str] = field(default_factory=set)
    scaffold_counts: dict[str, int] = field(default_factory=dict)
    decorated_evaluations: int = 0
    total_evaluations: int = 0
    consecutive_decorations: int = 0
    decoration_cap: float = 0.10
    per_scaffold_limit: int = 2
    consecutive_decoration_limit: int = 2


def validate_action(
    state: GuardState,
    *,
    candidate_id: str,
    scaffold_id: str,
    decorated: bool,
    compatible: bool,
) -> tuple[bool, str | None]:
    if candidate_id in state.evaluated:
        return False, "duplicate"
    if not compatible:
        return False, "incompatible"
    if state.scaffold_counts.get(scaffold_id, 0) >= state.per_scaffold_limit:
        return False, "per_scaffold_limit"
    if decorated and state.consecutive_decorations >= state.consecutive_decoration_limit:
        return False, "consecutive_decoration_limit"
    projected_total = state.total_evaluations + 1
    projected_decorated = state.decorated_evaluations + int(decorated)
    if decorated and projected_decorated / projected_total > state.decoration_cap:
        return False, "decoration_cap"
    return True, None
