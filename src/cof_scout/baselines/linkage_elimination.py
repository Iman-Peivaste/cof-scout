from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinkageEliminationState:
    """Pre-specified corrected linkage-elimination rule.

    A class is removed only after ``k`` consecutive hitless scaffold evaluations,
    never after any hit, and never when it is the last remaining class.
    """

    linkages: tuple[str, ...]
    k: int
    active: set[str] = field(init=False)
    hits: dict[str, int] = field(init=False)
    misses: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        self.active = set(self.linkages)
        self.hits = dict.fromkeys(self.linkages, 0)
        self.misses = dict.fromkeys(self.linkages, 0)

    def observe(self, linkage: str, *, hit: bool) -> None:
        if linkage not in self.hits:
            raise KeyError(linkage)
        if hit:
            self.hits[linkage] += 1
            self.misses[linkage] = 0
            self.active.add(linkage)
            return
        self.misses[linkage] += 1
        if (
            self.hits[linkage] == 0
            and self.misses[linkage] >= self.k
            and linkage in self.active
            and len(self.active) > 1
        ):
            self.active.remove(linkage)

    def eligible(self, linkage: str) -> bool:
        return linkage in self.active
