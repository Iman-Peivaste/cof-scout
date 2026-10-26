from __future__ import annotations

from dataclasses import dataclass

GAP_MIN = 1.8
GAP_MAX = 2.2
CBM_MAX = 0.0
VBM_MIN = 1.23
PERSISTENCE_MIN = 0.5


@dataclass(frozen=True)
class GateVerdict:
    gap: bool
    cbm: bool
    vbm: bool
    persistence: bool

    @property
    def hit(self) -> bool:
        return self.gap and self.cbm and self.vbm and self.persistence


def evaluate_gates(
    *, gap_ev: float, cbm_v: float, vbm_v: float, persistence: float
) -> GateVerdict:
    return GateVerdict(
        gap=GAP_MIN <= gap_ev <= GAP_MAX,
        cbm=cbm_v <= CBM_MAX,
        vbm=vbm_v >= VBM_MIN,
        persistence=persistence >= PERSISTENCE_MIN,
    )
