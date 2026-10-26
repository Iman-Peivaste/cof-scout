"""Transfer-function and band-edge calibration for xTB fragment results."""

import json
from pathlib import Path

from cof_scout.paths import data_path

_DEFAULT_SLOPE = 0.4603819837174193
_DEFAULT_INTERCEPT = -0.09832894975094009
_DEFAULT_CBM_OFFSET = 2.62


def load_transfer_function(path: str | Path | None = None) -> dict:
    """Load the empirical xTB-to-PBE gap transfer function."""
    target = Path(path) if path is not None else data_path("week3v2_clean6_transfer_function.json")
    return json.loads(target.read_text(encoding="utf-8"))


def load_cbm_calibration(path: str | Path | None = None) -> dict:
    """Load the xTB-EA-to-NHE CBM offset."""
    target = Path(path) if path is not None else data_path("cbm_calibration.json")
    return json.loads(target.read_text(encoding="utf-8"))


def apply_transfer_function(gap_xtb: float, transfer_fn: dict | None = None) -> float:
    """Convert ``IP - EA`` to the PBE-calibrated gap in eV."""
    transfer_fn = transfer_fn or {}
    return (transfer_fn.get("slope", _DEFAULT_SLOPE) * gap_xtb
            + transfer_fn.get("intercept", _DEFAULT_INTERCEPT))


def compute_band_edges(ea_ev: float, gap_corrected: float,
                       cbm_calibration: dict | None = None) -> tuple[float, float]:
    """Return ``(CBM, VBM)`` in V vs NHE."""
    cbm_calibration = cbm_calibration or {}
    cbm = -ea_ev + cbm_calibration.get("C_offset", _DEFAULT_CBM_OFFSET)
    return cbm, cbm + gap_corrected


def full_calibration(ip_ev: float, ea_ev: float, transfer_fn: dict | None = None,
                     cbm_calibration: dict | None = None) -> dict:
    """Compute raw gap, calibrated gap, CBM, and VBM from xTB IP/EA values."""
    gap_xtb = ip_ev - ea_ev
    gap_corrected = apply_transfer_function(gap_xtb, transfer_fn)
    cbm, vbm = compute_band_edges(ea_ev, gap_corrected, cbm_calibration)
    return {
        "gap_xtb_eV": round(gap_xtb, 6),
        "gap_corrected_eV": round(gap_corrected, 6),
        "cbm_vs_nhe": round(cbm, 6),
        "vbm_vs_nhe": round(vbm, 6),
    }


if __name__ == "__main__":
    transfer_function = load_transfer_function()
    cbm_calibration = load_cbm_calibration()
    result = full_calibration(7.775, 3.127, transfer_function, cbm_calibration)
    print("=== N016 / L011 / hydrazone / R001 ===")
    print(f"  gap_xTB        = {result['gap_xtb_eV']:.4f} eV")
    print(f"  gap_corrected  = {result['gap_corrected_eV']:.4f} eV")
    print(f"  CBM vs NHE     = {result['cbm_vs_nhe']:+.4f} V")
    print(f"  VBM vs NHE     = {result['vbm_vs_nhe']:+.4f} V")
    gap, cbm, vbm = (result["gap_corrected_eV"], result["cbm_vs_nhe"], result["vbm_vs_nhe"])
    print("\n=== Four-gate electronic check ===")
    print(f"  Gap gate [1.8, 2.2]: {'PASS' if 1.8 <= gap <= 2.2 else 'FAIL'}")
    print(f"  CBM gate (< 0 V):     {'PASS' if cbm < 0 else 'FAIL'}")
    print(f"  VBM gate (>= +1.23):  {'PASS' if vbm >= 1.23 else 'FAIL'}")
