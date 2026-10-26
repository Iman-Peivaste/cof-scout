"""The complete COF fragment evaluation oracle."""

import time
from pathlib import Path

from cof_scout.inventory.assembly import assemble_fragment, load_library
from cof_scout.oracle.calibration import full_calibration, load_cbm_calibration, load_transfer_function
from cof_scout.oracle.geometry import build_geometry
from cof_scout.oracle.stability import STABILITY_THRESHOLD, assess_stability
from cof_scout.oracle.xtb import evaluate_xtb

GAP_MIN, GAP_MAX = 1.8, 2.2
CBM_MAX, VBM_MIN, STAB_MIN = 0.0, 1.23, STABILITY_THRESHOLD


def _assess(result: dict) -> dict:
    """Apply the four photocatalyst hit gates to a calibrated result."""
    gap = result["gap_corrected_eV"]
    cbm = result["cbm_vs_nhe"]
    vbm = result["vbm_vs_nhe"]
    stability = (
        result["stability_score"]
        if "stability_score" in result else result["s_linkage"]
    )
    gap_pass = GAP_MIN <= gap <= GAP_MAX
    cbm_pass = cbm < CBM_MAX
    vbm_pass = vbm >= VBM_MIN
    stability_pass = stability >= STAB_MIN
    electronic_pass = gap_pass and cbm_pass and vbm_pass
    return {
        "gap_pass": gap_pass,
        "gap_margin_floor": round(gap - GAP_MIN, 6),
        "gap_margin_ceiling": round(GAP_MAX - gap, 6),
        "cbm_pass": cbm_pass,
        "cbm_margin": round(CBM_MAX - cbm, 6),
        "vbm_pass": vbm_pass,
        "vbm_margin": round(vbm - VBM_MIN, 6),
        "stability_pass": stability_pass,
        "stability_margin": round(stability - STAB_MIN, 6),
        "electronic_pass": electronic_pass,
        "is_hit": electronic_pass and stability_pass,
    }


def evaluate_candidate(node_id: str, linker_id: str, linkage: str,
                       r_group_id: str = "R001", library: dict | None = None,
                       transfer_fn: dict | None = None, cbm_cal: dict | None = None,
                       work_dir: str | Path = "output/oracle") -> dict:
    """Evaluate one candidate through assembly, geometry, xTB, and calibration.

    The returned dict always contains ``status``.  A non-``success`` status is
    an expected screening outcome, not an exception for callers to handle.
    """
    start = time.perf_counter()
    linkage_norm = linkage.strip().lower().replace("-", "_").replace(" ", "_")
    result = {"node": node_id, "linker": linker_id, "linkage": linkage_norm,
              "r_group": r_group_id, "status": "pending"}
    library = load_library() if library is None else library
    transfer_fn = load_transfer_function() if transfer_fn is None else transfer_fn
    cbm_cal = load_cbm_calibration() if cbm_cal is None else cbm_cal
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fragment = assemble_fragment(node_id, linker_id, linkage_norm, r_group_id, library)
    if fragment is None:
        result["status"] = "assembly_failed"
        result["compute_time_s"] = round(time.perf_counter() - start, 2)
        return result
    result["fragment_smiles"] = fragment

    namespace = f"{node_id}_{linker_id}_{linkage_norm}_{r_group_id}"
    xyz_path = work_dir / f"{namespace}.xyz"
    try:
        build_geometry(fragment, xyz_path)
    except Exception as error:
        result.update(status="build_failed", error=str(error),
                      compute_time_s=round(time.perf_counter() - start, 2))
        return result

    ip_ev, ea_ev, gap_xtb = evaluate_xtb(xyz_path, work_dir, namespace=namespace)
    if ip_ev is None:
        result.update(status="xtb_failed", compute_time_s=round(time.perf_counter() - start, 2))
        return result
    result.update(ip_eV=round(ip_ev, 6), ea_eV=round(ea_ev, 6), gap_xtb_eV=round(gap_xtb, 6))

    result.update(full_calibration(ip_ev, ea_ev, transfer_fn, cbm_cal))
    optimized_xyz = work_dir / f"{namespace}.xtbopt.xyz"
    charges_path = work_dir / f"{namespace}.charges"
    try:
        stability = assess_stability(
            fragment,
            linkage_norm,
            optimized_xyz if optimized_xyz.is_file() else xyz_path,
            charges_path if charges_path.is_file() else None,
        )
    except Exception as error:
        result.update(
            status="stability_failed",
            error=str(error),
            compute_time_s=round(time.perf_counter() - start, 2),
        )
        return result
    result["stability"] = stability
    result["stability_score"] = stability["score"]
    # Backward-compatible alias retained for existing traces and analysis tools.
    result["s_linkage"] = stability["score"]
    result["criterion"] = _assess(result)
    result["is_hit"] = result["criterion"]["is_hit"]
    result.update(status="success", compute_time_s=round(time.perf_counter() - start, 2))
    return result


if __name__ == "__main__":
    import json

    library = load_library()
    transfer_function = load_transfer_function()
    cbm_calibration = load_cbm_calibration()
    known_hit = evaluate_candidate("N016", "L011", "hydrazone", "R001", library,
                                   transfer_function, cbm_calibration)
    print(json.dumps(known_hit, indent=2))
    assert known_hit["status"] == "success" and known_hit["is_hit"]

    guard = evaluate_candidate("N001", "L001", "beta_ketoenamine", "R001", library,
                               transfer_function, cbm_calibration)
    assert guard["status"] == "assembly_failed"
    print(f"\nN001/L001/bke: {guard['status']} (expected assembly_failed)")

    imine = evaluate_candidate("N001", "L001", "imine", "R001", library,
                               transfer_function, cbm_calibration)
    print(f"\nN001/L001/imine: status={imine['status']}, is_hit={imine.get('is_hit')}")
    assert imine["status"] == "success"
    assert not imine["criterion"]["stability_pass"]
