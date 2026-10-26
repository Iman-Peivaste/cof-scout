"""xTB geometry optimization and vertical IP/EA calculation for COF fragments."""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_xtb() -> str:
    """Locate xTB via XTB_PATH, the active Conda environment, or PATH."""
    env_path = os.environ.get("XTB_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(f"XTB_PATH={env_path!r} is not a file")
    sibling = Path(sys.executable).resolve().parent / "xtb"
    if sibling.is_file():
        return str(sibling)
    return "xtb"


def _safe_namespace(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "cof"


def _run_xtb(cmd: list[str], cwd: Path, log_path: Path) -> bool:
    """Run xTB and redirect both output streams to its log file."""
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd, stdout=handle, stderr=subprocess.STDOUT, text=True,
                cwd=str(cwd), check=False,
            )
        return result.returncode == 0
    except OSError as error:
        log_path.write_text(f"Could not execute xTB: {error}\n", encoding="utf-8")
        return False


def _parse_vipea_log(log_path: Path) -> tuple[float | None, float | None]:
    """Extract the delta-SCC vertical IP and EA values (eV) from an xTB log."""
    ip_ev = ea_ev = None
    pattern = re.compile(r"delta\s+SCC\s+(IP|EA)\s+\(eV\):\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            if match.group(1) == "IP":
                ip_ev = float(match.group(2))
            else:
                ea_ev = float(match.group(2))
    return ip_ev, ea_ev


def run_xtb_opt(xyz_path: Path, work_dir: Path, namespace: str,
                level: str = "tight") -> Path | None:
    """Run GFN1-xTB geometry optimization and return its optimized XYZ path."""
    log = work_dir / f"{namespace}_opt.log"
    command = [_find_xtb(), xyz_path.name, "--gfn", "1", "--opt", level,
               "--namespace", namespace]
    start = time.perf_counter()
    ok = _run_xtb(command, work_dir, log)
    elapsed = time.perf_counter() - start
    optimized = work_dir / f"{namespace}.xtbopt.xyz"
    if not ok or not optimized.is_file():
        print(f"  [xTB opt FAILED] check {log} ({elapsed:.1f}s)", file=sys.stderr)
        return None
    print(f"  xTB --opt {level}: {elapsed:.1f}s -> {optimized.name}")
    return optimized


def run_xtb_vipea(xyz_path: Path, work_dir: Path, namespace: str) -> tuple[float | None, float | None, float | None]:
    """Run GFN1-xTB --vipea and return ``(IP, EA, IP - EA)`` in eV."""
    log = work_dir / f"{namespace}_vipea.log"
    # Avoid treating a charge file from an earlier run of the same namespace as
    # fresh MCAP evidence if the current xTB version does not recreate it.
    (work_dir / f"{namespace}.charges").unlink(missing_ok=True)
    command = [_find_xtb(), xyz_path.name, "--gfn", "1", "--vipea", "--namespace", namespace]
    start = time.perf_counter()
    ok = _run_xtb(command, work_dir, log)
    elapsed = time.perf_counter() - start
    if not ok:
        print(f"  [xTB vipea FAILED] check {log} ({elapsed:.1f}s)", file=sys.stderr)
        return None, None, None
    ip_ev, ea_ev = _parse_vipea_log(log)
    if ip_ev is None or ea_ev is None:
        print(f"  [xTB vipea parse FAILED] IP={ip_ev} EA={ea_ev}", file=sys.stderr)
        return None, None, None
    gap = ip_ev - ea_ev
    print(f"  xTB --vipea: IP={ip_ev:.3f} eV EA={ea_ev:.3f} eV gap={gap:.3f} eV ({elapsed:.1f}s)")
    return ip_ev, ea_ev, gap


def evaluate_xtb(xyz_path: Path | str, work_dir: Path | str,
                 namespace: str | None = None) -> tuple[float | None, float | None, float | None]:
    """Execute the full GFN1-xTB ``--opt tight`` then ``--vipea`` workflow."""
    xyz_path, work_dir = Path(xyz_path), Path(work_dir)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"XYZ input does not exist: {xyz_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    namespace = _safe_namespace(namespace or xyz_path.stem)
    target_xyz = work_dir / xyz_path.name
    if target_xyz.resolve() != xyz_path.resolve():
        shutil.copy2(xyz_path, target_xyz)
    optimized = run_xtb_opt(target_xyz, work_dir, namespace)
    return (None, None, None) if optimized is None else run_xtb_vipea(optimized, work_dir, namespace)


if __name__ == "__main__":
    from cof_scout.inventory.assembly import assemble_fragment, load_library
    from geometry import build_geometry

    library = load_library()
    work_dir = Path("output/xtb_test")
    fragment = assemble_fragment("N016", "L011", "hydrazone", "R001", library)
    assert fragment is not None
    xyz_path = work_dir / "N016_L011_hydrazone.xyz"
    build_geometry(fragment, xyz_path)
    print("=== N016 / L011 / hydrazone / R001 ===")
    ip, ea, gap = evaluate_xtb(xyz_path, work_dir, "N016_L011_hyd_R001")
    if gap is not None:
        print(f"\nRaw xTB fundamental gap: {gap:.4f} eV")
    else:
        print("xTB failed — check output/xtb_test/N016_L011_hyd_R001_opt.log")
