"""Mechanism-constrained candidate-specific stability scoring for COF fragments.

MCAP is an expert-defined screening index, not a fitted hydrolysis probability.
Linkage-specific envelopes encode the final bond mechanism; deterministic local
descriptors rank candidates within those envelopes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, rdMolTransforms


STABILITY_MODEL_VERSION = "mcap-v1"
STABILITY_THRESHOLD = 0.70
SIGMOID_STEEPNESS = 4.0
SASA_PROBE_RADIUS_ANGSTROM = 1.4
LOCAL_ENVIRONMENT_RADIUS_ANGSTROM = 5.0
RIGIDITY_TORSION_SCALE_DEGREES = 30.0

MECHANISM_ENVELOPES: dict[str, tuple[float, float]] = {
    "imine": (0.35, 0.65),
    "hydrazone": (0.70, 0.80),
    "beta_ketoenamine": (0.75, 0.85),
    "vinylene": (0.90, 1.00),
}

MECHANISM_NAMES = {
    "imine": "reversible_aryl_imine",
    "hydrazone": "acyl_hydrazone_conjugation",
    "beta_ketoenamine": "irreversible_keto_enamine_tautomerization",
    "vinylene": "hydrolysis_resistant_carbon_carbon_linkage",
}

# Frozen normalization intervals for the maximum linkage-carbon GFN1-xTB charge.
# They approximate the 5th--95th percentile range in the current feasible fragment
# inventory and contain no outcome labels. Lower electrophilicity is favorable.
CHARGE_INTERVALS = {
    "imine": (0.13, 0.18),
    "hydrazone": (0.08, 0.17),
    "beta_ketoenamine": (0.35, 0.39),
}

LOCAL_PROTECTION_WEIGHTS = {
    "shielding": 0.40,
    "local_hydrophobicity": 0.25,
    "electronic_deactivation": 0.20,
    "rigidity": 0.15,
}

_LINKAGE_PATTERNS = {
    "imine": Chem.MolFromSmarts("[c:1]-[CH1:2]=[N:3]-[c:4]"),
    "hydrazone": Chem.MolFromSmarts(
        "[c:1]-[CH1:2]=[N:3]-[NH1:4]-[C:5](=[O:6])"
    ),
    "vinylene": Chem.MolFromSmarts("[c:1]-[CH1:2]=[CH1:3]-[c:4]"),
    "beta_ketoenamine": Chem.MolFromSmarts(
        "[C:1](=[O:2])-[C:3]=[C:4]-[NH1:5]-[c:6]"
    ),
}

# Positions in each SMARTS match. Geometry and charge files preserve RDKit atom
# order from the explicit-H molecule written by geometry.build_geometry().
_REACTIVE_POSITIONS = {
    "imine": (1, 2),
    "hydrazone": (1, 2),
    "vinylene": (1, 2),
    "beta_ketoenamine": (0, 1, 2, 3, 4),
}
_TORSION_POSITIONS = {
    "imine": (0, 1, 2, 3),
    "hydrazone": (0, 1, 2, 3),
    "vinylene": (0, 1, 2, 3),
    "beta_ketoenamine": (0, 2, 3, 4),
}
_CHARGE_POSITION = {
    "imine": 1,
    "hydrazone": 1,
    "beta_ketoenamine": 0,
}
_APOLAR_ATOMIC_NUMBERS = frozenset({6, 9, 17, 35, 53})


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalized_sigmoid(protection: float) -> float:
    """Map [0, 1] exactly onto [0, 1], retaining midpoint 0.5."""
    protection = _clip(protection)

    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    lower = sigmoid(-SIGMOID_STEEPNESS / 2.0)
    upper = sigmoid(SIGMOID_STEEPNESS / 2.0)
    current = sigmoid(SIGMOID_STEEPNESS * (protection - 0.5))
    return (current - lower) / (upper - lower)


def _read_xyz_coordinates(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is incomplete: {path}")
    atom_count = int(lines[0].strip())
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"XYZ atom count mismatch for {path}: header={atom_count}, rows={len(atom_lines)}"
        )
    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ row in {path}: {line!r}")
        symbols.append(fields[0])
        coordinates.append(tuple(float(value) for value in fields[1:4]))
    return symbols, coordinates


def _molecule_with_coordinates(fragment_smiles: str, xyz_path: Path) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(fragment_smiles)
    if molecule is None:
        raise ValueError("MCAP received an invalid fragment SMILES")
    molecule = Chem.AddHs(molecule)
    symbols, coordinates = _read_xyz_coordinates(xyz_path)
    expected = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    if symbols != expected:
        raise ValueError(
            "XYZ atom order does not match the assembled explicit-H molecule"
        )
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(index, coordinate)
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def _linkage_matches(molecule: Chem.Mol, linkage: str) -> tuple[tuple[int, ...], ...]:
    pattern = _LINKAGE_PATTERNS.get(linkage)
    if pattern is None:
        raise ValueError(f"MCAP has no mechanistic definition for linkage {linkage!r}")
    matches = molecule.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        raise ValueError(f"No {linkage} linkage motif was found in the assembled fragment")
    return matches


def _atom_sasa(molecule: Chem.Mol) -> list[float]:
    periodic_table = Chem.GetPeriodicTable()
    radii = [periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in molecule.GetAtoms()]
    rdFreeSASA.CalcSASA(molecule, radii)
    return [float(atom.GetProp("SASA")) for atom in molecule.GetAtoms()]


def _site_shielding(
    molecule: Chem.Mol,
    sasa: list[float],
    reactive_indices: tuple[int, ...],
) -> float:
    periodic_table = Chem.GetPeriodicTable()
    exposed_reference = sum(
        4.0 * math.pi * (
            periodic_table.GetRvdw(molecule.GetAtomWithIdx(index).GetAtomicNum())
            + SASA_PROBE_RADIUS_ANGSTROM
        ) ** 2
        for index in reactive_indices
    )
    observed = sum(sasa[index] for index in reactive_indices)
    return _clip(1.0 - observed / exposed_reference)


def _is_apolar_atom(molecule: Chem.Mol, atom_index: int) -> bool:
    atom = molecule.GetAtomWithIdx(atom_index)
    if atom.GetAtomicNum() == 1:
        neighbors = atom.GetNeighbors()
        return bool(neighbors and neighbors[0].GetAtomicNum() in _APOLAR_ATOMIC_NUMBERS)
    return atom.GetAtomicNum() in _APOLAR_ATOMIC_NUMBERS


def _site_local_hydrophobicity(
    molecule: Chem.Mol,
    sasa: list[float],
    reactive_indices: tuple[int, ...],
) -> float:
    conformer = molecule.GetConformer()
    local_indices: list[int] = []
    for atom_index in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_index)
        if any(
            position.Distance(conformer.GetAtomPosition(reactive_index))
            <= LOCAL_ENVIRONMENT_RADIUS_ANGSTROM
            for reactive_index in reactive_indices
        ):
            local_indices.append(atom_index)
    total = sum(sasa[index] for index in local_indices)
    if total <= 0.0:
        return 0.0
    apolar = sum(
        sasa[index] for index in local_indices
        if _is_apolar_atom(molecule, index)
    )
    return _clip(apolar / total)


def _site_rigidity(
    molecule: Chem.Mol,
    match: tuple[int, ...],
    linkage: str,
) -> tuple[float, float]:
    indices = tuple(match[position] for position in _TORSION_POSITIONS[linkage])
    angle = rdMolTransforms.GetDihedralDeg(molecule.GetConformer(), *indices)
    deviation = min(abs(angle), abs(180.0 - abs(angle)))
    rigidity = math.exp(-((deviation / RIGIDITY_TORSION_SCALE_DEGREES) ** 2))
    return _clip(rigidity), round(deviation, 6)


def _read_xtb_charges(path: Path, atom_count: int) -> list[float]:
    charges = [float(value) for value in path.read_text(encoding="utf-8").split()]
    if len(charges) != atom_count:
        raise ValueError(
            f"xTB charge count mismatch for {path}: expected {atom_count}, got {len(charges)}"
        )
    return charges


def _gasteiger_charges(molecule: Chem.Mol) -> list[float]:
    fallback = Chem.Mol(molecule)
    AllChem.ComputeGasteigerCharges(fallback, nIter=12, throwOnParamFailure=True)
    charges = [float(atom.GetProp("_GasteigerCharge")) for atom in fallback.GetAtoms()]
    if not all(math.isfinite(value) for value in charges):
        raise ValueError("Gasteiger fallback produced a non-finite charge")
    return charges


def _electronic_deactivation(
    charges: list[float],
    matches: tuple[tuple[int, ...], ...],
    linkage: str,
) -> tuple[float, float | None]:
    if linkage == "vinylene":
        return 1.0, None
    position = _CHARGE_POSITION[linkage]
    weakest_site_charge = max(charges[match[position]] for match in matches)
    q_min, q_max = CHARGE_INTERVALS[linkage]
    score = _clip((q_max - weakest_site_charge) / (q_max - q_min))
    return score, round(weakest_site_charge, 8)


def assess_stability(
    fragment_smiles: str,
    linkage: str,
    xyz_path: str | Path,
    charges_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate MCAP from an assembled fragment and its optimized geometry.

    The two linkage sites represented in the Node--Linker--Node fragment are
    aggregated by their least favorable local descriptor value. This implements a
    conservative weakest-link interpretation for network persistence.
    """
    linkage_norm = linkage.strip().lower().replace("-", "_").replace(" ", "_")
    if linkage_norm not in MECHANISM_ENVELOPES:
        raise ValueError(f"Unsupported MCAP linkage: {linkage_norm!r}")
    molecule = _molecule_with_coordinates(fragment_smiles, Path(xyz_path))
    matches = _linkage_matches(molecule, linkage_norm)
    sasa = _atom_sasa(molecule)

    shielding_values: list[float] = []
    hydrophobicity_values: list[float] = []
    rigidity_values: list[float] = []
    torsion_deviations: list[float] = []
    for match in matches:
        reactive = tuple(match[position] for position in _REACTIVE_POSITIONS[linkage_norm])
        shielding_values.append(_site_shielding(molecule, sasa, reactive))
        hydrophobicity_values.append(
            _site_local_hydrophobicity(molecule, sasa, reactive)
        )
        rigidity, deviation = _site_rigidity(molecule, match, linkage_norm)
        rigidity_values.append(rigidity)
        torsion_deviations.append(deviation)

    charge_source = "not_required_for_vinylene"
    charges: list[float]
    charge_path = Path(charges_path) if charges_path is not None else None
    if linkage_norm == "vinylene":
        charges = []
    elif charge_path is not None and charge_path.is_file():
        charges = _read_xtb_charges(charge_path, molecule.GetNumAtoms())
        charge_source = "gfn1_xtb_mulliken"
    else:
        charges = _gasteiger_charges(molecule)
        charge_source = "rdkit_gasteiger_fallback"

    electronic_deactivation, weakest_site_charge = _electronic_deactivation(
        charges, matches, linkage_norm
    )
    components = {
        "shielding": min(shielding_values),
        "local_hydrophobicity": min(hydrophobicity_values),
        "electronic_deactivation": electronic_deactivation,
        "rigidity": min(rigidity_values),
    }
    local_protection = sum(
        LOCAL_PROTECTION_WEIGHTS[name] * components[name]
        for name in LOCAL_PROTECTION_WEIGHTS
    )
    lower, upper = MECHANISM_ENVELOPES[linkage_norm]
    score = lower + (upper - lower) * _normalized_sigmoid(local_protection)

    rounded_components = {
        name: round(value, 6) for name, value in components.items()
    }
    rounded_components["local_protection"] = round(local_protection, 6)
    return {
        "model": STABILITY_MODEL_VERSION,
        "score": round(score, 6),
        "threshold": STABILITY_THRESHOLD,
        "pass": score >= STABILITY_THRESHOLD,
        "mechanistic_class": linkage_norm,
        "mechanism": MECHANISM_NAMES[linkage_norm],
        "envelope": {"lower": lower, "upper": upper},
        "evidence_type": "expert_mechanistic_prior_plus_computed_local_descriptors",
        "candidate_specific_prediction": True,
        "aggregation": "weakest_linkage_site",
        "linkage_sites": len(matches),
        "components": rounded_components,
        "diagnostics": {
            "charge_source": charge_source,
            "weakest_site_linkage_carbon_charge": weakest_site_charge,
            "torsion_deviation_degrees": torsion_deviations,
            "sasa_probe_radius_angstrom": SASA_PROBE_RADIUS_ANGSTROM,
            "local_environment_radius_angstrom": LOCAL_ENVIRONMENT_RADIUS_ANGSTROM,
        },
    }
