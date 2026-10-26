"""Canonical COF fragment assembler using RDKit reaction SMARTS.

The publication workflow keeps the evaluated chemistry in this repository rather
than importing a mutable implementation from a neighbouring checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

from cof_scout.paths import data_path


_DEFAULT_LIBRARY = data_path("building_blocks_library_v2.json")

_RXN_IMINE = AllChem.ReactionFromSmarts(
    "[CH1:1](=[O:2])[c:3].[NH2:4][c:5]>>[CH1:1](=[N:4][c:5])[c:3]"
)
_RXN_HYDRAZONE = AllChem.ReactionFromSmarts(
    "[CH1:1](=[O:2])[c:3].[NH2:4][NH:5][C:6]=[O:7]>>"
    "[CH1:1](=[N:4][NH:5][C:6]=[O:7])[c:3]"
)
_RXN_VINYLENE = AllChem.ReactionFromSmarts(
    "[CH1:1](=[O:2])[c:3].[CH2:4]=[CH:5][c:6]>>"
    "[CH1:1](=[CH:5][c:6])[c:3]"
)
_RXN_BKE_TAUTO = AllChem.ReactionFromSmarts(
    "[c:1]([OH1:2])[c:3][CH1:4]=[N:5]>>"
    "[C:1](=[O:2])[C:3]=[C:4][NH:5]"
)
_PAT_ENAMINE = Chem.MolFromSmarts("[#6]=[#6]-[#7H1]")


def load_library(path: str | Path | None = None) -> dict[str, Any]:
    """Load the building-block library JSON."""
    target = Path(path) if path is not None else _DEFAULT_LIBRARY
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def _get_entry(library: dict, collection: str, entry_id: str) -> dict | None:
    return next(
        (
            entry for entry in library.get(collection, [])
            if str(entry.get("id", "")).strip() == entry_id
        ),
        None,
    )


def _first_product(reaction, reactants):
    products = reaction.RunReactants(reactants)
    if not products:
        return None
    for product in products[0]:
        try:
            Chem.SanitizeMol(product)
            return Chem.MolFromSmiles(Chem.MolToSmiles(product))
        except Exception:
            continue
    return None


def _run_twice(reaction, node_smiles: str, linker_smiles: str):
    """Condense one node arm against the linker, then condense the second arm."""
    node_a = Chem.MolFromSmiles(node_smiles)
    node_b = Chem.MolFromSmiles(node_smiles)
    linker = Chem.MolFromSmiles(linker_smiles)
    if None in (node_a, node_b, linker):
        return None
    half = _first_product(reaction, (node_a, linker))
    if half is None:
        return None
    return _first_product(reaction, (node_b, half)) or _first_product(
        reaction, (half, node_b)
    )


def _assemble_smarts(node_smiles: str, linker_smiles: str, linkage: str):
    if linkage == "imine":
        return _run_twice(_RXN_IMINE, node_smiles, linker_smiles)
    if linkage == "hydrazone":
        return _run_twice(_RXN_HYDRAZONE, node_smiles, linker_smiles)
    if linkage == "vinylene":
        return _run_twice(_RXN_VINYLENE, node_smiles, linker_smiles)
    if linkage == "beta_ketoenamine":
        full = _run_twice(_RXN_IMINE, node_smiles, linker_smiles)
        if full is None:
            return None
        for _ in range(6):
            next_product = _first_product(_RXN_BKE_TAUTO, (full,))
            if next_product is None:
                break
            full = next_product
        if not full.GetSubstructMatches(_PAT_ENAMINE):
            return None
        return full
    return None


def _select_r_target(molecule) -> int | None:
    matches = molecule.GetSubstructMatches(Chem.MolFromSmarts("[cH]"))
    if not matches:
        return None
    ranks = Chem.CanonicalRankAtoms(molecule, breakTies=True)
    return min((match[0] for match in matches), key=lambda index: (ranks[index], index))


def _attach_r_group(fragment_smiles: str, r_group_smiles: str) -> str | None:
    if r_group_smiles in ("[H]", "H"):
        return fragment_smiles
    molecule = Chem.MolFromSmiles(fragment_smiles)
    substituent = Chem.MolFromSmiles(r_group_smiles)
    if molecule is None or substituent is None:
        return None
    target = _select_r_target(molecule)
    if target is None:
        return fragment_smiles
    combined = Chem.RWMol(Chem.CombineMols(molecule, substituent))
    combined.AddBond(target, molecule.GetNumAtoms(), Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
    except Exception:
        return None


def assemble_fragment(
    node_id: str,
    linker_id: str,
    linkage: str,
    r_group_id: str = "R001",
    library: dict | None = None,
) -> str | None:
    """Return canonical fragment SMILES, or ``None`` if not constructible."""
    if library is None:
        library = load_library()
    node = _get_entry(library, "nodes", node_id)
    linker = _get_entry(library, "linkers", linker_id)
    r_group = _get_entry(library, "r_groups", r_group_id)
    if node is None or linker is None:
        return None
    if r_group is None:
        r_group = {"smiles": "[H]"}
    if not (node.get("compatible_linkages") or []):
        return None
    if not (linker.get("compatible_linkages") or []):
        return None
    linkage_normalized = linkage.strip().lower().replace("-", "_").replace(" ", "_")
    fragment = _assemble_smarts(
        node.get("smiles", ""), linker.get("smiles", ""), linkage_normalized
    )
    if fragment is None:
        return None
    result = _attach_r_group(
        Chem.MolToSmiles(fragment), r_group.get("smiles", "[H]")
    )
    if result is None:
        return None
    return Chem.MolToSmiles(Chem.MolFromSmiles(result))


__all__ = ["assemble_fragment", "load_library"]


if __name__ == "__main__":
    active_library = load_library()
    examples = (
        ("N002", "L001", "beta_ketoenamine", "R001"),
        ("N016", "L011", "hydrazone", "R001"),
    )
    for example in examples:
        assembled = assemble_fragment(*example, library=active_library)
        assert assembled is not None
        print("/".join(example), assembled)
