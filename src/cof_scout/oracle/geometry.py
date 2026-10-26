"""3D geometry builder for COF fragments.

Pipeline: SMILES -> explicit H atoms -> ETKDGv3 conformers -> MMFF94s -> XYZ.
The resulting geometry is an initial structure for a later xTB optimization.
"""

from pathlib import Path
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem


def build_geometry(smiles: str, out_path: str | Path) -> int:
    """Embed, MMFF-optimize, and write an XYZ structure for ``smiles``.

    Twenty ETKDGv3 conformers are generated deterministically and the lowest
    MMFF94s-energy converged conformer is written. Returns atom count with H.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=20, params=params))

    if conf_ids:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise RuntimeError("MMFF94s has no parameters for this fragment")
        # Each entry is (not_converged_flag, energy_in_kcal_per_mol).
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, maxIters=2000, mmffVariant="MMFF94s"
        )
        converged = [index for index, (flag, _) in enumerate(results) if flag == 0]
        candidates = converged or list(range(len(results)))
        best = min(candidates, key=lambda index: results[index][1])
        xyz_block = Chem.MolToXYZBlock(mol, confId=conf_ids[best])
    else:
        warnings.warn(
            f"ETKDGv3 returned no conformers for {smiles[:60]!r}; using one conformer.",
            RuntimeWarning,
            stacklevel=2,
        )
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("ETKDGv3 failed to embed a 3D conformer")
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise RuntimeError("MMFF94s has no parameters for this fragment")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000, mmffVariant="MMFF94s")
        xyz_block = Chem.MolToXYZBlock(mol)

    if not xyz_block or not xyz_block.strip():
        raise RuntimeError(f"XYZ generation failed for SMILES: {smiles[:60]}")
    out_path.write_text(xyz_block, encoding="utf-8")
    return mol.GetNumAtoms()


if __name__ == "__main__":
    from cof_scout.inventory.assembly import assemble_fragment, load_library

    library = load_library()
    out_dir = Path("output/geometry_test")

    frag = assemble_fragment("N016", "L011", "hydrazone", "R001", library)
    assert frag is not None, "N016/L011/hydrazone should assemble"
    path = out_dir / "N016_L011_hydrazone.xyz"
    n_atoms = build_geometry(frag, path)
    print(f"N016/L011/hydrazone: {n_atoms} atoms")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    data_lines = [line for line in lines[2:] if line.strip()]
    assert int(lines[0]) == n_atoms == len(data_lines), "Atom count mismatch"
    print(f"  XYZ header says {lines[0]} atoms — matches")

    frag2 = assemble_fragment("N002", "L001", "beta_ketoenamine", "R001", library)
    assert frag2 is not None, "N002/L001/beta-ketoenamine should assemble"
    n_atoms_2 = build_geometry(frag2, out_dir / "N002_L001_bke.xyz")
    print(f"N002/L001/beta_ketoenamine: {n_atoms_2} atoms")

    copy_path = out_dir / "N016_L011_hydrazone_copy.xyz"
    build_geometry(frag, copy_path)
    assert path.read_text(encoding="utf-8") == copy_path.read_text(encoding="utf-8")
    print("Determinism check: PASSED")
