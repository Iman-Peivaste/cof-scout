"""Tests for the mechanism-constrained aqueous persistence score."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from assembler import assemble_fragment, load_library
from oracle import _assess
from stability import MECHANISM_ENVELOPES, STABILITY_MODEL_VERSION, assess_stability


class StabilityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = load_library()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.candidates = {
            "imine": ("N001", "L001"),
            "hydrazone": ("N016", "L011"),
            "beta_ketoenamine": ("N002", "L001"),
            "vinylene": ("N003", "L018"),
        }
        cls.artifacts = {}
        for linkage, (node, linker) in cls.candidates.items():
            smiles = assemble_fragment(
                node, linker, linkage, "R001", cls.library
            )
            if smiles is None:
                raise AssertionError(f"Test candidate did not assemble: {linkage}")
            molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(molecule, params) != 0:
                raise AssertionError(f"Test geometry failed to embed: {linkage}")
            xyz_path = cls.root / f"{linkage}.xyz"
            xyz_path.write_text(Chem.MolToXYZBlock(molecule), encoding="utf-8")
            charges_path = cls.root / f"{linkage}.charges"
            charges_path.write_text(
                "\n".join("0.15" for _ in range(molecule.GetNumAtoms())) + "\n",
                encoding="utf-8",
            )
            cls.artifacts[linkage] = (smiles, xyz_path, charges_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_linkages_stay_inside_mechanistic_envelopes(self):
        for linkage, (smiles, xyz_path, charges_path) in self.artifacts.items():
            result = assess_stability(smiles, linkage, xyz_path, charges_path)
            lower, upper = MECHANISM_ENVELOPES[linkage]
            self.assertGreaterEqual(result["score"], lower)
            self.assertLessEqual(result["score"], upper)
            self.assertEqual(result["model"], STABILITY_MODEL_VERSION)
            self.assertEqual(result["linkage_sites"], 2)

    def test_current_stability_eligibility_is_preserved(self):
        for linkage, (smiles, xyz_path, charges_path) in self.artifacts.items():
            result = assess_stability(smiles, linkage, xyz_path, charges_path)
            self.assertEqual(result["pass"], linkage != "imine")

    def test_charge_changes_score_without_crossing_imine_envelope(self):
        smiles, xyz_path, charges_path = self.artifacts["imine"]
        atom_count = len(charges_path.read_text(encoding="utf-8").split())
        low_charge = self.root / "imine_low.charges"
        high_charge = self.root / "imine_high.charges"
        low_charge.write_text("\n".join(["0.13"] * atom_count), encoding="utf-8")
        high_charge.write_text("\n".join(["0.18"] * atom_count), encoding="utf-8")
        less_electrophilic = assess_stability(smiles, "imine", xyz_path, low_charge)
        more_electrophilic = assess_stability(smiles, "imine", xyz_path, high_charge)
        self.assertGreater(less_electrophilic["score"], more_electrophilic["score"])
        self.assertFalse(less_electrophilic["pass"])
        self.assertFalse(more_electrophilic["pass"])

    def test_missing_xtb_charges_uses_recorded_fallback(self):
        smiles, xyz_path, _ = self.artifacts["hydrazone"]
        result = assess_stability(smiles, "hydrazone", xyz_path, None)
        self.assertEqual(
            result["diagnostics"]["charge_source"], "rdkit_gasteiger_fallback"
        )
        self.assertTrue(result["pass"])

    def test_oracle_assessment_reports_electronic_pass_separately(self):
        result = _assess({
            "gap_corrected_eV": 2.0,
            "cbm_vs_nhe": -0.2,
            "vbm_vs_nhe": 1.8,
            "s_linkage": 0.60,
        })
        self.assertTrue(result["electronic_pass"])
        self.assertFalse(result["stability_pass"])
        self.assertFalse(result["is_hit"])


if __name__ == "__main__":
    unittest.main()
