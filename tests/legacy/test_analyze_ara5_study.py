"""Tests for publication-study aggregation and reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analyze_ara5_study import aggregate_study, distribution, write_outputs


ROOT = Path(__file__).resolve().parent


class Ara5StudyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "study_id": "test_study",
            "status": "draft",
            "primary_endpoint": {
                "name": "distinct_hit_scaffolds_at_2",
                "definition": "Test endpoint.",
            },
            "campaign_configuration": {"oracle_attempts": 2},
            "execution": {"model": "mock"},
            "conditions": {
                "adaptive_ara5": {
                    "replicates": 1,
                    "run_ids": ["ara5_publication_mock_canary_v1"],
                }
            },
        }

    def test_distribution_is_deterministic_and_reports_preregistered_statistics(self):
        first = distribution([1, 2, 4, 5, 8], label="example", bootstrap_samples=200)
        second = distribution([1, 2, 4, 5, 8], label="example", bootstrap_samples=200)
        self.assertEqual(first, second)
        self.assertEqual(first["n"], 5)
        self.assertEqual(first["median"], 4.0)
        self.assertEqual(first["interquartile_range"], 3.0)
        self.assertEqual(len(first["bootstrap_95_percent_confidence_interval"]), 2)

    def test_aggregate_validates_and_summarizes_fixed_horizon_run(self):
        baselines = {
            "methods": {
                "scaffold_first_random": {
                    "per_seed": [
                        {"seed": 0, "hits": 1, "scaffolds": 1},
                        {"seed": 1, "hits": 0, "scaffolds": 0},
                    ]
                }
            }
        }
        result = aggregate_study(
            self.config,
            ROOT / "output" / "runs",
            baseline_results=baselines,
            bootstrap_samples=200,
        )
        self.assertTrue(result["complete"])
        condition = result["conditions"]["adaptive_ara5"]
        self.assertEqual(condition["completed_replicates"], 1)
        self.assertEqual(condition["runs"][0]["hit_scaffolds"], 1)
        self.assertEqual(condition["runs"][0]["hypothesis_updates"], 2)
        self.assertEqual(condition["summary"]["hit_scaffolds"]["values"], [1.0])
        self.assertIn("scaffold_first_random", result["lookup_baselines"])

    def test_missing_definitive_run_fails_loudly(self):
        config = {
            **self.config,
            "conditions": {
                "adaptive_ara5": {
                    "replicates": 1,
                    "run_ids": ["does_not_exist"],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "missing runs"):
            aggregate_study(
                config, ROOT / "output" / "runs", bootstrap_samples=20
            )

    def test_outputs_are_machine_and_reader_accessible(self):
        result = aggregate_study(
            self.config,
            ROOT / "output" / "runs",
            bootstrap_samples=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_outputs(result, output)
            self.assertTrue((output / "study_results.json").is_file())
            self.assertTrue((output / "campaign_replicates.csv").is_file())
            summary = (output / "study_summary.md").read_text(encoding="utf-8")
            self.assertIn("Primary endpoint", summary)
            self.assertIn("computational screening outcomes", summary)


if __name__ == "__main__":
    unittest.main()
