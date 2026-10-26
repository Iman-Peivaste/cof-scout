"""Tests for publication-study prediction and baseline infrastructure."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ara5_study_baselines import (
    _maxmin_indices,
    candidate_random_search,
    gp_ucb_search,
    scaffold_first_random_search,
    selection_metrics,
)
from ara5_loop import _append_event, _mock_result
from ara5_notebook import build_notebook
from ara5_static_portfolio import (
    StaticPortfolioEntry,
    evaluate_static_portfolio,
    select_static_portfolio,
)
from prompt_builder import _load_feasible_set
from reference_oracle import ReferenceOracle
from scripts.run_ara5_publication_study import build_command
from validate_ara5_trace import validate_events


class Ara5StudyBaselineTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            ("N001", "L001", "imine", "R001"),
            ("N001", "L001", "imine", "R002"),
            ("N002", "L002", "beta_ketoenamine", "R001"),
            ("N003", "L003", "hydrazone", "R001"),
            ("N004", "L004", "vinylene", "R001"),
            ("N005", "L005", "vinylene", "R001"),
        ]
        self.labels = np.asarray([0, 1, 1, 0, 1, 0], dtype=float)
        self.statuses = ["success", "success", "success", "build_failed", "success", "success"]
        self.features = np.asarray([
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ], dtype=np.float32)

    def test_metrics_count_attempted_failures_and_distinct_scaffolds(self):
        result = selection_metrics(
            [0, 1, 2, 3], self.candidates, self.labels, self.statuses
        )
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["scaffolds"], 2)
        self.assertEqual(result["scaffolds_tested"], 3)
        self.assertEqual(result["oracle_failures"], 1)
        self.assertAlmostEqual(result["decoration_rate"], 0.25)

    def test_scaffold_first_random_never_decorates_within_budget(self):
        outcomes = scaffold_first_random_search(
            self.candidates, self.labels, self.statuses, seeds=5, budget=4
        )
        self.assertEqual(len(outcomes), 5)
        self.assertTrue(all(row["scaffolds_tested"] == 4 for row in outcomes))
        self.assertTrue(all(row["decoration_rate"] == 0.0 for row in outcomes))

    def test_candidate_random_is_reproducible(self):
        first = candidate_random_search(
            self.candidates, self.labels, self.statuses, seeds=3, budget=4
        )
        second = candidate_random_search(
            self.candidates, self.labels, self.statuses, seeds=3, budget=4
        )
        self.assertEqual(first, second)

    def test_maxmin_is_reproducible_and_unique(self):
        first = _maxmin_indices(self.features, budget=4, seed=7)
        second = _maxmin_indices(self.features, budget=4, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_scaffold_aware_gp_excludes_decorations_after_a_hit(self):
        outcomes = gp_ucb_search(
            self.candidates, self.labels, self.statuses, self.features,
            seeds=1, budget=4, initial=2, kappa=1.0, scaffold_aware=True,
        )
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(outcomes[0]["selected_candidates"]), 4)


class Ara5StaticPortfolioTests(unittest.TestCase):
    def test_portfolio_is_locked_before_any_oracle_evaluation(self):
        entries = [
            {
                "candidate": {
                    "node_id": "N002", "linker_id": linker,
                    "linkage": "beta_ketoenamine", "r_group_id": "R001",
                },
                "purpose": "Test a distinct linker scaffold.",
                "chemical_rationale": "The linker changes conjugation at fixed node chemistry.",
                "prediction": {
                    "axis": "gap", "direction": direction,
                    "statement": statement, "reference_iteration": None,
                    "hit_probability": probability,
                    "gate_predictions": gates,
                    "limiting_gate": limiting,
                    "falsification_rule": falsification,
                },
            }
            for linker, direction, statement, probability, gates, limiting, falsification in (
                (
                    "L001", "wider", "The corrected gap will exceed the window.", 0.3,
                    {"gap": False, "cbm": True, "vbm": True, "stability": True},
                    "gap", "A corrected gap at or below 2.1 eV would falsify this prediction.",
                ),
                (
                    "L002", "pass", "The candidate will pass all four gates.", 0.7,
                    {"gap": True, "cbm": True, "vbm": True, "stability": True},
                    "none", "Failure of any gate would falsify this prediction.",
                ),
            )
        ]
        raw = json.dumps({
            "scientific_rationale": "Choose two distinct scaffolds before observing outcomes.",
            "portfolio_entries": entries,
        })
        calls = []

        def request_batch(context):
            calls.append(context)
            return raw, {"prompt_tokens": 10, "completion_tokens": 10}

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            trace_path = run_dir / "trace.jsonl"
            events = []
            _append_event(trace_path, events, {
                "event_type": "campaign_plan",
                "plan": {
                    "objective": "Static portfolio control.",
                    "landscape_assessment": "Outcome blind.",
                    "strategy": "Lock all candidates before evaluation.",
                    "hypotheses": [{
                        "hypothesis_id": "H_STATIC", "statement": "Static selection works.",
                        "rationale": "Chemical priors.", "confidence": "medium",
                        "discriminating_test": "Evaluate the locked portfolio.",
                    }],
                    "decision_principles": ["Prefer distinct scaffolds."],
                    "saturation_criteria": ["Fixed horizon."],
                },
            })
            portfolio = select_static_portfolio(
                request_batch, _load_feasible_set(), trace_path, events,
                budget=2, batch_size=2, max_decoration_fraction=0.0,
            )
            self.assertEqual(len(portfolio), 2)
            self.assertEqual(
                [event["event_type"] for event in events],
                ["campaign_plan", "static_portfolio_batch", "static_portfolio_locked"],
            )
            evaluate_static_portfolio(
                portfolio, _mock_result, run_dir, trace_path, events
            )
            notebook = build_notebook(events)
            summary = validate_events(events, notebook)
            self.assertEqual(summary["experiments"], 2)
            self.assertEqual(len(calls), 1)
            experiments = [event for event in events if event["event_type"] == "experiment"]
            self.assertEqual(experiments[0]["prediction_adjudication"]["gate_accuracy"], 1.0)

    def test_static_evaluation_resumes_from_locked_prefix(self):
        prediction = {
            "axis": "overall", "direction": "pass",
            "statement": "The candidate will pass.", "reference_iteration": None,
            "hit_probability": 0.7,
            "gate_predictions": {
                "gap": True, "cbm": True, "vbm": True, "stability": True,
            },
            "limiting_gate": "none",
            "falsification_rule": "Failure of any gate falsifies the prediction.",
        }
        entries = [
            StaticPortfolioEntry.from_mapping({
                "candidate": {
                    "node_id": "N002", "linker_id": linker,
                    "linkage": "beta_ketoenamine", "r_group_id": "R001",
                },
                "purpose": "Resume test.",
                "chemical_rationale": "Distinct linker scaffold.",
                "prediction": prediction,
            })
            for linker in ("L001", "L002")
        ]
        calls = 0

        def interrupted_evaluator(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("simulated interruption")
            return _mock_result(*args)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trace_path = run_dir / "trace.jsonl"
            events = []
            _append_event(trace_path, events, {
                "event_type": "campaign_plan",
                "plan": {
                    "objective": "Static resume test.",
                    "landscape_assessment": "Outcome blind.",
                    "strategy": "Keep locked order.",
                    "hypotheses": [],
                    "decision_principles": [],
                    "saturation_criteria": [],
                },
            })
            _append_event(trace_path, events, {
                "event_type": "static_portfolio_locked",
                "candidate_count": 2,
                "portfolio_sha256": "test",
                "portfolio": [asdict(entry) for entry in entries],
            })
            with self.assertRaises(KeyboardInterrupt):
                evaluate_static_portfolio(
                    entries, interrupted_evaluator, run_dir, trace_path, events
                )
            self.assertEqual(
                sum(event["event_type"] == "experiment" for event in events), 1
            )
            evaluate_static_portfolio(
                entries, _mock_result, run_dir, trace_path, events
            )
            self.assertEqual(validate_events(events)["experiments"], 2)


class ReferenceOracleTests(unittest.TestCase):
    def test_hash_validated_lookup_returns_independent_result_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feasible = root / "feasible.csv"
            feasible.write_text(
                "node_id,linker_id,linkage_norm,r_group_id,is_feasible\n"
                "N002,L001,beta_ketoenamine,R001,true\n",
                encoding="utf-8",
            )
            landscape = root / "landscape.jsonl"
            result = _mock_result(
                "N002", "L001", "beta_ketoenamine", "R001", root
            )
            landscape.write_text(json.dumps({
                "node_id": "N002", "linker_id": "L001",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
                "result": result,
            }) + "\n", encoding="utf-8")
            digest = hashlib.sha256(landscape.read_bytes()).hexdigest()
            manifest = root / "reference_manifest.json"
            manifest.write_text(json.dumps({
                "reference_schema": "test-reference-v1",
                "records": 1,
                "stability_model": "mock_fixed",
                "output_sha256": digest,
            }), encoding="utf-8")
            oracle = ReferenceOracle(landscape, feasible, manifest)
            first = oracle.evaluate(
                "N002", "L001", "beta_ketoenamine", "R001", root
            )
            first["criterion"]["is_hit"] = "changed"
            second = oracle.evaluate(
                "N002", "L001", "beta_ketoenamine", "R001", root
            )
            self.assertIs(second["criterion"]["is_hit"], False)
            self.assertEqual(
                oracle.provenance["reference_landscape_sha256"], digest
            )


class PublicationLauncherTests(unittest.TestCase):
    def test_launcher_builds_fixed_adaptive_and_static_commands(self):
        config = json.loads(
            (Path(__file__).resolve().parent / "data" / "ara5_publication_study_v1.json")
            .read_text(encoding="utf-8")
        )
        adaptive_id, adaptive = build_command(
            config, "adaptive_ara5", 1, resume=False
        )
        static_id, static = build_command(
            config, "one_shot_llm_portfolio", 1, resume=True
        )
        self.assertEqual(adaptive_id, "ara5_mcap_full_rep01")
        self.assertIn("--publication-mode", adaptive)
        self.assertIn("--reference-landscape", adaptive)
        self.assertEqual(static_id, "ara5_mcap_static_rep01")
        self.assertIn("--batch-size", static)
        self.assertEqual(static[-1], "--resume")


if __name__ == "__main__":
    unittest.main()
