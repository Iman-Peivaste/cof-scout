"""Tests for the read-only COF-SCOUT live dashboard data layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ara5_dashboard import (
    campaign_metrics,
    candidate_id,
    event_rows,
    load_trace,
    transcript_entries,
    transcript_markdown,
)


def _experiment(iteration: int, linker: str, hit: bool,
                margin: float | None) -> dict:
    return {
        "event_index": iteration + 1,
        "event_type": "experiment",
        "iteration": iteration,
        "timestamp": f"2026-08-16T00:00:{iteration:02d}+00:00",
        "action": {"experiment": {
            "node_id": "N002", "linker_id": linker,
            "linkage": "beta_ketoenamine", "r_group_id": "R001",
        }},
        "result": {
            "status": "success",
            "criterion": {"is_hit": hit, "vbm_margin": margin},
        },
        "cost": {
            "action": {"prompt_tokens": 10, "completion_tokens": 2},
            "reflection": {"prompt_tokens": 5, "completion_tokens": 3},
        },
    }


class Ara5DashboardTests(unittest.TestCase):
    def test_load_trace_tolerates_partial_final_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                json.dumps({"event_index": 1, "event_type": "campaign_plan"})
                + "\n{\"event_index\":2",
                encoding="utf-8",
            )
            events, malformed = load_trace(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(malformed, 1)

    def test_campaign_metrics_count_scaffolds_usage_and_rejections(self):
        events = [
            {"event_index": 1, "event_type": "campaign_plan",
             "cost": {"prompt_tokens": 20, "completion_tokens": 4}},
            _experiment(1, "L001", False, -0.02),
            {"event_index": 3, "event_type": "action_rejected",
             "rejection": "duplicate", "cost": {
                 "prompt_tokens": 7, "completion_tokens": 1,
             }},
            _experiment(2, "L002", True, 0.30),
        ]
        metrics = campaign_metrics(events, {"max_experiments": 50})
        self.assertEqual(metrics["experiments"], 2)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["robust_hit_scaffolds"], 1)
        self.assertEqual(metrics["scaffolds"], 2)
        self.assertEqual(metrics["rejections"], 1)
        self.assertEqual(metrics["usage"]["calls"], 6)
        self.assertEqual(metrics["usage"]["prompt_tokens"], 57)

    def test_candidate_and_event_rows_are_human_readable(self):
        event = _experiment(1, "L002", True, 0.10)
        self.assertEqual(
            candidate_id(event), "N002/L002/beta_ketoenamine/R001"
        )
        row = event_rows([event])[0]
        self.assertEqual(row["stage"], "Experiment + reflection")
        self.assertIn("HIT", row["summary"])

    def test_transcript_expands_experiment_in_chronological_order(self):
        event = _experiment(1, "L002", True, 0.10)
        event.update({
            "action_context": "decision prompt",
            "raw_action_response": "decision response",
            "scientific_rationale": "decision rationale",
            "oracle_observation": "measured observation",
            "reflection_scientific_rationale": "reflection rationale",
            "reflection": {"next_priority": "new question"},
            "reflection_attempts": [{
                "attempt": 1,
                "context": "reflection prompt",
                "raw_response": "reflection response",
                "accepted": True,
            }],
        })
        entries = transcript_entries([event])
        self.assertEqual(
            [(entry["source"], entry["phase"]) for entry in entries],
            [("Model", "Experiment + reflection"),
             ("Oracle", "Observation"),
             ("Model", "Reflection")],
        )
        self.assertEqual(entries[0]["prompt"], "decision prompt")
        self.assertEqual(entries[2]["response"], "reflection response")

    def test_transcript_preserves_rejected_reflection_and_exports_exact_text(self):
        event = _experiment(1, "L002", False, -0.10)
        event.update({
            "action_context": "decision prompt",
            "raw_action_response": "decision response",
            "oracle_observation": "measured observation",
            "reflection_attempts": [{
                "attempt": 1,
                "context": "first reflection prompt",
                "raw_response": "malformed reflection",
                "accepted": False,
                "parse_error": "missing reflection envelope",
            }],
        })
        entries = transcript_entries([event])
        reflection = entries[-1]
        self.assertEqual(reflection["status"], "rejected")
        markdown = transcript_markdown(entries, "test_run")
        self.assertIn("first reflection prompt", markdown)
        self.assertIn("malformed reflection", markdown)
        self.assertIn("missing reflection envelope", markdown)


if __name__ == "__main__":
    unittest.main()
