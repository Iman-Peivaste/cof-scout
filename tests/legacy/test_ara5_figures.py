"""Tests for reusable COF-SCOUT campaign figures."""

from __future__ import annotations

import unittest

from ara5_dashboard import campaign_metrics
from ara5_figures import (
    campaign_trajectory_figure,
    decision_episode_figure,
    method_comparison_figure,
    node_linkage_matrix_figure,
)


def _experiment(
    iteration: int,
    node: str,
    linker: str,
    linkage: str,
    r_group: str,
    hit: bool,
    vbm_margin: float,
) -> dict:
    return {
        "event_type": "experiment",
        "event_index": iteration + 1,
        "iteration": iteration,
        "action": {"experiment": {
            "node_id": node,
            "linker_id": linker,
            "linkage": linkage,
            "r_group_id": r_group,
            "intent": "explore",
            "purpose": "Test a recorded hypothesis.",
            "hypothesis_id": "H1",
            "prediction": {"statement": "Expected overall pass."},
        }},
        "result": {
            "node": node,
            "linker": linker,
            "linkage": linkage,
            "r_group": r_group,
            "status": "success",
            "gap_corrected_eV": 2.0,
            "cbm_vs_nhe": -0.2,
            "vbm_vs_nhe": 1.8,
            "s_linkage": 0.9,
            "criterion": {
                "is_hit": hit,
                "gap_pass": hit,
                "cbm_pass": True,
                "vbm_pass": hit,
                "stability_pass": True,
                "vbm_margin": vbm_margin,
            },
        },
        "reflection": {
            "observation_summary": "Recorded result interpreted.",
            "next_priority": "Test a contrasting scaffold.",
        },
    }


class Ara5FigureTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {
                "event_type": "campaign_plan",
                "event_index": 1,
                "plan": {"objective": "Find diverse valid COFs."},
            },
            {
                "event_type": "tool_call",
                "event_index": 2,
                "action": {"tool_name": "scaffold_frontier"},
            },
            _experiment(1, "N001", "L001", "imine", "R001", False, -0.1),
            _experiment(2, "N002", "L018", "vinylene", "R001", True, 0.40),
            _experiment(3, "N002", "L018", "vinylene", "R002", True, 0.10),
        ]

    def test_trajectory_has_four_cumulative_series_and_linkages(self):
        figure = campaign_trajectory_figure(self.events)
        names = [trace.name for trace in figure.data]
        self.assertTrue({
            "Tested scaffolds", "Raw hits", "Distinct hit scaffolds",
            "Robust hit scaffolds", "imine", "vinylene",
        }.issubset(names))
        raw_hits = next(trace for trace in figure.data if trace.name == "Raw hits")
        self.assertEqual(list(raw_hits.y), [0, 1, 2])

    def test_method_comparison_contains_both_baselines_and_ara5(self):
        baselines = {
            "random": {"per_seed": [{"seed": 0, "hits": 1, "scaffolds": 1}]},
            "gp_ucb_morgan": {
                "per_seed": [{"seed": 0, "hits": 8, "scaffolds": 3}]
            },
        }
        figure = method_comparison_figure(
            baselines, campaign_metrics(self.events, {"max_experiments": 3})
        )
        names = [trace.name for trace in figure.data]
        self.assertTrue(any(name.startswith("Random") for name in names))
        self.assertTrue(any(name.startswith("GP-UCB") for name in names))
        self.assertIn("COF-SCOUT", names)

    def test_method_comparison_accepts_publication_study_schema(self):
        baselines = {
            "methods": {
                "candidate_random": {
                    "per_seed": [{"seed": 0, "hits": 1, "scaffolds": 1}]
                },
                "scaffold_first_random": {
                    "per_seed": [{"seed": 0, "hits": 2, "scaffolds": 2}]
                },
                "scaffold_maxmin": {
                    "per_seed": [{"seed": 0, "hits": 3, "scaffolds": 3}]
                },
                "candidate_morgan_gp_ucb": {
                    "per_seed": [{"seed": 0, "hits": 4, "scaffolds": 2}]
                },
                "scaffold_aware_morgan_gp_ucb": {
                    "per_seed": [{"seed": 0, "hits": 5, "scaffolds": 4}]
                },
            }
        }
        figure = method_comparison_figure(
            baselines, {"hits": 6, "hit_scaffolds": 5}
        )
        names = [trace.name for trace in figure.data]
        self.assertTrue(any(name.startswith("Candidate random") for name in names))
        self.assertTrue(any(name.startswith("Scaffold-first random") for name in names))
        self.assertTrue(any(name.startswith("Scaffold MaxMin") for name in names))
        self.assertTrue(any(name.startswith("Candidate GP-UCB") for name in names))
        self.assertTrue(any(name.startswith("Scaffold-aware GP-UCB") for name in names))
        self.assertIn("COF-SCOUT", names)

    def test_matrix_encodes_miss_hit_robust_and_untested(self):
        figure = node_linkage_matrix_figure(self.events)
        heatmap = figure.data[0]
        values = {
            (node, linkage): heatmap.z[row_index][column_index]
            for row_index, node in enumerate(heatmap.y)
            for column_index, linkage in enumerate(heatmap.x)
        }
        self.assertEqual(values[("N001", "imine")], 1)
        self.assertEqual(values[("N002", "vinylene")], 3)
        self.assertEqual(values[("N001", "vinylene")], 0)

    def test_decision_episode_defaults_to_first_hit_and_serializes(self):
        figure = decision_episode_figure(self.events)
        self.assertEqual(figure.layout.meta["selected_iteration"], 2)
        self.assertIn("Pre-register", figure.to_html(include_plotlyjs=False))

    def test_all_figures_serialize_to_html(self):
        baselines = {"random": {"per_seed": []}, "gp_ucb_morgan": {"per_seed": []}}
        figures = [
            campaign_trajectory_figure(self.events),
            method_comparison_figure(baselines, {"hits": 2, "hit_scaffolds": 1}),
            node_linkage_matrix_figure(self.events),
            decision_episode_figure(self.events, 1),
        ]
        for figure in figures:
            self.assertIn("plotly-graph-div", figure.to_html(include_plotlyjs=False))


if __name__ == "__main__":
    unittest.main()
