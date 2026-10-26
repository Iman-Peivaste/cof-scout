"""Unit and end-to-end tests for the COF-SCOUT scientist foundation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_ara5_campaign import summarize
from analyze_ara5_run import _model_usage, render
from ara5_contracts import (
    ActionKind,
    VALIDATION_SCOPE_CROSS_SCAFFOLD,
    VALIDATION_SCOPE_LOCAL,
    classify_validation_scope,
)
from ara5_loop import (
    ScriptedScientist,
    _action_context,
    _mock_result,
    _mock_scientist,
    _persistent_rejection_feedback,
    _score_preregistered_prediction,
    _target_constraint_geometry,
    build_scientist_system_prompt,
    run_campaign,
)
from ara5_notebook import (
    DECISION_STATE_SCHEMA_VERSION,
    LEGACY_DECISION_STATE_SCHEMA_VERSION,
    MAX_HIT_SCAFFOLDS,
    MAX_HYPOTHESES,
    build_decision_state,
    build_evaluated_candidate_memory,
    build_notebook,
    build_scientific_memory,
    decision_state_sha256,
    notebook_sha256,
)
from ara5_protocol import (
    extract_scientific_plan,
    extract_scientist_action,
    extract_scientist_reflection,
)
from ara5_tools import ScientistToolbox
from assembler import load_library
from llm_client import make_client
from prompt_builder import _load_feasible_set
from validate_ara5_trace import validate_events


PLAN_RESPONSE = """A concise, reviewable scientific plan.
SCIENTIST_PLAN:
{"objective":"Find distinct scaffolds.","landscape_assessment":"Multiple gates may limit different families.","strategy":"Use discriminating experiments and revise from evidence.","hypotheses":[{"hypothesis_id":"H1","statement":"A linker change can narrow the gap.","rationale":"Conjugation and torsion depend on the linker.","confidence":"medium","discriminating_test":"Compare matched linker variants."}],"decision_principles":["Prefer informative one-variable comparisons."],"saturation_criteria":["Require repeated evidence before exhaustion."]}"""


TOOL_ACTION = """I need a factual inventory count before selecting chemistry.
SCIENTIST_ACTION:
{"action":"tool","purpose":"Inspect the inventory.","expected_information":"Determine family coverage.","tool":{"name":"inventory_summary","arguments":{}}}"""


EXPERIMENT_ACTION = """This is a controlled baseline for H1.
SCIENTIST_ACTION:
{"action":"experiment","purpose":"Establish a reference.","experiment":{"candidate":{"node_id":"N002","linker_id":"L001","linkage":"beta_ketoenamine","r_group_id":"R001"},"intent":"explore","purpose":"Measure the baseline frame.","hypothesis_id":"H1","expected_information":"Whether the gap requires linker refinement.","prediction":{"axis":"gap","direction":"wider","statement":"The corrected gap will be near or above the upper boundary.","reference_iteration":null,"hit_probability":0.35,"gate_predictions":{"gap":false,"cbm":true,"vbm":true,"stability":true},"limiting_gate":"gap","falsification_rule":"A corrected gap at or below 2.10 eV would falsify the predicted wide-gap limitation."},"decision_rule":{"if_supported":"Change the linker.","if_falsified":"Change the design axis.","if_inconclusive":"Inspect gate margins."}}}"""


REFLECTION_RESPONSE = """The observation isolates the gap as the limiting gate.
SCIENTIST_REFLECTION:
{"observation_summary":"The reference misses above the gap ceiling.","prediction_supported":true,"causal_interpretation":"The stable frame is suitable for a linker comparison.","hypothesis_updates":[{"hypothesis_id":"H1","status":"active","confidence":"medium","evidence":"The baseline provides the required matched reference.","revised_statement":null}],"strategy_update":"Change the linker while fixing other components.","next_priority":"Test the matched L002 scaffold."}"""


class Ara5ProtocolTests(unittest.TestCase):
    def test_openai_client_timeout_is_configurable(self):
        client = make_client(
            "openai_compat", "test-model", api_key="test-key",
            base_url="http://example.invalid/v1", timeout_s=12.5,
        )
        self.assertEqual(client.timeout, 12.5)

    def test_scientist_prompt_excludes_ara3_control_protocol(self):
        prompt = build_scientist_system_prompt()
        self.assertNotIn("YOUR TASK EACH ITERATION", prompt)
        self.assertNotIn("Propose exactly ONE candidate", prompt)
        self.assertIn("Act like a scientist", prompt)

    def test_target_constraint_geometry_is_exact(self):
        geometry = _target_constraint_geometry()
        coupled = geometry["coupled_feasibility"]
        self.assertEqual(coupled["at_gap_1.8_eV"], [-0.57, 0.0])
        self.assertEqual(coupled["at_gap_2.2_eV"], [-0.97, 0.0])
        self.assertIn("not unconditionally better", coupled["design_warning"])

    def test_validation_scope_is_deterministic(self):
        self.assertIsNone(classify_validation_scope("explore", True))
        self.assertEqual(
            classify_validation_scope("validate", True),
            VALIDATION_SCOPE_CROSS_SCAFFOLD,
        )
        self.assertEqual(
            classify_validation_scope("validate", False),
            VALIDATION_SCOPE_LOCAL,
        )

    def test_duplicate_rejection_feedback_persists_candidate_identity(self):
        event = {
            "event_type": "action_rejected",
            "rejection": "Candidate has already consumed an oracle experiment.",
            "action": {"experiment": {
                "node_id": "N002", "linker_id": "L004",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
            }},
        }
        feedback = _persistent_rejection_feedback([event, event])
        self.assertEqual(len(feedback), 1)
        self.assertIn("N002/L004/beta_ketoenamine/R001", feedback[0])
        self.assertIn("2 time(s)", feedback[0])

    def test_plan_protocol_preserves_rationale(self):
        rationale, plan = extract_scientific_plan(PLAN_RESPONSE)
        self.assertIn("reviewable", rationale)
        self.assertEqual(plan.hypotheses[0].hypothesis_id, "H1")
        self.assertEqual(plan.hypotheses[0].confidence, "medium")

    def test_json_envelopes_parse_with_or_without_fences(self):
        cases = (
            (extract_scientific_plan, "plan", json.loads(
                PLAN_RESPONSE.split("SCIENTIST_PLAN:\n", 1)[1]
            )),
            (extract_scientist_action, "action", json.loads(
                TOOL_ACTION.split("SCIENTIST_ACTION:\n", 1)[1]
            )),
            (extract_scientist_reflection, "reflection", json.loads(
                REFLECTION_RESPONSE.split("SCIENTIST_REFLECTION:\n", 1)[1]
            )),
        )
        for extractor, key, payload in cases:
            raw = json.dumps({
                "scientific_rationale": f"Envelope rationale for {key}.",
                key: payload,
            })
            rationale, parsed = extractor(f"```json\n{raw}\n```")
            self.assertEqual(rationale, f"Envelope rationale for {key}.")
            self.assertIsNotNone(parsed)

    def test_legacy_protocol_recovers_corrupted_inline_marker(self):
        inline = TOOL_ACTION.replace(
            "\nSCIENTIST_ACTION:\n", " SCI####IST_ACTION: "
        )
        rationale, action = extract_scientist_action(inline)
        self.assertIn("factual inventory", rationale)
        self.assertEqual(action.tool_name, "inventory_summary")

        private_character = TOOL_ACTION.replace(
            "SCIENTIST_ACTION:", "SCI\ue200ST_ACTION:"
        )
        _, private_action = extract_scientist_action(private_character)
        self.assertEqual(private_action.tool_name, "inventory_summary")

    def test_protocol_rejects_response_without_json(self):
        with self.assertRaisesRegex(ValueError, "SCIENTIST_ACTION"):
            extract_scientist_action("I would inspect the inventory next.")

    def test_malformed_envelope_feedback_does_not_request_legacy_marker(self):
        payload = TOOL_ACTION.split("SCIENTIST_ACTION:\n", 1)[1]
        malformed = json.dumps({
            "scientific_rationale": "Inspect the inventory.",
            "action": json.loads(payload),
        }) + " trailing prose"
        with self.assertRaisesRegex(ValueError, "Invalid JSON envelope") as raised:
            extract_scientist_action(malformed)
        self.assertNotIn("SCIENTIST_ACTION", str(raised.exception))

    def test_live_scientist_requests_native_json_mode(self):
        class CapturingClient:
            def __init__(self):
                self.json_object = None

            def chat_structured(self, messages, temperature=0.4, max_tokens=4096,
                                json_object=False):
                self.json_object = json_object
                return "{}", {}

        from ara5_loop import LiveScientist
        client = CapturingClient()
        LiveScientist(client, "system", 0.4).act("context")
        self.assertTrue(client.json_object)

    def test_plan_protocol_recovers_unambiguous_model_sentinel_noise(self):
        for noisy_marker in (
            "SCIENCE_PLAN:",
            "SCIENCE PLAN:",
            "SCI\ufeffENTIST_PLAN:",
            "SCI####IST_PLAN:",
            "SCI\ue200IST_PLAN:",
            "SCI\u0415NTIST_PLAN:",
        ):
            noisy = PLAN_RESPONSE.replace("SCIENTIST_PLAN:", noisy_marker)
            rationale, plan = extract_scientific_plan(noisy)
            self.assertIn("reviewable", rationale)
            self.assertEqual(plan.objective, "Find distinct scaffolds.")

    def test_action_and_reflection_protocol_recover_confusable_markers(self):
        noisy_action = TOOL_ACTION.replace(
            "SCIENTIST_ACTION:", "SCI\u0415NTIST_ACTION:"
        )
        _, action = extract_scientist_action(noisy_action)
        self.assertEqual(action.tool_name, "inventory_summary")

        noisy_reflection = REFLECTION_RESPONSE.replace(
            "SCIENTIST_REFLECTION:", "SCI\ue200IST_REFLECTION:"
        )
        _, reflection = extract_scientist_reflection(noisy_reflection)
        self.assertTrue(reflection.prediction_supported)

    def test_plan_protocol_rejects_unrelated_marker(self):
        invalid = PLAN_RESPONSE.replace("SCIENTIST_PLAN:", "FINAL_ANSWER:")
        with self.assertRaisesRegex(ValueError, "SCIENTIST_PLAN"):
            extract_scientific_plan(invalid)

    def test_tool_and_experiment_actions_validate(self):
        _, tool = extract_scientist_action(TOOL_ACTION)
        self.assertEqual(tool.kind, ActionKind.TOOL)
        self.assertEqual(tool.tool_name, "inventory_summary")
        _, experiment = extract_scientist_action(EXPERIMENT_ACTION)
        self.assertEqual(experiment.kind, ActionKind.EXPERIMENT)
        self.assertEqual(
            experiment.experiment.candidate_key,
            ("N002", "L001", "beta_ketoenamine", "R001"),
        )

    def test_model_usage_does_not_double_count_reflection_attempts(self):
        events = [{
            "event_type": "experiment",
            "cost": {
                "action": {"prompt_tokens": 10, "completion_tokens": 2, "usd": 0.1,
                           "estimated": False},
                "reflection": {"prompt_tokens": 20, "completion_tokens": 4, "usd": 0.2,
                               "estimated": False},
            },
            "reflection_attempts": [
                {"cost": {"prompt_tokens": 8, "completion_tokens": 1, "usd": 0.08}},
                {"cost": {"prompt_tokens": 12, "completion_tokens": 3, "usd": 0.12}},
            ],
        }]
        usage = _model_usage(events)
        self.assertEqual(usage["calls"], 3)
        self.assertEqual(usage["prompt_tokens"], 30)
        self.assertEqual(usage["completion_tokens"], 6)

    def test_invalid_prediction_direction_is_rejected(self):
        invalid = EXPERIMENT_ACTION.replace('"direction":"wider"', '"direction":"probably lower"')
        with self.assertRaisesRegex(ValueError, "prediction direction"):
            extract_scientist_action(invalid)

    def test_publication_prediction_fields_are_validated(self):
        _, action = extract_scientist_action(EXPERIMENT_ACTION)
        prediction = action.experiment.prediction
        self.assertTrue(prediction.is_machine_scoreable)
        self.assertEqual(prediction.hit_probability, 0.35)
        self.assertEqual(prediction.limiting_gate, "gap")
        invalid = EXPERIMENT_ACTION.replace(
            '"hit_probability":0.35', '"hit_probability":1.35'
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            extract_scientist_action(invalid)

    def test_prediction_adjudication_uses_oracle_facts(self):
        _, action = extract_scientist_action(EXPERIMENT_ACTION)
        result = _mock_result(
            "N002", "L001", "beta_ketoenamine", "R001", Path(".")
        )
        scored = _score_preregistered_prediction(action.to_dict(), result)
        self.assertTrue(scored["complete_publication_prediction"])
        self.assertEqual(scored["observed_failed_gates"], ["gap"])
        self.assertEqual(scored["gate_accuracy"], 1.0)
        self.assertTrue(scored["limiting_gate_correct"])
        self.assertTrue(scored["hit_classification_correct"])
        self.assertAlmostEqual(scored["brier_score"], 0.1225)

    def test_reflection_requires_evidence_link(self):
        rationale, reflection = extract_scientist_reflection(REFLECTION_RESPONSE)
        self.assertIn("limiting gate", rationale)
        self.assertTrue(reflection.prediction_supported)
        self.assertEqual(reflection.hypothesis_updates[0].hypothesis_id, "H1")

    def test_reflection_allows_no_material_ledger_or_strategy_change(self):
        payload = json.loads(
            REFLECTION_RESPONSE.split("SCIENTIST_REFLECTION:\n", 1)[1]
        )
        payload["hypothesis_updates"] = []
        payload["strategy_update"] = None
        raw = json.dumps({
            "scientific_rationale": "The result confirms existing policy without revising it.",
            "reflection": payload,
        })
        _, reflection = extract_scientist_reflection(raw)
        self.assertEqual(reflection.hypothesis_updates, ())
        self.assertIsNone(reflection.strategy_update)

    def test_stop_action_requires_named_saturation_criterion(self):
        missing = "Stopping now.\nSCIENTIST_ACTION:\n"
        missing += '{"action":"stop","purpose":"Stop the campaign.","reason":"Enough evidence."}'
        with self.assertRaisesRegex(ValueError, "saturation_criterion"):
            extract_scientist_action(missing)


class Ara5NotebookAndToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = load_library()
        cls.feasible_set = _load_feasible_set()
        cls.toolbox = ScientistToolbox(cls.library, cls.feasible_set)

    def test_toolbox_has_no_ground_truth_interface(self):
        self.assertFalse(any("ground_truth" in name for name in vars(self.toolbox)))
        inventory = self.toolbox.run("inventory_summary", {}, [])
        self.assertEqual(inventory["evidence_scope"], "public_inventory_and_current_run")
        self.assertEqual(inventory["result"]["candidates"], len(self.feasible_set))
        with self.assertRaisesRegex(ValueError, "Unknown scientist tool"):
            self.toolbox.run("ground_truth_lookup", {}, [])
        component = self.toolbox.run(
            "inspect_components", {"r_group_id": "R006"}, []
        )["result"]["r_group_id"]
        self.assertNotIn("effect", component)
        self.assertEqual(component["name"], "methoxy")

    def test_candidate_neighborhood_accepts_a_nested_candidate(self):
        evidence = self.toolbox.run(
            "candidate_neighborhood",
            {"candidate": {
                "node_id": "N002", "linker_id": "L001",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
            }, "limit": 5},
            [],
        )
        self.assertEqual(
            evidence["normalized_arguments"]["node_id"], "N002"
        )
        self.assertEqual(
            evidence["result"]["query"],
            ["N002", "L001", "beta_ketoenamine", "R001"],
        )
        self.assertLessEqual(len(evidence["result"]["neighbors"]), 5)

    def test_candidate_resolver_explains_incompatible_linkage(self):
        evidence = self.toolbox.run(
            "resolve_candidate",
            {"candidate": {
                "node_id": "N016", "linker_id": "L002",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
            }, "limit": 8},
            [],
        )
        result = evidence["result"]
        self.assertFalse(result["in_active_inventory"])
        self.assertEqual(result["component_compatibility"]["pair_active_linkages"], ["imine"])
        alternatives = [row["candidate"] for row in result["nearest_feasible"]]
        self.assertIn(["N002", "L002", "beta_ketoenamine", "R001"], alternatives)
        self.assertIn(["N016", "L002", "imine", "R001"], alternatives)

    def test_notebook_is_deterministic_and_separates_interpretation(self):
        _, plan = extract_scientific_plan(PLAN_RESPONSE)
        _, action = extract_scientist_action(EXPERIMENT_ACTION)
        _, reflection = extract_scientist_reflection(REFLECTION_RESPONSE)
        events = [
            {"event_type": "campaign_plan", "plan": plan.to_dict()},
            {"event_type": "experiment", "iteration": 1, "action": action.to_dict(),
             "result": _mock_result("N002", "L001", "beta_ketoenamine", "R001", Path(".")),
             "reflection": reflection.to_dict()},
        ]
        first = build_notebook(events)
        second = build_notebook(json.loads(json.dumps(events)))
        self.assertEqual(first, second)
        self.assertEqual(notebook_sha256(first), notebook_sha256(second))
        self.assertEqual(first["deterministic_facts"]["experiments"], 1)
        self.assertEqual(
            first["scientist_interpretations"]["hypotheses"]["H1"]["status"],
            "active",
        )

    def test_decision_state_is_deterministic_and_bounded(self):
        long_text = "mechanistic explanation " * 80
        events = [{
            "event_type": "campaign_plan",
            "plan": {
                "objective": long_text,
                "strategy": long_text,
                "decision_principles": [long_text] * 20,
                "saturation_criteria": [long_text] * 20,
                "hypotheses": [{
                    "hypothesis_id": f"H{index:02d}",
                    "statement": long_text,
                    "rationale": long_text,
                    "confidence": "medium",
                    "discriminating_test": long_text,
                } for index in range(20)],
            },
        }]
        for index in range(25):
            result = _mock_result(
                f"N{index:03d}", "L002", "imine", "R001", Path(".")
            )
            events.append({
                "event_type": "experiment", "iteration": index + 1,
                "action": {"experiment": {
                    "node_id": f"N{index:03d}", "linker_id": "L002",
                    "linkage": "imine", "r_group_id": "R001",
                }},
                "result": result,
            })
        first = build_decision_state(events)
        second = build_decision_state(json.loads(json.dumps(events)))
        self.assertEqual(first, second)
        self.assertEqual(decision_state_sha256(first), decision_state_sha256(second))
        facts = first["deterministic_facts"]
        interpretations = first["scientist_interpretations"]
        self.assertEqual(len(facts["hit_scaffolds"]), MAX_HIT_SCAFFOLDS)
        self.assertEqual(facts["hit_scaffolds_omitted"], 25 - MAX_HIT_SCAFFOLDS)
        self.assertEqual(len(interpretations["hypotheses"]), MAX_HYPOTHESES)
        self.assertEqual(interpretations["hypotheses_omitted"], 20 - MAX_HYPOTHESES)
        self.assertLess(len(json.dumps(first)), 70000)

    def test_evaluated_candidate_memory_keeps_old_results_and_failure_reason(self):
        events = []
        for index in range(10):
            linker_id = "L001" if index == 0 else "L002"
            events.append({
                "event_type": "experiment", "iteration": index + 1,
                "action": {"experiment": {
                    "node_id": f"N{index + 1:03d}", "linker_id": linker_id,
                    "linkage": "beta_ketoenamine", "r_group_id": "R001",
                }},
                "result": _mock_result(
                    f"N{index + 1:03d}", linker_id,
                    "beta_ketoenamine", "R001", Path("."),
                ),
            })
        memory = build_evaluated_candidate_memory(events)
        self.assertEqual(len(memory), 10)
        self.assertIn("N001/L001/beta_ketoenamine/R001", memory[0])
        self.assertIn("MISS failed=gap=2.238>2.200", memory[0])
        self.assertIn("N010/L002/beta_ketoenamine/R001 HIT", memory[-1])
        self.assertIn("all_gates_passed", memory[-1])

    def test_scientific_memory_retains_matched_changes_and_evidence_scope(self):
        _, first_action = extract_scientist_action(EXPERIMENT_ACTION)
        second_raw = EXPERIMENT_ACTION.replace('"linker_id":"L001"', '"linker_id":"L002"')
        second_raw = second_raw.replace('"reference_iteration":null', '"reference_iteration":1')
        _, second_action = extract_scientist_action(second_raw)
        events = [
            {"event_type": "experiment", "iteration": 1,
             "action": first_action.to_dict(),
             "result": _mock_result("N002", "L001", "beta_ketoenamine", "R001", Path("."))},
            {"event_type": "experiment", "iteration": 2,
             "action": second_action.to_dict(),
             "result": _mock_result("N002", "L002", "beta_ketoenamine", "R001", Path("."))},
        ]
        memory = build_scientific_memory(events)
        self.assertEqual(memory["evidence_scope"], "observations_from_current_run_only")
        pair = memory["matched_one_component_changes"][0]
        self.assertEqual(pair["iterations"], [1, 2])
        self.assertEqual(pair["changed_factor"], "linker")
        self.assertEqual(pair["change"], "L001->L002")
        self.assertTrue(pair["prediction_linked"])
        self.assertAlmostEqual(pair["property_deltas"]["gap_eV"], -0.114, places=6)
        scope = memory["linkage_evidence_scope"][0]
        self.assertEqual(scope["linkage"], "beta_ketoenamine")
        self.assertEqual(scope["nodes_tested"], ["N002"])
        self.assertEqual(scope["linkers_tested"], ["L001", "L002"])

    def test_decision_state_v2_adds_memory_without_breaking_v1_reconstruction(self):
        events = [{
            "event_type": "experiment", "iteration": 1,
            "action": {"experiment": {
                "node_id": "N002", "linker_id": "L001",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
            }},
            "result": _mock_result(
                "N002", "L001", "beta_ketoenamine", "R001", Path(".")
            ),
        }]
        current = build_decision_state(events)
        legacy = build_decision_state(
            events, schema_version=LEGACY_DECISION_STATE_SCHEMA_VERSION
        )
        self.assertEqual(
            current["decision_state_schema_version"], DECISION_STATE_SCHEMA_VERSION
        )
        self.assertIn("scientific_memory", current)
        self.assertEqual(
            legacy["decision_state_schema_version"],
            LEGACY_DECISION_STATE_SCHEMA_VERSION,
        )
        self.assertNotIn("scientific_memory", legacy)

    def test_action_context_marks_candidate_memory_exhaustive(self):
        memory = [
            "E9 N002/L004/beta_ketoenamine/R001 MISS failed=VBM=1.022<1.230"
        ]
        context = _action_context(
            {}, {}, 50, 9, 0, 6, [], [], memory,
        )
        self.assertIn("PERMANENT EVALUATED-CANDIDATE MEMORY", context)
        self.assertIn("This list is exhaustive", context)
        self.assertIn("N002/L004/beta_ketoenamine/R001", context)
        self.assertIn("select a different exact candidate", context)
        self.assertIn("SCIENTIFIC LEARNING MEMORY NOTE", context)
        self.assertIn("seek transfer or counterexample experiments", context)

    def test_invalid_candidate_repair_persists_in_decision_state(self):
        repair = self.toolbox.run(
            "resolve_candidate",
            {"candidate": {
                "node_id": "N016", "linker_id": "L002",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
            }},
            [],
        )
        events = [{
            "event_type": "action_rejected", "iteration": 3,
            "rejection": "Candidate is absent from the exact active inventory.",
            "action": {"experiment": {
                "node_id": "N016", "linker_id": "L002",
                "linkage": "beta_ketoenamine", "r_group_id": "R001",
            }},
            "repair_evidence": repair,
        }]
        state = build_decision_state(events)
        ledger = state["rejection_memory"]["invalid_candidates"]
        self.assertEqual(ledger[0]["attempts"], 1)
        self.assertEqual(ledger[0]["candidate"][0], "N016")
        self.assertTrue(ledger[0]["nearest_feasible"])

    def test_campaign_summary_uses_only_trace_observations(self):
        empty = self.toolbox.run("campaign_summary", {}, [])
        self.assertEqual(empty["result"]["experiments"], 0)
        self.assertEqual(empty["result"]["hits"], 0)

    def test_matched_pair_analysis_uses_observed_experiments(self):
        _, first_action = extract_scientist_action(EXPERIMENT_ACTION)
        second_raw = EXPERIMENT_ACTION.replace('"linker_id":"L001"', '"linker_id":"L002"')
        _, second_action = extract_scientist_action(second_raw)
        events = [
            {"event_type": "experiment", "iteration": 1,
             "action": first_action.to_dict(),
             "result": _mock_result("N002", "L001", "beta_ketoenamine", "R001", Path("."))},
            {"event_type": "experiment", "iteration": 2,
             "action": second_action.to_dict(),
             "result": _mock_result("N002", "L002", "beta_ketoenamine", "R001", Path("."))},
        ]
        evidence = self.toolbox.run(
            "matched_pair_analysis", {"factor": "linker", "axis": "gap"}, events
        )
        transitions = evidence["result"]["matched_transitions"]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["transition"], "L001->L002")
        self.assertAlmostEqual(transitions[0]["mean_delta"], -0.114, places=6)


class Ara5EndToEndTests(unittest.TestCase):
    def test_mock_campaign_records_complete_provenance_without_real_oracle(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            events = run_campaign(
                _mock_scientist(), toolbox, _mock_result, run_dir,
                max_experiments=2, max_tool_calls=3, max_decisions_per_experiment=8,
            )
            experiments = [event for event in events if event.get("event_type") == "experiment"]
            tools = [event for event in events if event.get("event_type") == "tool_call"]
            self.assertEqual(len(experiments), 2)
            self.assertEqual(len(tools), 2)
            self.assertTrue((run_dir / "trace.jsonl").is_file())
            self.assertIn("action_context", experiments[0])
            self.assertIn("raw_action_response", experiments[0])
            self.assertIn("reflection_attempts", experiments[0])
            self.assertIn("notebook_sha256_before", experiments[0])
            self.assertIn("notebook_sha256_after", experiments[0])
            self.assertIn("decision_state_sha256_before", experiments[0])
            self.assertIn("decision_state_sha256_after", experiments[0])
            self.assertIn(
                "N002/L001/beta_ketoenamine/R001",
                experiments[1]["action_context"],
            )
            self.assertIn(
                "MISS failed=gap=2.238>2.200",
                experiments[1]["action_context"],
            )
            notebook = build_notebook(events)
            self.assertEqual(notebook["deterministic_facts"]["experiments"], 2)
            self.assertEqual(notebook["deterministic_facts"]["hit_scaffolds"], 1)
            self.assertEqual(
                notebook["scientist_interpretations"]["hypotheses"]["H1"]["status"],
                "supported",
            )
            audit = render(events, {"run_id": "mock", "model": "mock"})
            self.assertIn("Pre-registered branching decision", audit)
            self.assertIn("Final hypothesis ledger", audit)
            self.assertIn("N002/L002/beta_ketoenamine/R001", audit)
            summary = validate_events(events, notebook)
            self.assertEqual(summary["experiments"], 2)
            self.assertEqual(summary["tool_calls"], 2)
            campaign = summarize(events)
            self.assertEqual(campaign["metrics"]["experiments"], 2)
            self.assertEqual(campaign["metrics"]["hit_scaffolds"], 1)
            self.assertEqual(campaign["metrics"]["first_hit_iteration"], 2)
            self.assertEqual(campaign["metrics"]["machine_scored_predictions"], 2)
            self.assertEqual(campaign["metrics"]["mean_gate_prediction_accuracy"], 1.0)

            resumed = run_campaign(
                _mock_scientist(), toolbox, _mock_result, run_dir,
                max_experiments=2, max_tool_calls=3, max_decisions_per_experiment=8,
                resume=True,
            )
            self.assertEqual(len(resumed), len(events))

    def test_portfolio_guards_reject_unjustified_and_over_budget_decoration(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        repeat_explore = EXPERIMENT_ACTION.replace(
            '"r_group_id":"R001"', '"r_group_id":"R002"'
        )
        repeat_refine = repeat_explore.replace('"intent":"explore"', '"intent":"refine"')
        distinct = EXPERIMENT_ACTION.replace('"linker_id":"L001"', '"linker_id":"L002"')
        scientist = ScriptedScientist(
            PLAN_RESPONSE,
            [EXPERIMENT_ACTION, repeat_explore, repeat_refine, distinct],
            [REFLECTION_RESPONSE, REFLECTION_RESPONSE],
        )
        with tempfile.TemporaryDirectory() as directory:
            events = run_campaign(
                scientist, toolbox, _mock_result, Path(directory) / "run",
                max_experiments=2, max_tool_calls=0,
                max_decisions_per_experiment=6, max_decoration_fraction=0.0,
            )
        rejections = [
            event for event in events
            if event.get("event_type") == "action_rejected"
        ]
        self.assertIn(
            "A previously tested scaffold requires refine or validate intent.",
            [event["rejection"] for event in rejections],
        )
        self.assertIn(
            "Decoration budget exhausted; select a previously untested scaffold.",
            [event["rejection"] for event in rejections],
        )
        self.assertEqual(
            sum(event.get("event_type") == "experiment" for event in events), 2
        )

    def test_validation_scope_is_stored_and_reported(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        cross_scaffold = EXPERIMENT_ACTION.replace(
            '"intent":"explore"', '"intent":"validate"'
        )
        local_confirmation = cross_scaffold.replace(
            '"r_group_id":"R001"', '"r_group_id":"R002"'
        )
        scientist = ScriptedScientist(
            PLAN_RESPONSE,
            [cross_scaffold, local_confirmation],
            [REFLECTION_RESPONSE, REFLECTION_RESPONSE],
        )
        with tempfile.TemporaryDirectory() as directory:
            events = run_campaign(
                scientist, toolbox, _mock_result, Path(directory) / "run",
                max_experiments=2, max_tool_calls=0,
                max_decisions_per_experiment=4, max_decoration_fraction=0.5,
            )
        experiments = [
            event for event in events if event.get("event_type") == "experiment"
        ]
        self.assertEqual(
            [event["validation_scope"] for event in experiments],
            [VALIDATION_SCOPE_CROSS_SCAFFOLD, VALIDATION_SCOPE_LOCAL],
        )
        campaign = summarize(events)
        self.assertEqual(campaign["metrics"]["validation_scope_counts"], {
            VALIDATION_SCOPE_CROSS_SCAFFOLD: 1,
            VALIDATION_SCOPE_LOCAL: 1,
        })
        audit = render(events, None)
        self.assertIn(VALIDATION_SCOPE_CROSS_SCAFFOLD, audit)
        self.assertIn(VALIDATION_SCOPE_LOCAL, audit)

    def test_duplicate_candidate_gets_unevaluated_scaffold_repair_evidence(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        distinct = EXPERIMENT_ACTION.replace(
            '"linker_id":"L001"', '"linker_id":"L002"'
        )
        scientist = ScriptedScientist(
            PLAN_RESPONSE,
            [EXPERIMENT_ACTION, EXPERIMENT_ACTION, distinct],
            [REFLECTION_RESPONSE, REFLECTION_RESPONSE],
        )
        with tempfile.TemporaryDirectory() as directory:
            events = run_campaign(
                scientist, toolbox, _mock_result, Path(directory) / "run",
                max_experiments=2, max_tool_calls=0,
                max_decisions_per_experiment=4,
            )
        rejection = next(
            event for event in events
            if event.get("event_type") == "action_rejected"
        )
        self.assertEqual(
            rejection["rejection"],
            "Candidate has already consumed an oracle experiment.",
        )
        self.assertEqual(rejection["repair_evidence"]["tool"], "scaffold_frontier")
        self.assertTrue(rejection["repair_evidence"]["result"]["returned"])

    def test_audit_renders_candidate_repair_evidence(self):
        events = [{
            "event_type": "action_rejected", "iteration": 1,
            "rejection": "Candidate is absent from the exact active inventory.",
            "repair_evidence": {"tool": "resolve_candidate", "result": {
                "nearest_feasible": [{"candidate": ["N016", "L002", "imine", "R001"]}],
            }},
        }]
        audit = render(events, None)
        self.assertIn("Deterministic compatibility and repair evidence", audit)
        self.assertIn("resolve_candidate", audit)

    def test_existing_trace_requires_explicit_resume(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_campaign(
                _mock_scientist(), toolbox, _mock_result, run_dir,
                max_experiments=1, max_tool_calls=2, max_decisions_per_experiment=5,
            )
            with self.assertRaises(FileExistsError):
                run_campaign(
                    _mock_scientist(), toolbox, _mock_result, run_dir,
                    max_experiments=1,
                )

    def test_unregistered_saturation_claim_is_rejected(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        invalid_stop = """I believe the campaign is complete.
SCIENTIST_ACTION:
{"action":"stop","purpose":"End early.","saturation_criterion":"I have a feeling the space is exhausted.","reason":"No further experiments appear useful."}"""
        scientist = ScriptedScientist(
            PLAN_RESPONSE, [invalid_stop, EXPERIMENT_ACTION], [REFLECTION_RESPONSE]
        )
        with tempfile.TemporaryDirectory() as directory:
            events = run_campaign(
                scientist, toolbox, _mock_result, Path(directory) / "run",
                max_experiments=1, max_tool_calls=1, max_decisions_per_experiment=4,
            )
        rejected = [event for event in events if event.get("event_type") == "action_rejected"]
        experiments = [event for event in events if event.get("event_type") == "experiment"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("pre-registered saturation criterion", rejected[0]["rejection"])
        self.assertEqual(len(experiments), 1)

    def test_publication_mode_rejects_early_stop_and_completes_budget(self):
        library = load_library()
        toolbox = ScientistToolbox(library, _load_feasible_set())
        registered_stop = """The initial criterion is already satisfied.
SCIENTIST_ACTION:
{"action":"stop","purpose":"End the campaign.","saturation_criterion":"Require repeated evidence before exhaustion.","reason":"A premature stopping test."}"""
        scientist = ScriptedScientist(
            PLAN_RESPONSE, [registered_stop, EXPERIMENT_ACTION], [REFLECTION_RESPONSE]
        )
        with tempfile.TemporaryDirectory() as directory:
            events = run_campaign(
                scientist, toolbox, _mock_result, Path(directory) / "run",
                max_experiments=1, max_tool_calls=0,
                max_decisions_per_experiment=3, fixed_horizon=True,
                require_scored_predictions=True,
            )
        rejected = [event for event in events if event.get("event_type") == "action_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("Fixed-horizon publication mode", rejected[0]["rejection"])
        self.assertEqual(
            sum(event.get("event_type") == "experiment" for event in events), 1
        )


if __name__ == "__main__":
    unittest.main()
