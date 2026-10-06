"""Tests for reflector.py."""

import json

import pytest
from clinic_agent.loop.reflector import (
    ReflectorInput, FailureReport, build_reflector_input, reflect,
    HeldoutLeakError, InvalidJSONError, ReflectorError
)
from clinic_agent.loop.patch import Patch, FailureCategory, PolicyChange, Evidence


# Load all heldout scenario data for parametrization
import yaml
from pathlib import Path

HELDOUT_SCENARIOS = []
HELDOUT_PERSONAS = {}
HELDOUT_OPENINGS = {}
# Build parametrize lists for slice tests (8+ words only; shorter texts covered by whole-text tests)
HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_PERSONA = []
HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_OPENING = []
scenarios_dir = Path(__file__).resolve().parents[1] / "clinic_agent" / "evals" / "scenarios"
for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data.get("split") == "heldout":
        scenario_id = data.get("id", "")
        HELDOUT_SCENARIOS.append(scenario_id)
        HELDOUT_PERSONAS[scenario_id] = data.get("patient_persona", "")
        HELDOUT_OPENINGS[scenario_id] = data.get("opening", "")

        # Track scenarios with 8+ words for slice tests
        persona_words = HELDOUT_PERSONAS[scenario_id].split()
        if len(persona_words) >= 8:
            HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_PERSONA.append(scenario_id)

        opening_words = HELDOUT_OPENINGS[scenario_id].split()
        if len(opening_words) >= 8:
            HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_OPENING.append(scenario_id)


class TestBuildReflectorInput:
    """Test build_reflector_input filters correctly."""

    def test_includes_only_train_failures(self, tmp_path):
        """Only train-split scenarios with pass rate < 1.0 should be included."""
        results = {
            "scenarios": [
                {
                    "id": "test1",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": "Test transcript",
                    "opening": "Hello",
                    "patient_persona": "Patient persona one",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "test2",
                    "split": "train",
                    "pass_rate": 1.0,  # No failures
                    "is_infra_error": False,
                    "transcript": "Test transcript",
                    "opening": "Hello",
                    "patient_persona": "Patient persona two",
                    "tool_trace": [],
                    "state_check_failures": [],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "test3",
                    "split": "heldout",  # Heldout - should be excluded
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": "Test transcript",
                    "opening": "Hello",
                    "patient_persona": "Patient persona three",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "test4",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": True,  # Infra error - should be excluded
                    "transcript": "Test transcript",
                    "opening": "Hello",
                    "patient_persona": "Patient persona four",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        assert len(input_data.failures) == 1
        assert input_data.failures[0].scenario_id == "test1"

    @pytest.mark.parametrize("heldout_id", HELDOUT_SCENARIOS if HELDOUT_SCENARIOS else ["test_heldout_scenario"])
    def test_heldout_id_never_in_prompt(self, heldout_id, tmp_path):
        """Heldout scenario IDs should never appear in the built prompt, even in train data."""
        results = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": "Test transcript",
                    "opening": "Hello",
                    "patient_persona": "Unique persona description",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    # If train transcript contains heldout ID, should raise
                    "transcript": f"Test scenario {heldout_id} mentioned",
                    "opening": "Hello",
                    "patient_persona": "Generic train patient",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        # Should raise HeldoutLeakError because heldout ID appears in train transcript
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    def test_infra_error_runs_excluded(self, tmp_path):
        """Scenarios marked as infra-error should be excluded."""
        results = {
            "scenarios": [
                {
                    "id": "infra_test",
                    "split": "train",
                    "pass_rate": 0.0,
                    "is_infra_error": True,  # Infra error
                    "transcript": "Sample conversation transcript",
                    "opening": "Hello there",
                    "patient_persona": "Patient description goes here",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        assert len(input_data.failures) == 0

    @pytest.mark.parametrize("heldout_id", HELDOUT_SCENARIOS if HELDOUT_SCENARIOS else ["test_heldout_scenario"])
    def test_heldout_opening_not_in_prompt(self, heldout_id, tmp_path):
        """Heldout scenario opening text should not appear in the built prompt."""
        heldout_opening = HELDOUT_OPENINGS.get(heldout_id, "")
        if not heldout_opening:
            pytest.skip(f"No opening found for heldout scenario {heldout_id}")

        results = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "opening": heldout_opening,
                    "patient_persona": "Test",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": "Sample conversation transcript",
                    "opening": "Hello there",
                    "patient_persona": "Patient description goes here",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    @pytest.mark.parametrize("heldout_id", HELDOUT_SCENARIOS if HELDOUT_SCENARIOS else ["test_heldout_scenario"])
    def test_heldout_persona_never_in_prompt(self, heldout_id, tmp_path):
        """Heldout scenario persona text should never appear in the built prompt."""
        heldout_persona = HELDOUT_PERSONAS.get(heldout_id, "")
        if not heldout_persona:
            pytest.skip(f"No persona found for heldout scenario {heldout_id}")

        results = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "patient_persona": heldout_persona,
                    "opening": "Hello",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    # Inject FULL heldout persona into train transcript - should be caught
                    "transcript": f"Test transcript with full persona: {heldout_persona}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    @pytest.mark.parametrize("heldout_id", HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_PERSONA if HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_PERSONA else ["test_heldout_scenario"])
    def test_heldout_persona_slice_not_in_prompt(self, heldout_id, tmp_path):
        """A slice of heldout persona (min 8 words, up to 10) should trigger leak detection.
        Heldout personas under 8 words are covered by the full-text persona test."""
        heldout_persona = HELDOUT_PERSONAS.get(heldout_id, "")
        words = heldout_persona.split()
        slice_size = min(10, len(words))
        persona_slice = " ".join(words[:slice_size])

        results = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "patient_persona": heldout_persona,
                    "opening": "Hello",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    # Inject persona slice into train transcript
                    "transcript": f"Test transcript with persona slice: {persona_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    @pytest.mark.parametrize("heldout_id", HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_OPENING if HELDOUT_SCENARIOS_WITH_8PLUS_WORDS_OPENING else ["test_heldout_scenario"])
    def test_heldout_opening_slice_not_in_prompt(self, heldout_id, tmp_path):
        """A slice of heldout opening (min 8 words, up to 10) should trigger leak detection.
        Heldout openings under 8 words are covered by the full-text opening test."""
        heldout_opening = HELDOUT_OPENINGS.get(heldout_id, "")
        words = heldout_opening.split()
        slice_size = min(10, len(words))
        opening_slice = " ".join(words[:slice_size])

        results = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "opening": heldout_opening,
                    "patient_persona": "Test",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    # Inject opening slice into train transcript
                    "transcript": f"Test transcript with opening slice: {opening_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    def test_synthetic_heldout_8word_opening_boundary(self, tmp_path):
        """Test 8-word run detection with synthetic heldout opening: 9 words triggers, 7 words does not."""
        synthetic_opening = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
        # 20 distinctive words

        policy = {"tone": {"text": "Be nice"}}

        # Test 1: 9 consecutive words should trigger
        words = synthetic_opening.split()
        nine_word_slice = " ".join(words[5:14])  # Middle 9 words

        results = {
            "scenarios": [
                {
                    "id": "synthetic_heldout",
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "opening": synthetic_opening,
                    "patient_persona": "SyntheticPersonaXYZ",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": f"Test transcript with slice: {nine_word_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

        # Test 2: 7 consecutive words should NOT trigger (boundary)
        seven_word_slice = " ".join(words[5:12])  # Middle 7 words

        results = {
            "scenarios": [
                {
                    "id": "synthetic_heldout",
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "opening": synthetic_opening,
                    "patient_persona": "SyntheticPersonaXYZ",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": f"Test transcript with slice: {seven_word_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        # Should NOT raise - 7 words is below the 8-word threshold
        input_data = build_reflector_input(results, tmp_path, policy)
        assert len(input_data.failures) == 1

    def test_synthetic_heldout_8word_persona_boundary(self, tmp_path):
        """Test 8-word run detection with synthetic heldout persona: 9 words triggers, 7 words does not."""
        synthetic_persona = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciseis diecisiete dieciocho diecinueve veinte"
        # 20 distinctive words

        policy = {"tone": {"text": "Be nice"}}

        # Test 1: 9 consecutive words should trigger
        words = synthetic_persona.split()
        nine_word_slice = " ".join(words[5:14])  # Middle 9 words

        results = {
            "scenarios": [
                {
                    "id": "synthetic_heldout",
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "patient_persona": synthetic_persona,
                    "opening": "SyntheticOpeningXYZ",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": f"Test transcript with slice: {nine_word_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

        # Test 2: 7 consecutive words should NOT trigger (boundary)
        seven_word_slice = " ".join(words[5:12])  # Middle 7 words

        results = {
            "scenarios": [
                {
                    "id": "synthetic_heldout",
                    "split": "heldout",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "patient_persona": synthetic_persona,
                    "opening": "SyntheticOpeningXYZ",
                    "transcript": "Test transcript",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                },
                {
                    "id": "train_test",
                    "split": "train",
                    "pass_rate": 0.5,
                    "is_infra_error": False,
                    "transcript": f"Test transcript with slice: {seven_word_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail1"],
                    "trace_check_failures": [],
                    "judge_failures": ["fail2"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        # Should NOT raise - 7 words is below the 8-word threshold
        input_data = build_reflector_input(results, tmp_path, policy)
        assert len(input_data.failures) == 1

    def test_real_schema_heldout_9word_slice_raises(self, tmp_path):
        """Test that 9-word slice detection works with real results.json schema (opening/persona loaded from YAML)."""
        # Use a real heldout scenario ID
        heldout_id = "are_you_a_robot"  # This is a real heldout scenario
        heldout_opening = HELDOUT_OPENINGS.get(heldout_id, "")
        heldout_persona = HELDOUT_PERSONAS.get(heldout_id, "")

        # Take 9 consecutive words from the middle of the persona if available
        words = heldout_persona.split()
        if len(words) >= 9:
            nine_word_slice = " ".join(words[:9])
        else:
            nine_word_slice = heldout_persona  # Use full text if too short

        policy = {"tone": {"text": "Be nice"}}

        results = {
            "timestamp": "20240101_120000",
            "policy_hash": "abc123",
            "k": 3,
            "llm_mode": "mock",
            "agent_model": "test",
            "judge_model": "test",
            "sim_model": "test",
            "overall": {"pass_rate": 0.5, "passed": 1, "total": 2},
            "train": {"pass_rate": 0.5, "passed": 1, "total": 2},
            "heldout": {"pass_rate": 1.0, "passed": 1, "total": 1},
            "per_scenario": {
                heldout_id: {
                    "split": "heldout",
                    "tags": [],
                    "pass_count": 1,
                    "total_runs": 1,
                    "pass_rate": 1.0,
                    "runs": [
                        {"scenario_id": heldout_id, "run_index": 0, "passed": True, "error": None},
                    ]
                },
                "train_with_leak": {
                    "split": "train",
                    "tags": [],
                    "pass_count": 0,
                    "total_runs": 1,
                    "pass_rate": 0.0,
                    "runs": [
                        {"scenario_id": "train_with_leak", "run_index": 0, "passed": False, "error": None},
                    ]
                }
            },
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        # Inject the heldout slice into the train scenario's transcript
        # Since results.json doesn't have transcript, we simulate this by
        # using the test fixture format for this specific test
        results_test_fixture = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 1.0,
                    "is_infra_error": False,
                    "opening": heldout_opening,
                    "patient_persona": heldout_persona,
                    "transcript": "Test",
                    "tool_trace": [],
                    "state_check_failures": [],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {}
                },
                {
                    "id": "train_with_leak",
                    "split": "train",
                    "pass_rate": 0.0,
                    "is_infra_error": False,
                    "transcript": f"Test transcript with heldout slice: {nine_word_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail"],
                    "judge_failures": ["fail"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results_test_fixture, tmp_path, policy)

    def test_real_results_json_format(self, tmp_path):
        """Test that build_reflector_input works with real results.json schema."""
        # Create trace files
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        train_trace = traces_dir / "train_scenario_1.jsonl"
        # 3 runs with conversation_end markers
        train_trace.write_text(
            '{"event": "patient_turn", "response": "run 0"}\n'
            '{"event": "conversation_end"}\n'
            '{"event": "patient_turn", "response": "run 1 failing"}\n'
            '{"event": "conversation_end"}\n'
            '{"event": "patient_turn", "response": "run 2"}\n'
            '{"event": "conversation_end"}'
        )
        heldout_trace = traces_dir / "heldout_scenario_1.jsonl"
        heldout_trace.write_text('{"event": "patient_turn", "response": "heldout"}\n{"event": "conversation_end"}')

        results = {
            "timestamp": "20240101_120000",
            "policy_hash": "abc123",
            "k": 3,
            "llm_mode": "mock",
            "agent_model": "test",
            "judge_model": "test",
            "sim_model": "test",
            "overall": {"pass_rate": 0.5, "passed": 1, "total": 2},
            "train": {"pass_rate": 0.5, "passed": 1, "total": 2},
            "heldout": {"pass_rate": 1.0, "passed": 1, "total": 1},
            "per_scenario": {
                "train_scenario_1": {
                    "split": "train",
                    "tags": [],
                    "pass_count": 1,
                    "total_runs": 3,
                    "pass_rate": 0.33,
                    "runs": [
                        {"scenario_id": "train_scenario_1", "run_index": 0, "passed": False, "error": None, "scores": {"failures": []}},
                        {"scenario_id": "train_scenario_1", "run_index": 1, "passed": True, "error": None, "scores": {"failures": []}},
                        {"scenario_id": "train_scenario_1", "run_index": 2, "passed": False, "error": None, "scores": {"failures": []}}
                    ]
                },
                "heldout_scenario_1": {
                    "split": "heldout",
                    "tags": [],
                    "pass_count": 1,
                    "total_runs": 1,
                    "pass_rate": 1.0,
                    "runs": [
                        {"scenario_id": "heldout_scenario_1", "run_index": 0, "passed": True, "error": None, "scores": {"failures": []}}
                    ]
                }
            },
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        # Should include only train scenario with pass_rate < 1.0
        assert len(input_data.failures) == 1
        assert input_data.failures[0].scenario_id == "train_scenario_1"
        assert input_data.failures[0].pass_rate == 0.33
        # Should select run 0 (first failing run)
        assert "run 0" in input_data.failures[0].transcript
        assert "run 1 failing" not in input_data.failures[0].transcript

    def test_failing_run_transcript_in_prompt(self, tmp_path):
        """Failing run's transcript and policy text should appear in the prompt."""
        results = {
            "scenarios": [
                {
                    "id": "failing_scenario",
                    "split": "train",
                    "pass_rate": 0.0,
                    "is_infra_error": False,
                    "transcript": "Patient: Patient message\nAgent: Agent message",
                    "opening": "Hello",
                    "patient_persona": "Test",
                    "tool_trace": [],
                    "state_check_failures": ["fail"],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {}
                }
            ],
            "current_policy": {"tone": {"text": "Be polite and concise"}}
        }

        policy = {"tone": {"text": "Be polite and concise"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        # Should have one failure
        assert len(input_data.failures) == 1
        # Transcript should be in the failure
        assert "Patient message" in input_data.failures[0].transcript
        assert "Agent message" in input_data.failures[0].transcript
        # Policy text should be in the failure's current_policy_sections
        assert "Be polite and concise" in input_data.failures[0].current_policy_sections.get("tone", "")

    def test_passing_run_transcript_not_in_prompt(self, tmp_path):
        """Passing run's transcript should not appear (no failures to report)."""
        results = {
            "scenarios": [
                {
                    "id": "passing_scenario",
                    "split": "train",
                    "pass_rate": 1.0,  # All passed
                    "is_infra_error": False,
                    "transcript": "Patient: This should not appear\nAgent: This should not appear",
                    "opening": "Hello",
                    "patient_persona": "Test",
                    "tool_trace": [],
                    "state_check_failures": [],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {}
                }
            ],
            "current_policy": {"tone": {"text": "Be polite"}}
        }

        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        # Should have no failures (pass_rate >= 1.0)
        assert len(input_data.failures) == 0

    def test_heldout_9word_slice_in_trace_raises(self, tmp_path):
        """A heldout 9-word slice in a train trace raises HeldoutLeakError."""
        # Use test fixture format for this test (easier to control)
        heldout_id = "are_you_a_robot"
        heldout_persona = HELDOUT_PERSONAS.get(heldout_id, "")
        if len(heldout_persona.split()) < 9:
            pytest.skip(f"Heldout scenario {heldout_id} persona has fewer than 9 words")

        heldout_slice = " ".join(heldout_persona.split()[:9])

        results = {
            "scenarios": [
                {
                    "id": heldout_id,
                    "split": "heldout",
                    "pass_rate": 1.0,
                    "is_infra_error": False,
                    "patient_persona": heldout_persona,
                    "opening": "Hello",
                    "transcript": "Test",
                    "tool_trace": [],
                    "state_check_failures": [],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {}
                },
                {
                    "id": "train_with_leak",
                    "split": "train",
                    "pass_rate": 0.0,
                    "is_infra_error": False,
                    # Inject heldout slice into transcript
                    "transcript": f"Test transcript with heldout slice: {heldout_slice}",
                    "opening": "Hello",
                    "patient_persona": "Generic train persona",
                    "tool_trace": [],
                    "state_check_failures": ["fail"],
                    "judge_failures": ["fail"],
                    "current_policy_sections": {"tone": "Be nice"}
                }
            ],
            "current_policy": {"tone": {"text": "Be nice"}}
        }

        policy = {"tone": {"text": "Be nice"}}
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    def test_no_failing_runs_clean_error(self, tmp_path):
        """No failing runs should result in empty ReflectorInput (clean state)."""
        results = {
            "scenarios": [
                {
                    "id": "train_scenario",
                    "split": "train",
                    "pass_rate": 1.0,  # All passed
                    "is_infra_error": False,
                    "transcript": "Test",
                    "opening": "Hello",
                    "patient_persona": "Test",
                    "tool_trace": [],
                    "state_check_failures": [],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {}
                }
            ],
            "current_policy": {"tone": {"text": "Be polite"}}
        }

        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        # Should return empty input cleanly
        assert len(input_data.failures) == 0
        # Policy is kept as original dict in ReflectorInput
        assert input_data.current_policy == {"tone": {"text": "Be polite"}}

    def test_state_and_trace_failures_in_prompt(self, tmp_path):
        """Failing state and trace keys appear in the right prompt sections."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        trace_file = traces_dir / "test_scenario.jsonl"
        trace_file.write_text('{"event": "patient_turn", "response": "test"}\n{"event": "conversation_end"}')

        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {
                            "scenario_id": "test_scenario",
                            "run_index": 0,
                            "passed": False,
                            "error": None,
                            "scores": {
                                "state": {"appointments_created": False, "patient_verified": True},
                                "trace": {"must_call:book_appointment": False, "call_order": False},
                                "judge": {"test_check": True},
                                "failures": [
                                    "[FAIL] appointments_created: Expected 1, got 0",
                                    "[FAIL] must_call:book_appointment: Missing in trace",
                                    "[FAIL] call_order: Expected order: [...], actual calls: [...]"
                                ]
                            }
                        }
                    ]
                }
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)

        # Real-format detail under the right sections
        state_section = prompt.split("**State check failures:**")[1].split("**Trace check failures:**")[0]
        trace_section = prompt.split("**Trace check failures:**")[1].split("**Judge failures:**")[0]
        judge_section = prompt.split("**Judge failures:**")[1].split("**Tool calls:**")[0]
        assert "Expected 1, got 0" in state_section
        assert "appointments_created" in state_section
        assert "Missing in trace" in trace_section
        assert "must_call:book_appointment" in trace_section
        assert "call_order" in trace_section
        # Passing key absent from every failure section
        assert "patient_verified" not in state_section
        assert "test_check" not in judge_section
        assert "test_check" not in state_section
        assert "test_check" not in trace_section

    def test_full_policy_untruncated(self, tmp_path):
        """Late-in-section policy marker appears untruncated."""
        results = {
            "scenarios": [
                {
                    "id": "test",
                    "split": "train",
                    "pass_rate": 0.0,
                    "is_infra_error": False,
                    "transcript": "Test",
                    "opening": "Hello",
                    "patient_persona": "Test",
                    "tool_trace": [],
                    "state_check_failures": ["fail"],
                    "trace_check_failures": [],
                    "judge_failures": [],
                    "current_policy_sections": {}
                }
            ],
            "current_policy": {"tone": {"text": "Be polite and concise. THIS_MARKER_SHOULD_APPEAR_UNTRUNCATED"}}
        }

        policy = {"tone": {"text": "Be polite and concise. THIS_MARKER_SHOULD_APPEAR_UNTRUNCATED"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)

        assert "THIS_MARKER_SHOULD_APPEAR_UNTRUNCATED" in prompt
        assert "..." not in prompt or "THIS_MARKER_SHOULD_APPEAR_UNTRUNCATED" not in prompt.split("...")[0] if "..." in prompt else True

    def test_k3_selects_failing_run(self, tmp_path):
        """k=3: run 0 fails with marker, run 1 passes; prompt contains run 0 transcript not run 1."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        trace_file = traces_dir / "test_scenario.jsonl"
        # 3 runs: run 0 fails, run 1 passes, run 2 passes
        trace_file.write_text(
            '{"event": "patient_turn", "response": "RUN_ZERO_MARKER"}\n'
            '{"event": "agent_turn", "response": "test"}\n'
            '{"event": "conversation_end"}\n'
            '{"event": "patient_turn", "response": "run 1 passing"}\n'
            '{"event": "agent_turn", "response": "test"}\n'
            '{"event": "conversation_end"}\n'
            '{"event": "patient_turn", "response": "run 2"}\n'
            '{"event": "agent_turn", "response": "test"}\n'
            '{"event": "conversation_end"}'
        )

        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.66,  # 2/3 passed
                    "runs": [
                        {
                            "scenario_id": "test_scenario",
                            "run_index": 0,
                            "passed": False,
                            "error": None,
                            "scores": {"failures": ["[FAIL] state: failed"]}
                        },
                        {
                            "scenario_id": "test_scenario",
                            "run_index": 1,
                            "passed": True,
                            "error": None,
                            "scores": {"failures": []}
                        },
                        {
                            "scenario_id": "test_scenario",
                            "run_index": 2,
                            "passed": True,
                            "error": None,
                            "scores": {"failures": []}
                        }
                    ]
                }
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        # Should select run 0 (the first failing one)
        assert "RUN_ZERO_MARKER" in input_data.failures[0].transcript
        assert "run 1 passing" not in input_data.failures[0].transcript

    def test_segment_count_mismatch_raises(self, tmp_path):
        """Segment count not equal to run count raises ReflectorError."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        trace_file = traces_dir / "test_scenario.jsonl"
        # Only 1 segment but 3 runs
        trace_file.write_text('{"event": "patient_turn", "response": "test"}\n{"event": "conversation_end"}')

        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {"scenario_id": "test_scenario", "run_index": 0, "passed": False, "error": None, "scores": {"failures": []}},
                        {"scenario_id": "test_scenario", "run_index": 1, "passed": False, "error": None, "scores": {"failures": []}},
                        {"scenario_id": "test_scenario", "run_index": 2, "passed": False, "error": None, "scores": {"failures": []}}
                    ]
                }
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        from clinic_agent.loop.reflector import ReflectorError
        with pytest.raises(ReflectorError) as exc_info:
            build_reflector_input(results, tmp_path, policy)
        assert "segment count" in str(exc_info.value).lower()

    def test_no_failing_run_raises(self, tmp_path, caplog):
        """All-error scenario: excluded (no raise for that case alone when mixed handled elsewhere).

        Human-approved retarget: assert exclude + warning with scenario id, no raise
        from the exclude path itself when combined with the clean 'no failures' raise
        only when every failing scenario is all-error (see test_all_error_only_raises_no_failures).
        Here we still have a real failing scenario so exclude is silent aside from warning.
        """
        import logging
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "real_fail.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"KEEP_ME"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        # Intentionally no trace for test_scenario — must never be opened

        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {
                            "scenario_id": "test_scenario",
                            "run_index": 0,
                            "passed": False,
                            "error": "infra error",
                            "scores": {"failures": []},
                        }
                    ],
                },
                "real_fail": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{
                        "run_index": 0,
                        "passed": False,
                        "error": None,
                        "scores": {
                            "state": {"x": False},
                            "failures": ["[FAIL] x: bad"],
                        },
                    }],
                },
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        with caplog.at_level(logging.WARNING):
            input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "test_scenario" not in prompt
        assert "KEEP_ME" in prompt
        assert any(
            "test_scenario" in r.message and "all-error" in r.message.lower()
            for r in caplog.records
        )

    def test_all_error_only_raises_no_failures(self, tmp_path, caplog):
        """If every failing train scenario is all-error → ReflectorError('no failures')."""
        import logging
        results = {
            "per_scenario": {
                "all_err_a": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {"run_index": 0, "passed": False, "error": "infra a", "scores": None},
                    ],
                },
                "all_err_b": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {"run_index": 0, "passed": False, "error": "infra b", "scores": None},
                        {"run_index": 1, "passed": False, "error": "infra b2", "scores": None},
                    ],
                },
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ReflectorError) as exc_info:
                build_reflector_input(results, tmp_path, policy)
        assert "no failures" in str(exc_info.value).lower()
        assert any("all_err_a" in r.message for r in caplog.records)
        assert any("all_err_b" in r.message for r in caplog.records)

    def test_all_error_k1_excluded_no_raise_no_trace(self, tmp_path):
        """k=1, only run errored: excluded; with only all-error failures raise 'no failures'."""
        results = {
            "per_scenario": {
                "all_err": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {"run_index": 0, "passed": False, "error": "infra boom", "scores": None},
                    ],
                }
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        # No traces/ directory at all — must not be opened
        with pytest.raises(ReflectorError) as exc_info:
            build_reflector_input(results, tmp_path, policy)
        assert "no failures" in str(exc_info.value).lower()

    def test_k3_error_fail_pass_selects_fail_run(self, tmp_path):
        """k=3: run0 errored (ERR_RUN_MARK), run1 failed, run2 passed (PASS_RUN_MARK)."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "mix_scen.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"ERR_RUN_MARK"}\n'
            '{"event":"conversation_end"}\n'
            '{"event":"patient_turn","turn":1,"response":"FAIL_RUN_ONLY"}\n'
            '{"event":"agent_turn","turn":1,"response":"fail agent"}\n'
            '{"event":"conversation_end"}\n'
            '{"event":"patient_turn","turn":1,"response":"PASS_RUN_MARK"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = {
            "per_scenario": {
                "mix_scen": {
                    "split": "train",
                    "pass_rate": 0.33,
                    "runs": [
                        {"run_index": 0, "passed": False, "error": "agent boom", "scores": None},
                        {
                            "run_index": 1,
                            "passed": False,
                            "error": None,
                            "scores": {
                                "state": {"appointments_created": False},
                                "trace": {},
                                "judge": {},
                                "failures": ["[FAIL] appointments_created: Expected 1, got 0"],
                            },
                        },
                        {"run_index": 2, "passed": True, "error": None, "scores": {"failures": []}},
                    ],
                }
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "FAIL_RUN_ONLY" in prompt
        assert "ERR_RUN_MARK" not in prompt
        assert "PASS_RUN_MARK" not in prompt

    def test_mix_all_error_and_real_failing(self, tmp_path):
        """All-error train scenario excluded; real failing train scenario still appears."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "real_fail.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"REAL_FAIL_TEXT"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = {
            "per_scenario": {
                "all_err": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": "quota", "scores": None}],
                },
                "real_fail": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{
                        "run_index": 0,
                        "passed": False,
                        "error": None,
                        "scores": {
                            "state": {"x": False},
                            "failures": ["[FAIL] x: bad"],
                        },
                    }],
                },
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "REAL_FAIL_TEXT" in prompt
        assert "real_fail" in prompt
        assert "all_err" not in prompt

    def test_inconsistent_nonerror_none_failing_raises(self, tmp_path):
        """pass_rate < 1 with non-error runs but none failing → ReflectorError + id."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "weird.jsonl").write_text(
            '{"event":"patient_turn","response":"x"}\n{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = {
            "per_scenario": {
                "weird": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {"run_index": 0, "passed": True, "error": None, "scores": {"failures": []}},
                    ],
                }
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        with pytest.raises(ReflectorError) as exc_info:
            build_reflector_input(results, tmp_path, policy)
        assert "weird" in str(exc_info.value)
        assert "no failing" in str(exc_info.value).lower()

    def test_missing_trace_file_raises(self, tmp_path):
        """Missing trace file raises clear error."""
        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"scenario_id": "test_scenario", "run_index": 0, "passed": False, "error": None, "scores": {"failures": []}}]
                }
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        from clinic_agent.loop.reflector import ReflectorError
        with pytest.raises(ReflectorError) as exc_info:
            build_reflector_input(results, tmp_path, policy)
        assert "trace file not found" in str(exc_info.value).lower()

    def test_transcript_capped_and_collapsed(self, tmp_path):
        """Long transcript keeps last 30 entries after collapse; tool calls appear."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        trace_file = traces_dir / "test_scenario.jsonl"

        events = []
        for i in range(1, 61):
            role = "patient_turn" if i % 2 == 1 else "agent_turn"
            events.append(
                f'{{"event": "{role}", "turn": {i}, "response": "T{i:02d}"}}'
            )
        events.append('{"event": "tool_call", "tool": "test_tool", "args": {"x": 1}, "result": "ok"}')
        events.append('{"event": "conversation_end"}')
        trace_file.write_text("\n".join(events))

        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [
                        {
                            "scenario_id": "test_scenario",
                            "run_index": 0,
                            "passed": False,
                            "error": None,
                            "scores": {"failures": []}
                        }
                    ]
                }
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)

        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)

        assert "T60" in prompt
        assert "T01" not in prompt
        assert "earlier entries omitted" in prompt
        assert "test_tool" in prompt

    def test_k3_failing_run1_not_pass_marker(self, tmp_path):
        """k=3: run 0 passes (PASS_RUN_MARK), run 1 fails — prompt has run 1, not marker."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "fail_scen.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"PASS_RUN_MARK"}\n'
            '{"event":"agent_turn","turn":1,"response":"ok"}\n'
            '{"event":"conversation_end"}\n'
            '{"event":"patient_turn","turn":1,"response":"FAIL_RUN_TEXT"}\n'
            '{"event":"agent_turn","turn":1,"response":"nok"}\n'
            '{"event":"conversation_end"}\n'
            '{"event":"patient_turn","turn":1,"response":"run2"}\n'
            '{"event":"agent_turn","turn":1,"response":"x"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = {
            "per_scenario": {
                "fail_scen": {
                    "split": "train",
                    "pass_rate": 0.66,
                    "runs": [
                        {"run_index": 0, "passed": True, "error": None, "scores": {"state": {}, "trace": {}, "judge": {}, "failures": []}},
                        {"run_index": 1, "passed": False, "error": None, "scores": {"state": {"x": False}, "trace": {}, "judge": {}, "failures": ["[FAIL] x: bad"]}},
                        {"run_index": 2, "passed": True, "error": None, "scores": {"state": {}, "trace": {}, "judge": {}, "failures": []}},
                    ],
                },
                "heldout_ok": {
                    "split": "heldout",
                    "pass_rate": 1.0,
                    "runs": [{"run_index": 0, "passed": True, "error": None}],
                },
                "train_pass": {
                    "split": "train",
                    "pass_rate": 1.0,
                    "runs": [{"run_index": 0, "passed": True, "error": None}],
                },
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "FAIL_RUN_TEXT" in prompt
        assert "PASS_RUN_MARK" not in prompt

    def test_heldout_trace_mark_never_in_prompt(self, tmp_path):
        """Heldout with HELDOUT_TRACE_MARK in its trace file never reaches the prompt."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "train_fail.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"train only"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        (traces_dir / "are_you_a_robot.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"HELDOUT_TRACE_MARK"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = {
            "per_scenario": {
                "train_fail": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": None, "scores": {"state": {"a": False}, "failures": ["[FAIL] a: x"]}}],
                },
                "are_you_a_robot": {
                    "split": "heldout",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": None, "scores": {"failures": []}}],
                },
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "HELDOUT_TRACE_MARK" not in prompt
        assert "train only" in prompt

    def test_real_format_heldout_id_in_policy_raises(self, tmp_path):
        """Real per_scenario path: heldout id in policy text → HeldoutLeakError.

        Exercises _check_for_heldout_leak(..., all_scenarios_meta) (R3 target).
        """
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "train_fail.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"clean train"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        heldout_id = HELDOUT_SCENARIOS[0] if HELDOUT_SCENARIOS else "are_you_a_robot"
        results = {
            "per_scenario": {
                "train_fail": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{
                        "run_index": 0,
                        "passed": False,
                        "error": None,
                        "scores": {
                            "state": {"a": False},
                            "failures": ["[FAIL] a: x"],
                        },
                    }],
                },
                heldout_id: {
                    "split": "heldout",
                    "pass_rate": 1.0,
                    "runs": [{"run_index": 0, "passed": True, "error": None, "scores": {}}],
                },
            }
        }
        # Heldout id only appears via policy → leak check on real path must fire
        policy = {"tone": {"text": f"Never mention {heldout_id} in responses."}}
        with pytest.raises(HeldoutLeakError):
            build_reflector_input(results, tmp_path, policy)

    def test_missing_heldout_trace_does_not_raise(self, tmp_path):
        """Filter-first: missing heldout/passing traces must not raise (R9 regression)."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        (traces_dir / "train_fail.jsonl").write_text(
            '{"event":"patient_turn","turn":1,"response":"train only"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        # No traces for heldout or passing train
        results = {
            "per_scenario": {
                "train_fail": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{
                        "run_index": 0,
                        "passed": False,
                        "error": None,
                        "scores": {"state": {"a": False}, "failures": ["[FAIL] a: x"]},
                    }],
                },
                "happy_path_book": {
                    "split": "train",
                    "pass_rate": 1.0,
                    "runs": [{"run_index": 0, "passed": True, "error": None, "scores": {}}],
                },
                "are_you_a_robot": {
                    "split": "heldout",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": None, "scores": {}}],
                },
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "train only" in prompt
        assert "are_you_a_robot" not in prompt

    def test_sixty_distinct_turns_cap(self, tmp_path):
        """60 distinct turns T01..T60: prompt has T60, lacks T01, omitted marker, <=30 entries."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        events = []
        for i in range(1, 61):
            role = "patient_turn" if i % 2 == 1 else "agent_turn"
            events.append(f'{{"event":"{role}","turn":{i},"response":"T{i:02d}"}}')
        events.append('{"event":"conversation_end"}')
        (traces_dir / "cap_scen.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
        results = {
            "per_scenario": {
                "cap_scen": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": None, "scores": {"failures": []}}],
                }
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "T60" in prompt
        assert "T01" not in prompt
        assert "earlier entries omitted" in prompt
        transcript = prompt.split("**Transcript:**")[1].strip()
        entry_lines = [
            ln for ln in transcript.split("\n")
            if ln.startswith("[turn ") or ln.startswith("Patient:") or ln.startswith("Agent:")
        ]
        assert len(entry_lines) <= 30

    def test_twelve_identical_pairs_collapse(self, tmp_path):
        """12 identical patient/agent pairs -> one Agent, one Patient, repeated 11 more times."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        events = []
        for _ in range(12):
            events.append('{"event":"patient_turn","turn":1,"response":"Same patient"}')
            events.append('{"event":"agent_turn","turn":1,"response":"Same agent"}')
        events.append('{"event":"conversation_end"}')
        (traces_dir / "pair_scen.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
        results = {
            "per_scenario": {
                "pair_scen": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": None, "scores": {"failures": []}}],
                }
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert prompt.count("Same patient") == 1
        assert prompt.count("Same agent") == 1
        assert "repeated 11 more times" in prompt

    def test_newline_in_message_counts_as_one_entry(self, tmp_path):
        """A message containing a newline still counts as one entry."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        line = json.dumps({
            "event": "patient_turn",
            "turn": 3,
            "response": "line one\nline two",
        })
        (traces_dir / "nl_scen.jsonl").write_text(
            line + '\n{"event":"agent_turn","turn":3,"response":"ack"}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = {
            "per_scenario": {
                "nl_scen": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"run_index": 0, "passed": False, "error": None, "scores": {"failures": []}}],
                }
            }
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        assert input_data.failures[0].transcript.count("[turn 3]") == 2
        assert "line one" in input_data.failures[0].transcript
        assert "line two" in input_data.failures[0].transcript

    def test_reflector_md_protected_sections(self):
        """reflector.md names the four protected sections and add/edit rules."""
        text = (
            Path(__file__).resolve().parents[1]
            / "clinic_agent" / "loop" / "prompts" / "reflector.md"
        ).read_text(encoding="utf-8")
        for section in ("identity", "confirmation", "emergency", "injection"):
            assert section in text
        assert "PROTECTED" in text
        assert "appended" in text.lower() or "ONLY the new sentence" in text
        assert "full replacement" in text.lower()
        assert "changes[].text" in text

    def test_caught_by_layer_state_first(self, tmp_path):
        """Caught-by reports state when state fails (trust order)."""
        results = {
            "scenarios": [
                {
                    "id": "failing_scenario",
                    "split": "train",
                    "pass_rate": 0.0,
                    "is_infra_error": False,
                    "transcript": "Patient: hi\nAgent: hello",
                    "opening": "Hello",
                    "patient_persona": "Test",
                    "tool_trace": [],
                    "state_check_failures": ["state:x"],
                    "trace_check_failures": ["trace:y"],
                    "judge_failures": ["judge:z"],
                    "current_policy_sections": {},
                }
            ],
            "current_policy": {"tone": {"text": "Be polite"}},
        }
        policy = {"tone": {"text": "Be polite"}}
        input_data = build_reflector_input(results, tmp_path, policy)
        from clinic_agent.loop.reflector import _build_prompt
        prompt = _build_prompt(input_data.failures, input_data.current_policy)
        assert "Caught by (trust order state > trace > judge):** state" in prompt

    def test_run_dir_as_string(self, tmp_path):
        """run_dir as str works."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        trace_file = traces_dir / "test_scenario.jsonl"
        trace_file.write_text('{"event": "patient_turn", "response": "test"}\n{"event": "conversation_end"}')

        results = {
            "per_scenario": {
                "test_scenario": {
                    "split": "train",
                    "pass_rate": 0.0,
                    "runs": [{"scenario_id": "test_scenario", "run_index": 0, "passed": False, "error": None, "scores": {"failures": []}}]
                }
            }
        }

        policy = {"tone": {"text": "Be polite"}}
        # Pass run_dir as string, not Path
        input_data = build_reflector_input(results, str(tmp_path), policy)

        assert len(input_data.failures) == 1


class TestReflect:
    """Test reflect function with injected LLM."""

    def test_valid_patch_returned(self):
        """A valid JSON response should return a Patch."""
        input_data = ReflectorInput(
            failures=[
                FailureReport(
                    scenario_id="test",
                    split="train",
                    transcript="Test",
                    opening="Hi",
                    patient_persona="Test",
                    tool_trace=[],
                    state_check_failures=["fail"],
                    trace_check_failures=[],
                    judge_failures=["fail"],
                    current_policy_sections={"tone": "Be nice"},
                    pass_rate=0.5
                )
            ],
            current_policy={"tone": {"text": "Be nice"}}
        )

        def fake_llm(system: str, message: str) -> str:
            return """{
                "failure_category": "other",
                "evidence": [{"scenario_id": "test", "turn": 1, "observation": "test"}],
                "root_cause": "Test",
                "changes": [{"policy_section": "tone", "op": "edit", "text": "Be very nice"}],
                "expected_to_fix": ["test"],
                "regression_risk": "Low"
            }"""

        patch = reflect(input_data, fake_llm)
        assert isinstance(patch, Patch)
        assert patch.failure_category == FailureCategory.OTHER

    def test_invalid_json_then_valid_succeeds(self):
        """Invalid JSON on first attempt, valid on second should succeed."""
        input_data = ReflectorInput(
            failures=[
                FailureReport(
                    scenario_id="test",
                    split="train",
                    transcript="Test",
                    opening="Hi",
                    patient_persona="Test",
                    tool_trace=[],
                    state_check_failures=["fail"],
                    trace_check_failures=[],
                    judge_failures=["fail"],
                    current_policy_sections={"tone": "Be nice"},
                    pass_rate=0.5
                )
            ],
            current_policy={"tone": {"text": "Be nice"}}
        )

        call_count = [0]

        def fake_llm(system: str, message: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "invalid json {{{"
            else:
                return """{
                    "failure_category": "other",
                    "evidence": [{"scenario_id": "test", "turn": 1, "observation": "test"}],
                    "root_cause": "Test",
                    "changes": [{"policy_section": "tone", "op": "edit", "text": "Be very nice"}],
                    "expected_to_fix": ["test"],
                    "regression_risk": "Low"
                }"""

        patch = reflect(input_data, fake_llm)
        assert isinstance(patch, Patch)
        assert call_count[0] == 2

    def test_invalid_json_twice_fails_cleanly(self):
        """Invalid JSON twice should raise InvalidJSONError."""
        input_data = ReflectorInput(
            failures=[
                FailureReport(
                    scenario_id="test",
                    split="train",
                    transcript="Test",
                    opening="Hi",
                    patient_persona="Test",
                    tool_trace=[],
                    state_check_failures=["fail"],
                    trace_check_failures=[],
                    judge_failures=["fail"],
                    current_policy_sections={"tone": "Be nice"},
                    pass_rate=0.5
                )
            ],
            current_policy={"tone": {"text": "Be nice"}}
        )

        def fake_llm(system: str, message: str) -> str:
            return "invalid json {{{"

        with pytest.raises(InvalidJSONError):
            reflect(input_data, fake_llm)

    def test_no_failures_raises(self):
        """No failures should raise ReflectorError."""
        input_data = ReflectorInput(
            failures=[],
            current_policy={"tone": {"text": "Be nice"}}
        )

        def fake_llm(system: str, message: str) -> str:
            return "{}"

        with pytest.raises(ReflectorError):
            reflect(input_data, fake_llm)
