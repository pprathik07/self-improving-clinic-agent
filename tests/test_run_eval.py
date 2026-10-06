"""Tests for eval runner error handling and fail-fast behavior."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from clinic_agent.evals.run_eval import run_scenario, run_eval
from clinic_agent.evals.scenario_schema import Scenario
from clinic_agent.evals.scorers import ScoreReport
from clinic_agent.llm import FatalLLMError


class TestAuthErrorFailFast:
    """FatalLLMError aborts the run immediately (class-based, not substring)."""

    def test_auth_error_aborts_run_real_mode(self, tmp_path, monkeypatch):
        """In real mode, FatalLLMError from run_turn aborts the scenario."""
        scenario = Scenario(
            id="test_scenario",
            split="train",
            tags=[],
            seed_db="default",
            patient_persona="Test patient",
            opening="Hello",
            max_turns=5,
            expect={"state": {}, "trace": {}, "judge": []},
        )

        with patch("clinic_agent.evals.run_eval.run_turn") as mock_run:
            mock_run.side_effect = FatalLLMError(
                "LLM API fatal error (abort immediately): 403 PERMISSION_DENIED"
            )

            monkeypatch.setenv("MOCK_LLM", "0")

            with pytest.raises(FatalLLMError) as exc_info:
                run_scenario(
                    scenario,
                    "policy/policy_v1.yaml",
                    trace_dir=tmp_path,
                )

            assert "403" in str(exc_info.value) or "PERMISSION_DENIED" in str(exc_info.value)

    def test_run_eval_aborts_on_auth_error_real_mode(self, tmp_path, monkeypatch):
        """run_eval returns {} and does not write results.json on FatalLLMError."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        with patch("clinic_agent.evals.run_eval.run_scenario") as mock_run:
            mock_run.side_effect = FatalLLMError(
                "LLM API fatal error (abort immediately): 403 PERMISSION_DENIED"
            )

            monkeypatch.setenv("MOCK_LLM", "0")

            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

            assert results == {}


class TestJudgeErrorHandling:
    """Test that judge errors are marked as infrastructure errors, not scenario failures."""

    def test_judge_error_marked_as_infrastructure_error(self, tmp_path):
        """Judge errors should be marked in metadata and excluded from pass rate."""
        scenario = Scenario(
            id="test_scenario",
            split="train",
            tags=[],
            seed_db="default",
            patient_persona="Test patient",
            opening="Hello",
            max_turns=5,
            expect={"state": {}, "trace": {}, "judge": ["Test rubric"]},
        )

        # Mock judge to raise an error
        with patch("clinic_agent.evals.run_eval.judge_transcript") as mock_judge:
            mock_judge.side_effect = RuntimeError("Judge API error")

            report, run_meta = run_scenario(
                scenario,
                "policy/policy_v1.yaml",
                trace_dir=tmp_path,
            )

            # Should have error field set
            assert run_meta.get("error") is not None
            assert "judge" in run_meta["error"]

            # Should not be marked as passed
            assert run_meta.get("passed") is False

            # Scores should be None (not calculated)
            assert run_meta.get("scores") is None


class TestModelValidation:
    """Test model name validation in real mode."""

    def test_requires_different_models_in_real_mode(self, tmp_path, monkeypatch):
        """Real mode should require all three models to be different."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        # Set all models to the same value
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("JUDGE_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("SIM_MODEL", "gemini-2.0-flash")

        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )

        # Should return empty dict (validation failed)
        assert results == {}


class TestFatalAbortOnFirstCall:
    """Fatal 403/404/quota errors abort on call 1; no scenario is scored."""

    @pytest.mark.parametrize(
        "error_msg",
        [
            "403 PERMISSION_DENIED",
            "404 NOT_FOUND model not found",
            "RESOURCE_EXHAUSTED quota exceeded",
        ],
        ids=["403", "404", "quota"],
    )
    def test_fatal_error_aborts_on_call_1(self, tmp_path, monkeypatch, error_msg):
        """run_eval aborts on first fatal error; call_count==1; no results written."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        # Two scenarios — second must never be scored if abort is immediate
        for sid in ("first", "second"):
            (scenarios_dir / f"{sid}.yaml").write_text(f"""
id: {sid}
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {{}}
  trace: {{}}
  judge: []
""")

        call_count = {"n": 0}

        def fake_run_scenario(*args, **kwargs):
            call_count["n"] += 1
            raise FatalLLMError(f"LLM API fatal error (abort immediately): {error_msg}")

        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "model-a")
        monkeypatch.setenv("JUDGE_MODEL", "model-b")
        monkeypatch.setenv("SIM_MODEL", "model-c")

        # Keep policy reachable after chdir
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy(
            Path(__file__).resolve().parents[1] / "policy" / "policy_v1.yaml",
            policy_dir / "policy_v1.yaml",
        )
        monkeypatch.chdir(tmp_path)

        with patch("clinic_agent.evals.run_eval.run_scenario", side_effect=fake_run_scenario):
            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

        assert results == {}
        assert call_count["n"] == 1, f"Expected abort on call 1, got {call_count['n']}"
        # No results.json under runs/
        runs_dir = tmp_path / "runs"
        if runs_dir.exists():
            assert list(runs_dir.rglob("results.json")) == []


class TestRequireRealVsMock:
    """REQUIRE_REAL=1 with MOCK_LLM=1 must abort."""

    def test_require_real_aborts_when_mock_on(self, tmp_path, monkeypatch):
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        (scenarios_dir / "t.yaml").write_text("""
id: t
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("REQUIRE_REAL", "1")
        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )
        assert results == {}

    """Test that mock runs go to runs/_mock/ and real runs go to runs/."""

    def test_mock_runs_go_to_mock_subdirectory(self, tmp_path, monkeypatch):
        """Mock mode should write results to runs/_mock/timestamp/."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        # Copy policy file to tmp_path
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        # Set mock mode
        monkeypatch.setenv("MOCK_LLM", "1")

        # Change to temp directory for this test
        monkeypatch.chdir(tmp_path)

        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )

        # Check that results were written to runs/_mock/
        mock_runs_dir = tmp_path / "runs" / "_mock"
        assert mock_runs_dir.exists()
        assert list(mock_runs_dir.glob("*/results.json"))  # At least one results.json

        # Check that no results were written to runs/ directly
        runs_dir = tmp_path / "runs"
        runs_results = list(runs_dir.glob("*/results.json"))
        for path in runs_results:
            assert "_mock" not in str(path), f"Found results in wrong location: {path}"

    def test_real_runs_go_to_runs_directory(self, tmp_path, monkeypatch):
        """Real mode should write results to runs/timestamp/ (not _mock)."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        # Copy policy file to tmp_path
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        # Set real mode with different models
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "model1")
        monkeypatch.setenv("JUDGE_MODEL", "model2")
        monkeypatch.setenv("SIM_MODEL", "model3")

        # Change to temp directory for this test
        monkeypatch.chdir(tmp_path)

        # Mock run_scenario to avoid actual LLM calls
        with patch("clinic_agent.evals.run_eval.run_scenario") as mock_run:
            mock_report = ScoreReport()
            mock_report.state_checks = []
            mock_report.trace_checks = []
            mock_report.judge_checks = []
            mock_run.return_value = (mock_report, {
                "scenario_id": "test",
                "split": "train",
                "tags": [],
                "turns": 1,
                "final_state": "done",
                "passed": True,
                "error": None,
                "scores": mock_report.summary
            })

            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

            # Check that results were written to runs/ (not _mock)
            runs_dir = tmp_path / "runs"
            assert runs_dir.exists()
            runs_results = list(runs_dir.glob("*/results.json"))
            assert len(runs_results) > 0, "No results.json found in runs/"

            # Verify none are in _mock
            for path in runs_results:
                assert "_mock" not in str(path), f"Found results in _mock: {path}"
