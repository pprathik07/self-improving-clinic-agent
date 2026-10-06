"""Tests for llm.py abort / rate-limit retry policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clinic_agent.llm import (
    FatalLLMError,
    _RATE_LIMIT_SLEEPS,
    _generate_with_retries,
    _is_immediate_abort_error,
    _is_rate_limit_error,
)


class FakeAPIError(Exception):
    def __init__(self, code: int | None, message: str, details: dict | None = None):
        self.code = code
        self.details = details
        super().__init__(message)


class TestClassify:
    def test_403_by_code(self):
        e = FakeAPIError(403, "denied")
        assert _is_immediate_abort_error(e, str(e))

    def test_404_by_word_boundary(self):
        e = FakeAPIError(None, "error 404 model missing")
        assert _is_immediate_abort_error(e, str(e))

    def test_429_is_rate_limit_not_immediate(self):
        e = FakeAPIError(429, "RESOURCE_EXHAUSTED")
        assert _is_rate_limit_error(e, str(e))
        assert not _is_immediate_abort_error(e, str(e))

    def test_daily_quota_is_immediate(self):
        e = FakeAPIError(None, "You exceeded your daily quota for this project")
        assert _is_immediate_abort_error(e, str(e))
        assert not _is_rate_limit_error(e, str(e))


class TestGenerateWithRetries:
    def test_403_aborts_on_call_1_no_sleep(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(403, "403 PERMISSION_DENIED")

        client = SimpleNamespace(models=Models())
        with pytest.raises(FatalLLMError, match="abort immediately"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert calls["n"] == 1
        assert sleeps == []

    def test_404_aborts_on_call_1(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(404, "404 NOT_FOUND")

        client = SimpleNamespace(models=Models())
        with pytest.raises(FatalLLMError, match="abort immediately"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert calls["n"] == 1
        assert sleeps == []

    def test_429_then_success_default_sleeps(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                if calls["n"] <= 2:
                    raise FakeAPIError(429, "RESOURCE_EXHAUSTED")
                return MagicMock()

        client = SimpleNamespace(models=Models())
        monkeypatch.setattr(
            "clinic_agent.llm._parse_response",
            lambda r: __import__("clinic_agent.llm", fromlist=["LLMResponse"]).LLMResponse(content="ok"),
        )
        result = _generate_with_retries(client, "m", contents=[], config=None)
        assert result.content == "ok"
        assert sleeps == [10, 30]
        assert calls["n"] == 3

    def test_persistent_429_default_sleeps_then_aborts(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(429, "RATE_LIMIT")

        client = SimpleNamespace(models=Models())
        with pytest.raises(FatalLLMError, match="rate limit exhausted"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert sleeps == [10, 30, 60]
        assert calls["n"] == 4
        assert _RATE_LIMIT_SLEEPS == (10, 30, 60)

    def test_429_honours_retry_delay(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise FakeAPIError(
                        429,
                        "RESOURCE_EXHAUSTED",
                        details={
                            "error": {
                                "details": [{
                                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                    "retryDelay": "12s",
                                }]
                            }
                        },
                    )
                return MagicMock()

        client = SimpleNamespace(models=Models())
        monkeypatch.setattr(
            "clinic_agent.llm._parse_response",
            lambda r: __import__("clinic_agent.llm", fromlist=["LLMResponse"]).LLMResponse(content="ok"),
        )
        _generate_with_retries(client, "m", contents=[], config=None)
        assert sleeps == [12.0]

    def test_daily_quota_aborts_with_no_sleep(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(None, "daily quota exceeded for this API key")

        client = SimpleNamespace(models=Models())
        with pytest.raises(FatalLLMError, match="abort immediately"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert calls["n"] == 1
        assert sleeps == []


class TestEvalAbortZeroScored:
    """FatalLLMError aborts on call 1; unrelated RuntimeError with '404' does not."""

    def _setup(self, tmp_path, monkeypatch):
        from pathlib import Path
        import shutil

        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
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
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy(
            Path(__file__).resolve().parents[1] / "policy" / "policy_v1.yaml",
            policy_dir / "policy_v1.yaml",
        )
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "a")
        monkeypatch.setenv("JUDGE_MODEL", "b")
        monkeypatch.setenv("SIM_MODEL", "c")
        monkeypatch.chdir(tmp_path)
        return scenarios_dir

    def test_403_run_eval_zero_scored(self, tmp_path, monkeypatch):
        from clinic_agent.evals.run_eval import run_eval
        from clinic_agent.evals.scorers import ScoreReport

        scenarios_dir = self._setup(tmp_path, monkeypatch)
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise FatalLLMError("LLM API fatal error (abort immediately): 403 PERMISSION_DENIED")

        with patch("clinic_agent.evals.run_eval.run_scenario", side_effect=boom):
            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )
        assert results == {}
        assert calls["n"] == 1

    def test_404_run_eval_zero_scored(self, tmp_path, monkeypatch):
        from clinic_agent.evals.run_eval import run_eval

        scenarios_dir = self._setup(tmp_path, monkeypatch)
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise FatalLLMError("LLM API fatal error (abort immediately): 404 NOT_FOUND")

        with patch("clinic_agent.evals.run_eval.run_scenario", side_effect=boom):
            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )
        assert results == {}
        assert calls["n"] == 1

    def test_unrelated_runtimeerror_with_404_does_not_abort(self, tmp_path, monkeypatch):
        """A plain RuntimeError whose message contains '404' must NOT abort the run."""
        from clinic_agent.evals.run_eval import run_eval
        from clinic_agent.evals.scorers import ScoreReport

        scenarios_dir = self._setup(tmp_path, monkeypatch)
        calls = {"n": 0}

        def side_effect(scenario, *a, **k):
            calls["n"] += 1
            if scenario.id == "first":
                raise RuntimeError("tool failed: unexpected 404 in payload QUOTA text")
            report = ScoreReport()
            return report, {
                "scenario_id": scenario.id,
                "split": "train",
                "tags": [],
                "turns": 1,
                "final_state": "done",
                "passed": True,
                "error": None,
                "scores": report.summary,
            }

        with patch("clinic_agent.evals.run_eval.run_scenario", side_effect=side_effect):
            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )
        # Real mode + infra errors → invalid run cleaned up (return {})
        # But we must have attempted BOTH scenarios (not abort on call 1)
        assert calls["n"] == 2
        # Real mode invalidates when infra errors occurred
        assert results == {}
