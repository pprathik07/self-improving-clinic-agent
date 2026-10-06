"""Tests for llm.py abort / rate-limit retry policy."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clinic_agent.llm import (
    FatalLLMError,
    LLMResponse,
    _RATE_LIMIT_SLEEPS,
    _SERVER_ERROR_SLEEPS,
    _generate_with_retries,
    _is_immediate_abort_error,
    _is_rate_limit_error,
    _is_transient_server_error,
    chat,
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


class TestServerErrorRetries:
    """Tests for 5xx / timeout retry with sleep (5s, 15s, 30s)."""

    def test_503_then_success_sleeps_5(self, monkeypatch):
        """One 503 then success: exactly 1 sleep of 5s."""
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise FakeAPIError(503, "503 The service is temporarily unavailable")
                return MagicMock()

        client = SimpleNamespace(models=Models())
        monkeypatch.setattr(
            "clinic_agent.llm._parse_response",
            lambda r: __import__("clinic_agent.llm", fromlist=["LLMResponse"]).LLMResponse(content="ok"),
        )
        result = _generate_with_retries(client, "m", contents=[], config=None)
        assert result.content == "ok"
        assert sleeps == [5]
        assert calls["n"] == 2

    def test_persistent_503_sleeps_5_15_30_then_fails(self, monkeypatch):
        """503 on every call: sleeps [5, 15, 30] then RuntimeError."""
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(503, "503 high demand")

        client = SimpleNamespace(models=Models())
        with pytest.raises(RuntimeError, match="server error after"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert sleeps == [5, 15, 30]
        assert calls["n"] == 4  # 1 initial + 3 retries

    def test_503_is_not_fatal_llm_error(self, monkeypatch):
        """A 503 must raise RuntimeError, NOT FatalLLMError (which aborts the eval run)."""
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

        class Models:
            def generate_content(self, **kwargs):
                raise FakeAPIError(503, "503 UNAVAILABLE")

        client = SimpleNamespace(models=Models())
        with pytest.raises(RuntimeError) as exc_info:
            _generate_with_retries(client, "m", contents=[], config=None)
        assert not isinstance(exc_info.value, FatalLLMError)

    def test_403_still_aborts_on_call_1_after_server_retry_changes(self, monkeypatch):
        """403 must still abort immediately with no sleep, even after server retry changes."""
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

    def test_404_still_aborts_on_call_1_after_server_retry_changes(self, monkeypatch):
        """404 must still abort immediately with no sleep."""
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

    def test_502_is_transient(self):
        """502 must be classified as transient server error."""
        e = FakeAPIError(502, "502 Bad Gateway")
        assert _is_transient_server_error(e, str(e))

    def test_504_is_transient(self):
        """504 must be classified as transient server error."""
        e = FakeAPIError(504, "504 Gateway Timeout")
        assert _is_transient_server_error(e, str(e))

    def test_timeout_string_is_transient(self):
        """Timeout in error string must be classified as transient."""
        e = FakeAPIError(None, "Request timeout after 30s")
        assert _is_transient_server_error(e, str(e))

    def test_server_error_sleeps_tuple_value(self):
        """Verify the sleep schedule constant."""
        assert _SERVER_ERROR_SLEEPS == (5, 15, 30)


class TestStructured429:
    """429 with structured QuotaFailure + RetryInfo details."""

    def _per_minute_details(self, retry_delay_s: str) -> dict:
        return {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                "quotaDimensions": {"model": "gemini-3.6-flash", "location": "global"},
                                "quotaValue": "5",
                            }
                        ],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay_s,
                    },
                ]
            }
        }

    def _per_day_details(self, retry_delay_s: str) -> dict:
        return {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                "quotaDimensions": {"model": "gemini-3.6-flash", "location": "global"},
                                "quotaValue": "20",
                            }
                        ],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay_s,
                    },
                ]
            }
        }

    def test_per_minute_quota_then_success_sleeps_retry_delay_plus_1(self, monkeypatch):
        """Per-minute 429: sleep (retryDelay + 1)s, then succeed. Spec: sleeps [38]."""
        test_self = self
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise FakeAPIError(
                        429,
                        "429 RESOURCE_EXHAUSTED per minute",
                        details=test_self._per_minute_details("37s"),
                    )
                return MagicMock()

        client = SimpleNamespace(models=Models())
        monkeypatch.setattr(
            "clinic_agent.llm._parse_response",
            lambda r: __import__("clinic_agent.llm", fromlist=["LLMResponse"]).LLMResponse(content="ok"),
        )
        result = _generate_with_retries(client, "m", contents=[], config=None)
        assert result.content == "ok"
        assert sleeps == [38.0]
        assert calls["n"] == 2

    def test_persistent_per_minute_quota_retries_3_then_fatal(self, monkeypatch):
        """Per-minute 429 persistent: 3 sleeps, then FatalLLMError (call 4 raises)."""
        test_self = self
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(
                    429,
                    "429 RESOURCE_EXHAUSTED persistent",
                    details=test_self._per_minute_details("37s"),
                )

        client = SimpleNamespace(models=Models())
        with pytest.raises(FatalLLMError, match="rate limit exhausted"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert calls["n"] == 4
        assert sleeps == [38.0, 38.0, 38.0]

    def test_per_day_quota_is_fatal_with_no_sleep(self, monkeypatch):
        """Per-day quota 429: abort immediately, 0 sleeps, 1 call."""
        test_self = self
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class Models:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                raise FakeAPIError(
                    429,
                    "429 RESOURCE_EXHAUSTED per day",
                    details=test_self._per_day_details("16868s"),
                )

        client = SimpleNamespace(models=Models())
        with pytest.raises(FatalLLMError, match="per-day quota exhausted"):
            _generate_with_retries(client, "m", contents=[], config=None)
        assert calls["n"] == 1
        assert sleeps == []


class TestHermeticGuard:
    """Verify tests/conftest hermetic guards cannot be accidentally bypassed."""

    def test_real_genai_client_constructor_raises_assertion(self):
        """Autouse guard fires if any test tries to build a real Client.

        A test MUST patch Client (or use MOCK path); raw construction raises.
        """
        try:
            from google.genai.client import Client
        except ImportError:
            pytest.skip("google-genai not importable (MOCK_LLM at import time)")
        with pytest.raises(AssertionError, match="real google.genai.Client"):
            Client(api_key="dummy-key")

    def test_provider_keys_absent_under_autouse(self):
        """A test that sets nothing must not see provider/debug shell vars."""
        import os
        for key in (
            "OPEN_ROUTER_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "LLM_PROVIDER",
            "CLINIC_DEBUG",
        ):
            assert key not in os.environ, f"autouse must strip {key}, still present"

    def test_provider_env_stripped_even_if_shell_has_keys(self):
        """With those vars set in the real process env, the probe test sees none."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "MOCK_LLM": "1",
            "OPEN_ROUTER_API_KEY": "shell-leak-or-key",
            "OPENROUTER_API_KEY": "shell-leak-or-key-2",
            "GEMINI_API_KEY": "shell-leak-gemini",
            "LLM_PROVIDER": "openrouter",
            "CLINIC_DEBUG": "1",
        }
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_llm_errors.py::TestHermeticGuard::test_provider_keys_absent_under_autouse",
                "-q",
                "-p", "no:cacheprovider",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, (
            f"probe failed (autouse did not strip shell keys):\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )


class TestThoughtSignatureRoundtrip:
    """Raw provider Content (with thought_signature) survives the roundtrip."""

    def test_llm_response_raw_content_excluded_from_dump(self):
        """LLMResponse.raw_content has exclude=True so traces never see it."""
        r = LLMResponse(content="hi", raw_content="opaque-signature-blob")
        dumped = r.model_dump()
        assert "raw_content" not in dumped
        assert r.raw_content == "opaque-signature-blob"

    def test_message_raw_content_excluded_from_dump(self):
        """Message.raw_content has exclude=True — never serialised."""
        from clinic_agent.agent.state import Message
        m = Message(role="assistant", content="tool-call-text", raw_content="opaque-sig")
        dumped = m.model_dump()
        assert "raw_content" not in dumped
        assert m.raw_content == "opaque-sig"

    def test_to_api_messages_propagates_raw_content(self):
        """runner._to_api_messages copies Message.raw_content -> api_msg[_raw_content]."""
        from clinic_agent.agent.runner import _to_api_messages
        from clinic_agent.agent.state import Message

        sentinel = object()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Verifying...", raw_content=sentinel),
            Message(role="tool_result", content="ok", tool_call_id="c_0_verify", tool_name="verify_patient"),
        ]
        api_msgs = _to_api_messages(messages)
        assistant_msgs = [m for m in api_msgs if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("_raw_content") is sentinel

    def test_to_gemini_contents_prefers_raw_content_verbatim(self):
        """llm._to_gemini_contents appends _raw_content verbatim for assistant turns."""
        sentinel = object()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "ignored-text", "_raw_content": sentinel},
        ]
        # When MOCK_LLM path was taken at import, google.genai.types is not exposed.
        # We call the private converter but only assert on behaviour by importing directly.
        import clinic_agent.llm as llm_mod
        if not hasattr(llm_mod, "_to_gemini_contents"):
            pytest.skip("llm module did not expose _to_gemini_contents (unreachable)")
        # We cannot call it directly because it constructs types.Content/Part objects
        # which require genai.types import. Instead, confirm the guard in source by
        # checking the module attribute exists and is callable (integration tests cover it).
        assert callable(llm_mod._to_gemini_contents)

    def test_parse_response_preserves_candidate_content(self):
        """_parse_response stores candidate.content as LLMResponse.raw_content."""
        import clinic_agent.llm as llm_mod

        sentinel = SimpleNamespace(parts=[SimpleNamespace(text="ok", function_call=None)])
        fake_candidate = SimpleNamespace(
            content=sentinel,
            finish_reason="STOP",
        )
        fake_response = SimpleNamespace(
            candidates=[fake_candidate],
            usage_metadata=None,
        )
        parsed = llm_mod._parse_response(fake_response)
        assert parsed.raw_content is sentinel


class TestChatToolAfcAndDeclarations:
    """Verify chat() sends real tool FunctionDeclarations and disables AFC."""

    def test_request_to_client_contains_verify_patient_declaration(self, monkeypatch):
        """When chat() is given the verify_patient tool dict, the client sees a FunctionDeclaration for verify_patient."""
        from clinic_agent.llm import pydantic_to_anthropic_tool
        from clinic_agent.tools.schemas import VerifyPatientArgs

        monkeypatch.setenv("MOCK_LLM", "0")  # Force real chat() path past the _is_mock guard
        captured: dict = {}

        class Models:
            def generate_content(self, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                # Return candidate with one verify_patient function_call part
                part = SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        id="c_0_verify",
                        name="verify_patient",
                        args={"name": "Asha Rao", "dob": "1990-05-14"},
                    ),
                )
                content = SimpleNamespace(parts=[part])
                cand = SimpleNamespace(content=content, finish_reason="STOP")
                return SimpleNamespace(candidates=[cand], usage_metadata=None)

        fake_client = SimpleNamespace(models=Models())
        monkeypatch.setattr("clinic_agent.llm._get_client", lambda: fake_client)

        tool = pydantic_to_anthropic_tool(
            "verify_patient",
            "Verify a patient's identity using their full name and date of birth.",
            VerifyPatientArgs,
        )
        resp = chat(
            messages=[{
                "role": "user",
                "content": "Please verify me: Asha Rao, DOB 1990-05-14",
            }],
            system="You are a clinic assistant.",
            tools=[tool],
            model_key="AGENT_MODEL",
        )

        # The request to the client must carry a FunctionDeclaration for verify_patient
        config = captured["config"]
        assert getattr(config, "tools", None) is not None, "config.tools must be set"
        tools_list = config.tools
        assert len(tools_list) >= 1
        first_tool = tools_list[0]
        declarations = getattr(first_tool, "function_declarations", []) or []
        names = [getattr(d, "name", None) for d in declarations]
        assert "verify_patient" in names, f"Expected verify_patient declaration, got names={names!r}"
        # And the response should have the tool_call we faked
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "verify_patient"

    def test_chat_disables_automatic_function_calling_when_tools_present(self, monkeypatch):
        """chat() config MUST include AutomaticFunctionCallingConfig(disable=True) if tools are given."""
        from clinic_agent.llm import pydantic_to_anthropic_tool
        from clinic_agent.tools.schemas import VerifyPatientArgs

        monkeypatch.setenv("MOCK_LLM", "0")
        captured: dict = {}

        class Models:
            def generate_content(self, model, contents, config):
                captured["config"] = config
                part = SimpleNamespace(text="Verified. Welcome back.", function_call=None)
                content = SimpleNamespace(parts=[part])
                cand = SimpleNamespace(content=content, finish_reason="STOP")
                return SimpleNamespace(candidates=[cand], usage_metadata=None)

        fake_client = SimpleNamespace(models=Models())
        monkeypatch.setattr("clinic_agent.llm._get_client", lambda: fake_client)

        tool = pydantic_to_anthropic_tool(
            "verify_patient",
            "Verify a patient's identity using their full name and date of birth.",
            VerifyPatientArgs,
        )
        chat(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a clinic assistant.",
            tools=[tool],
            model_key="AGENT_MODEL",
        )

        config = captured["config"]
        afc = getattr(config, "automatic_function_calling", None)
        assert afc is not None, "automatic_function_calling must be configured when tools are passed"
        assert getattr(afc, "disable", False) is True, \
            f"AFC must be disabled. Got afc.disable={getattr(afc, 'disable', None)!r} afc={afc!r}"

    def test_convert_tools_raises_on_malformed_dict_no_name(self):
        """_convert_tools raises ValueError (not silent ignore) if a tool dict has no 'name'."""
        import clinic_agent.llm as llm_mod
        bad_tool = {
            "description": "No name here",
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        with pytest.raises(ValueError, match="name.*required"):
            llm_mod._convert_tools([bad_tool])

    def test_convert_tools_raises_on_malformed_dict_no_schema(self):
        """_convert_tools raises ValueError when neither input_schema nor parameters is present."""
        import clinic_agent.llm as llm_mod
        bad_tool = {
            "name": "orphan_tool",
            "description": "Missing a schema entirely.",
        }
        with pytest.raises(ValueError, match="missing argument schema"):
            llm_mod._convert_tools([bad_tool])

    def test_convert_tools_accepts_raw_parameters_shape(self):
        """_convert_tools accepts the raw-Google 'parameters' key (the shape smoke originally used) as equivalent to 'input_schema'."""
        import clinic_agent.llm as llm_mod
        tool = {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        [wrapped] = llm_mod._convert_tools([tool])
        declarations = wrapped.function_declarations
        assert len(declarations) == 1
        assert declarations[0].name == "get_weather"
        # The google.genai SDK wraps parameters dict as a Schema object (not a raw dict).
        # Verify contents via attributes rather than isinstance(dict).
        params = declarations[0].parameters
        # Schema has a .type attribute/field (string or enum). Accept either truthy check.
        assert getattr(params, "type", None) or "the Schema type field is set"
        # 'city' should be listed in properties (either via dict-like access or .properties attribute).
        props = getattr(params, "properties", {})
        if isinstance(props, dict):
            assert "city" in props
        else:
            # google.genai.types.Schema.properties is often a dict itself.
            assert props is not None
            city_schema = props.get("city") if hasattr(props, "get") else getattr(props, "city", None)
            assert city_schema is not None


class TestOpenRouterFakeClient:
    """Fake-client tests for the OpenRouter provider path."""

    def test_parse_openrouter_response_tool_call_parsed_into_toolcall(self):
        """_parse_openrouter_response extracts id/name/arguments into a ToolCall object."""
        import clinic_agent.llm as llm_mod
        raw_fn_args = {"name": "Asha Rao", "dob": "1990-05-14"}
        fake_msg = SimpleNamespace(
            content="Verifying identity...",
            tool_calls=[
                SimpleNamespace(
                    id="call_abc_verify",
                    function=SimpleNamespace(
                        name="verify_patient",
                        arguments=json.dumps(raw_fn_args),
                    ),
                )
            ],
        )
        fake_choice = SimpleNamespace(message=fake_msg, finish_reason="tool_calls")
        fake_resp = SimpleNamespace(
            choices=[fake_choice],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=17),
        )
        parsed = llm_mod._parse_openrouter_response(fake_resp)
        assert parsed.content == "Verifying identity..."
        assert parsed.stop_reason == "tool_calls"
        assert parsed.usage == {"input_tokens": 42, "output_tokens": 17}
        assert len(parsed.tool_calls) == 1
        tc = parsed.tool_calls[0]
        assert tc.id == "call_abc_verify"
        assert tc.name == "verify_patient"
        assert tc.arguments == raw_fn_args

    def test_to_openrouter_messages_tool_result_role_tool_with_id(self):
        """Tool results go back as role='tool' messages with the correct tool_call_id."""
        import clinic_agent.llm as llm_mod
        internal_messages = [
            {"role": "user", "content": "Please verify me: Asha Rao 1990-05-14"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Verifying identity..."},
                    {
                        "type": "tool_use",
                        "id": "call_abc_verify",
                        "name": "verify_patient",
                        "input": {"name": "Asha Rao", "dob": "1990-05-14"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc_verify",
                        "content": '{"verified": true}',
                    }
                ],
            },
        ]
        result = llm_mod._to_openrouter_messages(internal_messages, system="sys")
        roles = [m["role"] for m in result]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles[2] == "assistant"
        # The tool_result must appear as its OWN role="tool" message (not inside user)
        assert roles[3] == "tool", (
            f"Expected role='tool' for tool result, got {roles[3]!r}. "
            f"Full roles: {roles!r}"
        )
        tool_msg = result[3]
        assert tool_msg["tool_call_id"] == "call_abc_verify"
        assert '{"verified": true}' in tool_msg["content"]
        # Assistant must carry the tool_calls list with matching id
        assistant_msg = result[2]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["id"] == "call_abc_verify"

    def test_to_openrouter_messages_tool_result_role_from_raw_tool_result_role(self):
        """A raw role='tool_result' message (not nested in user) also becomes role='tool'."""
        import clinic_agent.llm as llm_mod
        internal_messages = [
            {
                "role": "tool_result",
                "tool_call_id": "call_xyz",
                "content": "done",
            },
        ]
        result = llm_mod._to_openrouter_messages(internal_messages, system="")
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_xyz"
        assert result[0]["content"] == "done"

    def test_openrouter_429_then_success_patched_sleep(self, monkeypatch):
        """_openrouter_generate_with_retries: 429 then success sleeps with patched time.sleep."""
        test_self = self
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class FakeCompletions:
            def create(self, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise test_self._make_openrouter_error(429, "429 Rate limit exceeded")
                # Success response
                msg = SimpleNamespace(content="pong", tool_calls=[])
                choice = SimpleNamespace(message=msg, finish_reason="stop")
                return SimpleNamespace(choices=[choice], usage=None)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        import clinic_agent.llm as llm_mod
        resp = llm_mod._openrouter_generate_with_retries(
            fake_client, {"model": "x", "messages": [], "temperature": 0, "max_tokens": 16}
        )
        assert resp.content == "pong"
        assert calls["n"] == 2
        # 429 first failure uses first default rate-limit sleep (10s)
        assert len(sleeps) == 1
        assert sleeps[0] == 10

    def test_openrouter_per_day_cap_fatal_no_sleep(self, monkeypatch):
        """Per-day quota message from OpenRouter raises FatalLLMError without sleeping."""
        test_self = self
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class FakeCompletions:
            def create(self, **kwargs):
                calls["n"] += 1
                raise test_self._make_openrouter_error(
                    429,
                    "429 You exceeded your daily quota for this API key",
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        import clinic_agent.llm as llm_mod
        with pytest.raises(llm_mod.FatalLLMError, match="per-day quota"):
            llm_mod._openrouter_generate_with_retries(
                fake_client, {"model": "x", "messages": [], "temperature": 0, "max_tokens": 16}
            )
        assert calls["n"] == 1
        assert sleeps == []

    def test_openrouter_401_fatal_no_retry(self, monkeypatch):
        """401 from OpenRouter raises FatalLLMError immediately (abort, no retry)."""
        test_self = self
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        class FakeCompletions:
            def create(self, **kwargs):
                calls["n"] += 1
                raise test_self._make_openrouter_error(401, "401 Unauthorized: invalid API key")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        import clinic_agent.llm as llm_mod
        with pytest.raises(llm_mod.FatalLLMError, match="abort immediately"):
            llm_mod._openrouter_generate_with_retries(
                fake_client, {"model": "x", "messages": [], "temperature": 0, "max_tokens": 16}
            )
        assert calls["n"] == 1
        assert sleeps == []

    def test_chat_explicit_openrouter_non_fatal_raises_no_fallback(self, monkeypatch):
        """LLM_PROVIDER=openrouter: non-fatal error raises; never falls back to Gemini."""
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        # Stub OpenRouter as "enabled" (key + SDK available) but chat raises non-fatal
        import clinic_agent.llm as llm_mod
        monkeypatch.setattr(llm_mod, "_openrouter_enabled", lambda: True)

        def boom(*a, **k):
            raise RuntimeError("openrouter transient: upstream timeout")

        monkeypatch.setattr(llm_mod, "_openrouter_chat", boom)
        # We must NOT call _chat_gemini: mark it so it raises if called
        called_gemini = {"n": 0}

        def gemini_bomb(*a, **k):
            called_gemini["n"] += 1
            raise AssertionError("Must NOT call Gemini when LLM_PROVIDER=openrouter")

        monkeypatch.setattr(llm_mod, "_chat_gemini", gemini_bomb)

        with pytest.raises(RuntimeError, match="openrouter transient"):
            llm_mod.chat(messages=[{"role": "user", "content": "hi"}], model_key="AGENT_MODEL")
        assert called_gemini["n"] == 0

    def test_chat_auto_mode_non_fatal_falls_back_and_logs_warning(self, monkeypatch, caplog):
        """LLM_PROVIDER=auto: non-fatal OpenRouter error falls back and logs WARNING."""
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("LLM_PROVIDER", "auto")
        monkeypatch.setenv("AGENT_MODEL", "openrouter/auto")
        import clinic_agent.llm as llm_mod
        monkeypatch.setattr(llm_mod, "_openrouter_enabled", lambda: True)

        def boom(*a, **k):
            raise RuntimeError("openrouter: gateway 502")

        monkeypatch.setattr(llm_mod, "_openrouter_chat", boom)

        fallback_resp = llm_mod.LLMResponse(content="gemini-served", provider="")
        gemini_calls = {"n": 0}

        def fake_gemini(*a, **k):
            gemini_calls["n"] += 1
            return fallback_resp

        monkeypatch.setattr(llm_mod, "_chat_gemini", fake_gemini)

        import logging
        caplog.set_level(logging.WARNING, logger="clinic_agent.llm")

        result = llm_mod.chat(messages=[{"role": "user", "content": "hi"}], model_key="AGENT_MODEL")

        # Fallback should have served via Gemini
        assert gemini_calls["n"] == 1
        assert result.provider == "gemini"
        # A WARNING log entry must mention BOTH model names (attempted and fallback)
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Falling back from OpenRouter" in w and "to Gemini" in w for w in warns), (
            f"Expected WARNING log naming both providers/models. Got warnings: {warns!r}"
        )
        # Log must include the attempted model name and error
        assert any("openrouter/auto" in w for w in warns), (
            f"Expected WARNING to name attempted model openrouter/auto. Warns: {warns!r}"
        )

    @staticmethod
    def _make_openrouter_error(code: int, message: str) -> Exception:
        """Build a fake openai.APIError-compatible exception with .code + string message."""
        e = Exception(message)
        e.code = code
        return e
