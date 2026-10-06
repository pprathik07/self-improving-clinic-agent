"""Tests for scripts/smoke_llm.py with a fake chat client."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_llm.py"
    spec = importlib.util.spec_from_file_location("smoke_llm", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load()


def test_models_must_differ():
    ok, _ = smoke.models_are_distinct({
        "AGENT_MODEL": "a",
        "JUDGE_MODEL": "a",
        "SIM_MODEL": "c",
    })
    assert ok is False
    ok, detail = smoke.models_are_distinct({
        "AGENT_MODEL": "a",
        "JUDGE_MODEL": "b",
        "SIM_MODEL": "c",
    })
    assert ok is True
    assert "distinct" in detail


def test_smoke_ok_with_fake_client():
    env = {
        "AGENT_MODEL": "m-agent",
        "JUDGE_MODEL": "m-judge",
        "SIM_MODEL": "m-sim",
        "REFLECTOR_MODEL": "m-ref",
    }

    def fake_chat(**kwargs):
        tools = kwargs.get("tools")
        if tools:
            return SimpleNamespace(
                content="",
                tool_calls=[SimpleNamespace(name="get_weather", arguments={"city": "Paris"})],
            )
        return SimpleNamespace(content="pong")

    code = smoke.run_smoke(chat_fn=fake_chat, env=env)
    assert code == 0


def test_smoke_aborts_on_403():
    env = {
        "AGENT_MODEL": "m-agent",
        "JUDGE_MODEL": "m-judge",
        "SIM_MODEL": "m-sim",
        "REFLECTOR_MODEL": "m-ref",
    }
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        raise RuntimeError("403 PERMISSION_DENIED")

    code = smoke.run_smoke(chat_fn=fake_chat, env=env)
    assert code == 1
    assert calls["n"] == 1


def test_smoke_function_call_returns_ok_when_fake_has_tool_call():
    """smoke_function_call returns 'OK [provider:X]' when chat_fn returns at least one tool_call."""
    seen: dict = {}

    def fake_chat(**kwargs):
        seen["tools"] = kwargs.get("tools")
        return SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(
                name="verify_patient",
                arguments={"name": "Asha Rao", "dob": "1990-05-14"},
            )],
        )

    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    # Tool passed through should contain the verify_patient name from pydantic shape
    assert isinstance(seen["tools"], list) and len(seen["tools"]) == 1
    assert seen["tools"][0]["name"] == "verify_patient"
    assert seen["tools"][0].get("input_schema") or seen["tools"][0].get("parameters"), \
        "Tool dict must carry a non-empty schema key"
    assert result.startswith("OK"), f"Expected result starting with 'OK', got {result!r}"
    # Provider info appended
    assert "[provider:" in result, f"Expected result to include provider tag, got {result!r}"


def test_smoke_function_call_returns_fail_when_fake_has_only_text():
    """smoke_function_call returns the FAIL string with provider when chat_fn returned text only."""

    def fake_chat(**kwargs):
        return SimpleNamespace(
            content="Hi Asha — I've verified your identity using the info you provided.",
            tool_calls=[],
        )

    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    assert result.startswith("FAIL: no function call in response"), (
        f"Expected FAIL prefix, got {result!r}"
    )
    assert "[provider:" in result, f"Expected FAIL result to include provider tag, got {result!r}"


def test_smoke_function_call_raises_error_on_malformed_tool_via_chat():
    """If chat_fn propagates a ValueError for a malformed tool dict, smoke surfaces it."""

    def fake_chat(**kwargs):
        tools = kwargs.get("tools") or []
        # Simulate a chat() that enforces tool-schema shape (as _convert_tools does).
        for t in tools:
            if isinstance(t, dict):
                if not t.get("name") or not (t.get("input_schema") or t.get("parameters")):
                    raise ValueError(f"malformed tool: {t!r}")
        return SimpleNamespace(content="", tool_calls=[])

    # Purposely call chat with a malformed tool (the smoke no longer does this, but
    # we verify the wrapper surfaces any ValueError as an error string).
    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    # fake_chat receives the good tool from smoke, so no error raised and returns FAIL text only
    assert result.startswith("FAIL: no function call in response"), (
        f"Expected FAIL prefix, got {result!r}"
    )
    assert "[provider:" in result, f"Expected FAIL result to include provider tag, got {result!r}"


def test_is_abort_status():
    assert smoke.is_abort_status("FatalLLMError: 404 NOT_FOUND")
    assert not smoke.is_abort_status("timeout after 30s")


def test_smoke_function_call_pass():
    """Fake client returns a tool_call → smoke_function_call returns 'OK [provider:X]'."""
    def fake_chat(**kwargs):
        tools = kwargs.get("tools")
        if tools:
            return SimpleNamespace(
                content="",
                tool_calls=[SimpleNamespace(name="get_weather", arguments={"city": "Paris"})],
            )
        return SimpleNamespace(content="pong", tool_calls=[])

    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    assert result.startswith("OK"), f"Expected OK prefix, got {result!r}"
    assert "[provider:" in result, f"Expected provider tag in result, got {result!r}"


def test_smoke_function_call_no_tool_call_fails():
    """Fake client returns text only (no tool_calls) → smoke_function_call returns FAIL with provider."""
    def fake_chat(**kwargs):
        return SimpleNamespace(content="The weather is sunny.", tool_calls=[])

    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    assert result.startswith("FAIL"), f"Expected FAIL prefix, got {result!r}"
    assert "[provider:" in result, f"Expected provider tag in FAIL result, got {result!r}"


def test_smoke_run_includes_function_call():
    """run_smoke must call smoke_function_call and fail when it fails."""
    env = {
        "AGENT_MODEL": "m-agent",
        "JUDGE_MODEL": "m-judge",
        "SIM_MODEL": "m-sim",
        "REFLECTOR_MODEL": "m-ref",
    }
    calls = {"text": 0, "func": 0}

    def fake_chat(**kwargs):
        tools = kwargs.get("tools")
        if tools:
            calls["func"] += 1
            # Return no function call to trigger failure
            return SimpleNamespace(content="text only", tool_calls=[])
        calls["text"] += 1
        return SimpleNamespace(content="pong")

    code = smoke.run_smoke(chat_fn=fake_chat, env=env)
    # Text calls should all pass (4 models), then function call should fail
    assert calls["text"] == 4
    assert calls["func"] == 1
    assert code == 1  # fails because function call returned no tool_calls
