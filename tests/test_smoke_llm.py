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


def test_is_abort_status():
    assert smoke.is_abort_status("FatalLLMError: 404 NOT_FOUND")
    assert not smoke.is_abort_status("timeout after 30s")


def test_smoke_function_call_pass():
    """Fake client returns a tool_call → smoke_function_call returns 'OK'."""
    def fake_chat(**kwargs):
        tools = kwargs.get("tools")
        if tools:
            return SimpleNamespace(
                content="",
                tool_calls=[SimpleNamespace(name="get_weather", arguments={"city": "Paris"})],
            )
        return SimpleNamespace(content="pong", tool_calls=[])

    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    assert result == "OK"


def test_smoke_function_call_no_tool_call_fails():
    """Fake client returns text only (no tool_calls) → smoke_function_call returns FAIL."""
    def fake_chat(**kwargs):
        return SimpleNamespace(content="The weather is sunny.", tool_calls=[])

    result = smoke.smoke_function_call(fake_chat, model_key="AGENT_MODEL")
    assert result.startswith("FAIL")


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
