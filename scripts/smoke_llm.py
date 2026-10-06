"""Minimal smoke test: one call per configured model via clinic_agent.llm.

Human runs this (needs a real key). Abort on first 401/403/404.
Never prints the API key.

Usage: uv run python scripts/smoke_llm.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


MODEL_KEYS = ("AGENT_MODEL", "JUDGE_MODEL", "SIM_MODEL", "REFLECTOR_MODEL")


def models_are_distinct(env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Return (ok, detail). agent/judge/sim must all differ."""
    src = env if env is not None else os.environ
    agent = src.get("AGENT_MODEL", "")
    judge = src.get("JUDGE_MODEL", "")
    sim = src.get("SIM_MODEL", "")
    vals = [agent, judge, sim]
    if not all(vals):
        return False, f"missing model env: AGENT={agent!r} JUDGE={judge!r} SIM={sim!r}"
    if len(set(vals)) < 3:
        return False, f"AGENT/JUDGE/SIM must all differ; got {vals}"
    return True, f"distinct: {vals}"


def smoke_one(model_key: str, chat_fn) -> str:
    """Call chat_fn once. Returns 'OK' or 'ErrorClass: message'."""
    try:
        resp = chat_fn(
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            system="You are a smoke-test probe. Reply briefly.",
            model_key=model_key,
            max_tokens=16,
        )
        _ = getattr(resp, "content", None)
        return "OK"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def smoke_function_call(chat_fn, model_key: str = "AGENT_MODEL") -> str:
    """Send a request with a function declaration and verify a function call is returned.

    Returns 'OK' if the reply contains at least one function/tool call,
    or an error string on failure. Never prints the API key.
    """
    _GET_WEATHER_TOOL = {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }
    try:
        resp = chat_fn(
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            system="You are a helpful assistant. Use the provided tools.",
            tools=[_GET_WEATHER_TOOL],
            model_key=model_key,
            max_tokens=128,
        )
        tool_calls = getattr(resp, "tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            return "OK"
        return "FAIL: no function call in response"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def is_abort_status(message: str) -> bool:
    return bool(
        __import__("re").search(r"\b(401|403|404)\b", message)
        or "PERMISSION_DENIED" in message.upper()
        or "UNAUTHENTICATED" in message.upper()
        or "NOT_FOUND" in message.upper()
    )


def run_smoke(chat_fn=None, env: dict[str, str] | None = None) -> int:
    """Run smoke checks. Returns process exit code."""
    load_dotenv()
    ok, detail = models_are_distinct(env)
    if not ok:
        print(f"FAIL: {detail}")
        return 1
    print(f"Models: {detail}")

    if chat_fn is None:
        from clinic_agent.llm import chat as chat_fn  # type: ignore

    for key in MODEL_KEYS:
        model = (env or os.environ).get(key, "")
        print(f"{key}={model} ...", end=" ", flush=True)
        result = smoke_one(key, chat_fn)
        print(result)
        if result != "OK" and is_abort_status(result):
            print(f"Aborting on fatal auth/model error from {key}.")
            return 1
        if result != "OK":
            return 1

    # Function-calling test on AGENT model
    agent_model = (env or os.environ).get("AGENT_MODEL", "")
    print(f"Function call ({agent_model}) ...", end=" ", flush=True)
    fc_result = smoke_function_call(chat_fn, model_key="AGENT_MODEL")
    print(fc_result)
    if fc_result != "OK" and is_abort_status(fc_result):
        print(f"Aborting on fatal auth/model error from function-call test.")
        return 1
    if fc_result != "OK":
        return 1

    return 0


def main() -> int:
    return run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
