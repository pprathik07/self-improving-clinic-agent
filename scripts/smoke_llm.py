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
    """Call chat_fn once. Returns 'OK [provider:X]' or 'ErrorClass: message'."""
    try:
        resp = chat_fn(
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            system="You are a smoke-test probe. Reply briefly.",
            model_key=model_key,
            max_tokens=16,
        )
        _ = getattr(resp, "content", None)
        provider = getattr(resp, "provider", "") or "unknown"
        return f"OK [provider:{provider}]"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def smoke_function_call(chat_fn, model_key: str = "AGENT_MODEL") -> str:
    """Send a request with one real agent tool schema and verify a function call is returned.

    Uses verify_patient from clinic_agent/tools/schemas.py in the exact shape the
    agent uses (via pydantic_to_anthropic_tool), so the tool-schema cannot be
    silently dropped by chat()'s tool converter. The user message explicitly
    asks to verify Asha Rao DOB 1990-05-14 so the model has a clear reason to
    emit a function call.
    """
    try:
        from clinic_agent.llm import pydantic_to_anthropic_tool
        from clinic_agent.tools.schemas import VerifyPatientArgs

        tool = pydantic_to_anthropic_tool(
            "verify_patient",
            "Verify a patient's identity using their full name and date of birth.",
            VerifyPatientArgs,
        )
        resp = chat_fn(
            messages=[{
                "role": "user",
                "content": "Please verify my identity. My full name is Asha Rao, and my date of birth is 1990-05-14.",
            }],
            system="You are a clinic scheduling assistant. Use the provided tools to verify the patient.",
            tools=[tool],
            model_key=model_key,
            max_tokens=256,
        )
        provider = getattr(resp, "provider", "") or "unknown"
        tool_calls = getattr(resp, "tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            return f"OK [provider:{provider}]"
        return f"FAIL: no function call in response [provider:{provider}]"
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
        ok = result.startswith("OK")
        if not ok and is_abort_status(result):
            print(f"Aborting on fatal auth/model error from {key}.")
            return 1
        if not ok:
            return 1

    # Function-calling test on AGENT model
    agent_model = (env or os.environ).get("AGENT_MODEL", "")
    print(f"Function call ({agent_model}) ...", end=" ", flush=True)
    fc_result = smoke_function_call(chat_fn, model_key="AGENT_MODEL")
    print(fc_result)
    fc_ok = fc_result.startswith("OK")
    if not fc_ok and is_abort_status(fc_result):
        print(f"Aborting on fatal auth/model error from function-call test.")
        return 1
    if not fc_ok:
        return 1

    return 0


def main() -> int:
    return run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
