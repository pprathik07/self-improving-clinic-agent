"""Reflector — analyzes eval failures and proposes structured policy patches.

Uses REFLECTOR_MODEL (separate from agent) to analyze transcripts, tool traces,
and state-check output for train-split failures, then proposes minimal policy
edits to fix them.

Key constraint: the reflector ONLY sees train-split failures, never heldout.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import yaml

from pydantic import BaseModel, Field

from clinic_agent.loop.patch import Patch, FailureCategory, Evidence, PolicyChange

logger = logging.getLogger(__name__)

_TURN_PREFIX_RE = re.compile(r"^\[turn \d+\]\s*")


# ---------------------------------------------------------------------------
# Failure report (input to reflector)
# ---------------------------------------------------------------------------

class FailureReport(BaseModel):
    """Information about a failing scenario for the reflector to analyze."""
    scenario_id: str
    split: str  # "train" or "heldout"
    transcript: str
    opening: str  # For heldout leak detection
    patient_persona: str  # For heldout leak detection
    tool_trace: list[str]  # Tool call descriptions (name(args) -> result)
    state_check_failures: list[str]
    trace_check_failures: list[str]
    judge_failures: list[str]
    current_policy_sections: dict[str, str]
    pass_rate: float  # 0.0 to 1.0


# ---------------------------------------------------------------------------
# Reflector input
# ---------------------------------------------------------------------------

class ReflectorInput(BaseModel):
    """Input to the reflector."""
    failures: list[FailureReport]
    current_policy: dict[str, Any]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ReflectorError(Exception):
    """Base exception for reflector errors."""
    pass


class HeldoutLeakError(ReflectorError):
    """Raised when heldout data would leak into the reflector prompt."""
    pass


class InvalidJSONError(ReflectorError):
    """Raised when the LLM response cannot be parsed as JSON."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scenario_yaml_index() -> dict[str, dict[str, Any]]:
    scenarios_dir = Path(__file__).resolve().parents[1] / "evals" / "scenarios"
    index: dict[str, dict[str, Any]] = {}
    for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        index[data.get("id", "")] = data
    return index


def _split_trace_segments(trace_file: Path) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            current.append(event)
            if event.get("event") == "conversation_end":
                segments.append(current)
                current = []
    if current:
        segments.append(current)
    return segments


def _format_tool_call(event: dict[str, Any]) -> str:
    tool_name = event.get("tool", "")
    args = event.get("args", {})
    result = event.get("result", {})
    result_str = str(result) if result is not None else ""
    if len(result_str) > 200:
        result_str = result_str[:200] + " [truncated]"
    return f"{tool_name}({args}) -> {result_str}"


def _extract_layer_failures(
    scores: dict[str, Any],
    layer: str,
) -> list[str]:
    """Build display lines for false keys in scores[layer].

    Real failures strings look like: "[FAIL] appointments_created: Expected 1, got 0"
    (no "state:" prefix). Judge keys may be truncated — prefix-match. Cap at 400 chars.
    """
    layer_checks = scores.get(layer) or {}
    failures_list = scores.get("failures") or []
    out: list[str] = []
    for key, passed in layer_checks.items():
        if passed:
            continue
        detail = None
        needle = f"[FAIL] {key}:"
        for f in failures_list:
            if needle in f:
                # Strip leading "[FAIL] " for the printed detail after the key
                rest = f.split(":", 1)
                if len(rest) == 2 and f.startswith("[FAIL] "):
                    # "[FAIL] key: detail" -> use detail part after first "key:"
                    after_fail = f[len("[FAIL] "):]
                    if after_fail.startswith(key + ":"):
                        detail = after_fail[len(key) + 1:].strip()
                    else:
                        detail = after_fail
                else:
                    detail = f
                break
            # Judge: truncated keys — prefix-match after [FAIL]
            if layer == "judge" and f.startswith("[FAIL] ") and key in f:
                after_fail = f[len("[FAIL] "):]
                detail = after_fail
                break
        if detail is None:
            # No matching failures string — use key name only, never invent text
            line = f"- {layer}: {key}"
        else:
            line = f"- {layer}: {key}: {detail}"
        if len(line) > 400:
            line = line[:400] + " [truncated]"
        out.append(line)
    return out


def _build_transcript_entries(segment: list[dict[str, Any]]) -> list[str]:
    """One patient_turn/agent_turn event -> one entry (newlines inside text OK)."""
    entries: list[str] = []
    for event in segment:
        et = event.get("event")
        if et == "patient_turn":
            role = "Patient"
        elif et == "agent_turn":
            role = "Agent"
        else:
            continue
        body = event.get("response", "")
        turn = event.get("turn")
        if turn is not None:
            entries.append(f"[turn {turn}] {role}: {body}")
        else:
            entries.append(f"{role}: {body}")
    return entries


def _entry_cmp_key(entry: str) -> str:
    """Compare entries without the [turn N] prefix."""
    return _TURN_PREFIX_RE.sub("", entry)


def _collapse_repeated_entries(entries: list[str]) -> list[str]:
    """Collapse consecutive identical blocks of 1 or 2 entries.

    Compare text without the turn prefix. Print the block once, then
    "[previous <n> entry(ies) repeated <N> more times]" where N = occurrences - 1.
    """
    if not entries:
        return []
    result: list[str] = []
    i = 0
    n = len(entries)
    while i < n:
        # Prefer 2-entry block if it repeats AND the two entries differ
        if (
            i + 3 < n
            and _entry_cmp_key(entries[i]) != _entry_cmp_key(entries[i + 1])
            and _entry_cmp_key(entries[i]) == _entry_cmp_key(entries[i + 2])
            and _entry_cmp_key(entries[i + 1]) == _entry_cmp_key(entries[i + 3])
        ):
            block = [entries[i], entries[i + 1]]
            keys = [_entry_cmp_key(entries[i]), _entry_cmp_key(entries[i + 1])]
            occurrences = 0
            j = i
            while (
                j + 1 < n
                and _entry_cmp_key(entries[j]) == keys[0]
                and _entry_cmp_key(entries[j + 1]) == keys[1]
            ):
                occurrences += 1
                j += 2
            result.extend(block)
            if occurrences > 1:
                result.append(
                    f"[previous 2 entry(ies) repeated {occurrences - 1} more times]"
                )
            i = j
            continue

        # 1-entry block
        key = _entry_cmp_key(entries[i])
        occurrences = 1
        j = i + 1
        while j < n and _entry_cmp_key(entries[j]) == key:
            occurrences += 1
            j += 1
        result.append(entries[i])
        if occurrences > 1:
            result.append(
                f"[previous 1 entry(ies) repeated {occurrences - 1} more times]"
            )
        i = j
    return result


def _cap_entries(entries: list[str], limit: int = 30) -> list[str]:
    """Keep the last `limit` entries; prepend omitted marker if any dropped."""
    if len(entries) <= limit:
        return entries
    omitted = len(entries) - limit
    return [f"[{omitted} earlier entries omitted]"] + entries[-limit:]


def _collapse_tool_calls(tool_trace: list[str]) -> list[str]:
    """Collapse consecutive identical tool-call strings (1-entry blocks)."""
    return _collapse_repeated_entries(tool_trace)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------

def build_reflector_input(results: dict[str, Any], run_dir: Path, policy: dict[str, Any]) -> ReflectorInput:
    """Build reflector input from eval results and trace files.

    For real results.json: decide reported scenarios from results alone
    (train, pass_rate < 1.0, no error runs), then open traces ONLY for those.
    """
    run_dir = Path(run_dir)
    policy_text_only = {
        k: v.get("text", "") if isinstance(v, dict) else v for k, v in policy.items()
    }

    # ---- Test fixture format ----
    if "scenarios" in results:
        scenarios = results.get("scenarios", [])
        train_failures: list[FailureReport] = []
        for scenario in scenarios:
            if scenario.get("split") == "heldout":
                continue
            pass_rate = scenario.get("pass_rate", 0.0)
            if pass_rate >= 1.0:
                continue
            if scenario.get("is_infra_error", False):
                continue
            train_failures.append(FailureReport(
                scenario_id=scenario.get("id", ""),
                split=scenario.get("split", "train"),
                transcript=scenario.get("transcript", ""),
                opening=scenario.get("opening", ""),
                patient_persona=scenario.get("patient_persona", ""),
                tool_trace=scenario.get("tool_trace", []),
                state_check_failures=scenario.get("state_check_failures", []),
                trace_check_failures=scenario.get("trace_check_failures", []),
                judge_failures=scenario.get("judge_failures", []),
                current_policy_sections=policy_text_only,
                pass_rate=pass_rate,
            ))
        if not train_failures:
            return ReflectorInput(failures=[], current_policy=policy)
        prompt = _build_prompt(train_failures, policy)
        _check_for_heldout_leak(prompt, scenarios)
        return ReflectorInput(failures=train_failures, current_policy=policy)

    # ---- Real results.json format ----
    if "per_scenario" not in results:
        return ReflectorInput(failures=[], current_policy=policy)

    scenario_yaml = _load_scenario_yaml_index()
    per_scenario = results.get("per_scenario", {})

    # Leak-check list: metadata only (never open heldout/passing traces)
    all_scenarios_meta: list[dict[str, Any]] = []
    for scenario_id, data in per_scenario.items():
        yaml_s = scenario_yaml.get(scenario_id, {})
        all_scenarios_meta.append({
            "id": scenario_id,
            "split": data.get("split", "train"),
            "opening": yaml_s.get("opening", ""),
            "patient_persona": yaml_s.get("patient_persona", ""),
        })

    # 1) Decide reported scenarios from results.json alone
    reported: list[tuple[str, dict[str, Any]]] = []
    all_error_excluded = 0
    for scenario_id, data in per_scenario.items():
        if data.get("split") != "train":
            continue
        if data.get("pass_rate", 1.0) >= 1.0:
            continue
        runs = data.get("runs", [])
        # Every run errored → exclude entirely (never open trace, never raise here)
        if runs and all(r.get("error") for r in runs):
            logger.warning(
                "Excluding all-error train scenario from reflector: %s",
                scenario_id,
            )
            all_error_excluded += 1
            continue
        reported.append((scenario_id, data))

    train_failures = []
    for scenario_id, data in reported:
        yaml_s = scenario_yaml.get(scenario_id, {})
        runs = data.get("runs", [])

        # Pick first failing non-error run (error runs are never the evidence)
        failing_run = None
        failing_run_index = None
        for i, run in enumerate(runs):
            if not run.get("passed") and not run.get("error"):
                failing_run = run
                failing_run_index = i
                break

        if failing_run is None:
            # pass_rate < 1.0 with non-error runs but none failing → inconsistent
            raise ReflectorError(
                f"No failing non-error run for scenario {scenario_id}"
            )

        # 2) Open trace ONLY for reported scenarios
        trace_file = run_dir / "traces" / f"{scenario_id}.jsonl"
        if not trace_file.exists():
            raise ReflectorError(f"Trace file not found for scenario {scenario_id}: {trace_file}")

        segments = _split_trace_segments(trace_file)
        if len(segments) != len(runs):
            raise ReflectorError(
                f"Trace segment count ({len(segments)}) != run count ({len(runs)}) "
                f"for scenario {scenario_id}"
            )

        segment = segments[failing_run_index]
        entries = _build_transcript_entries(segment)
        collapsed = _collapse_repeated_entries(entries)
        capped = _cap_entries(collapsed, limit=30)
        transcript = "\n".join(capped)

        tool_trace_raw: list[str] = []
        for event in segment:
            if event.get("event") == "tool_call":
                tool_trace_raw.append(_format_tool_call(event))
        tool_trace = _collapse_tool_calls(tool_trace_raw)

        scores = failing_run.get("scores") or {}
        state_check_failures = _extract_layer_failures(scores, "state")
        trace_check_failures = _extract_layer_failures(scores, "trace")
        judge_failures = _extract_layer_failures(scores, "judge")

        train_failures.append(FailureReport(
            scenario_id=scenario_id,
            split=data.get("split", "train"),
            transcript=transcript,
            opening=yaml_s.get("opening", ""),
            patient_persona=yaml_s.get("patient_persona", ""),
            tool_trace=tool_trace,
            state_check_failures=state_check_failures,
            trace_check_failures=trace_check_failures,
            judge_failures=judge_failures,
            current_policy_sections=policy_text_only,
            pass_rate=data.get("pass_rate", 0.0),
        ))

    if not train_failures:
        # Every failing train scenario was all-error → clean stop, not empty prompt
        if all_error_excluded > 0:
            raise ReflectorError(
                "no failures: every failing train scenario had only error runs"
            )
        return ReflectorInput(failures=[], current_policy=policy)

    prompt = _build_prompt(train_failures, policy)
    _check_for_heldout_leak(prompt, all_scenarios_meta)
    return ReflectorInput(failures=train_failures, current_policy=policy)


def _caught_by_layer(f: FailureReport) -> str:
    """First failing layer in trust order: state > trace > judge."""
    if f.state_check_failures:
        return "state"
    if f.trace_check_failures:
        return "trace"
    if f.judge_failures:
        return "judge"
    return "unknown"


def _build_prompt(failures: list[FailureReport], current_policy: dict[str, Any]) -> str:
    """Print failures and policy. Transcript is already final in FailureReport."""
    policy_sections = []
    for section_id, section in current_policy.items():
        text = section.get("text", str(section)) if isinstance(section, dict) else str(section)
        policy_sections.append(f"## {section_id}\n\n{text}\n")

    failure_descriptions = []
    for f in failures[:3]:
        state_block = "\n".join(f.state_check_failures) if f.state_check_failures else "(none)"
        trace_block = "\n".join(f.trace_check_failures) if f.trace_check_failures else "(none)"
        judge_block = "\n".join(f.judge_failures) if f.judge_failures else "(none)"
        tool_calls_str = "\n".join(f.tool_trace) if f.tool_trace else "None"
        caught = _caught_by_layer(f)
        failure_descriptions.append(
            f"### Scenario: {f.scenario_id}\n"
            f"**Caught by (trust order state > trace > judge):** {caught}\n"
            f"**State check failures:**\n{state_block}\n"
            f"**Trace check failures:**\n{trace_block}\n"
            f"**Judge failures:**\n{judge_block}\n"
            f"**Tool calls:**\n{tool_calls_str}\n\n"
            f"**Transcript:**\n{f.transcript}\n"
        )

    return (
        f"## Current Policy\n\n"
        f"{''.join(policy_sections)}\n"
        f"## Failing Scenarios (train split only)\n\n"
        f"{''.join(failure_descriptions)}\n"
        f"Analyze these failures and propose 1 to 3 policy changes."
    )


def _check_for_heldout_leak(prompt: str, all_scenarios: list[dict[str, Any]]) -> None:
    """Check that no heldout scenario data appears in the prompt."""
    prompt_lower = prompt.lower()
    prompt_normalized = " ".join(prompt_lower.split())

    for scenario in all_scenarios:
        if scenario.get("split") != "heldout":
            continue

        scenario_id = scenario.get("id", "").lower()
        opening = scenario.get("opening", "").lower()
        persona = scenario.get("patient_persona", "").lower()

        if scenario_id and scenario_id in prompt_lower:
            raise HeldoutLeakError(
                f"Heldout scenario ID '{scenario_id}' found in reflector prompt"
            )

        if opening and opening in prompt_lower:
            raise HeldoutLeakError(
                f"Heldout scenario opening text found in reflector prompt"
            )

        if persona and persona in prompt_lower:
            raise HeldoutLeakError(
                f"Heldout scenario persona text found in reflector prompt"
            )

        for text in [opening, persona]:
            if not text:
                continue
            text_normalized = " ".join(text.split())
            words = text_normalized.split()
            if len(words) < 8:
                continue
            for i in range(len(words) - 7):
                phrase = " ".join(words[i:i + 8])
                if phrase in prompt_normalized:
                    raise HeldoutLeakError(
                        f"Heldout text sequence (8+ words) found in reflector prompt"
                    )


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

def reflect(input_data: ReflectorInput, llm: Callable[[str, str], str]) -> Patch:
    """Analyze failures and propose a policy patch."""
    if not input_data.failures:
        raise ReflectorError("No failures to analyze")

    system_prompt = Path(__file__).parent / "prompts" / "reflector.md"
    system_prompt_text = system_prompt.read_text(encoding="utf-8")
    message = _build_prompt(input_data.failures, input_data.current_policy)

    for attempt in range(2):
        try:
            response_text = llm(system_prompt_text, message)
            patch = _parse_patch(response_text)
            if patch:
                return patch
            elif attempt == 0:
                logger.warning("First attempt returned None, retrying...")
                continue
            else:
                raise InvalidJSONError("Failed to parse patch after retry")
        except json.JSONDecodeError as e:
            if attempt == 0:
                logger.warning("JSON decode error on first attempt, retrying: %s", e)
                continue
            else:
                raise InvalidJSONError(f"Failed to parse JSON after retry: {e}")

    raise InvalidJSONError("Failed to produce valid patch")


def _parse_patch(response_text: str) -> Patch | None:
    """Parse and validate the reflector's patch response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return Patch(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Failed to parse patch: %s\nResponse: %s", e, text[:500])
        return None
