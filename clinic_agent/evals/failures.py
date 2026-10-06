"""Generate runs/<ts>/failures.md from results.json + traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_failures_md(run_dir: str | Path, results: dict[str, Any] | None = None) -> Path:
    """Write failures.md for a run directory. Returns the path written."""
    run_dir = Path(run_dir)
    if results is None:
        results_path = run_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"No results.json in {run_dir}")
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

    lines: list[str] = [
        f"# Failures — {results.get('timestamp', run_dir.name)}",
        "",
        f"- policy_hash: `{results.get('policy_hash', '')}`",
        f"- llm_mode: `{results.get('llm_mode', '')}`",
        f"- k: {results.get('k', '')}",
        "",
    ]

    failing = []
    for scenario_id, data in results.get("per_scenario", {}).items():
        if data.get("pass_rate", 1.0) >= 1.0:
            continue
        if data.get("split") != "train":
            # Still document heldout failures for human review, but mark split
            pass
        failing.append((scenario_id, data))

    if not failing:
        lines.append("No failing scenarios.")
        lines.append("")
    else:
        for scenario_id, data in failing:
            lines.extend(_scenario_section(run_dir, scenario_id, data))

    out = run_dir / "failures.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _scenario_section(run_dir: Path, scenario_id: str, data: dict[str, Any]) -> list[str]:
    lines = [
        f"## {scenario_id} ({data.get('split', '?')}) — pass rate {data.get('pass_rate', 0):.0%}",
        "",
    ]
    # Pick first failing non-error run
    failing_run = None
    failing_idx = None
    for i, run in enumerate(data.get("runs", [])):
        if not run.get("passed") and not run.get("error"):
            failing_run = run
            failing_idx = i
            break

    if failing_run is None:
        lines.append("_No non-error failing run (infra-only)._")
        lines.append("")
        return lines

    scores = failing_run.get("scores") or {}
    state_f = [k for k, v in (scores.get("state") or {}).items() if not v]
    trace_f = [k for k, v in (scores.get("trace") or {}).items() if not v]
    judge_f = [k for k, v in (scores.get("judge") or {}).items() if not v]

    caught = "unknown"
    if state_f:
        caught = "state"
    elif trace_f:
        caught = "trace"
    elif judge_f:
        caught = "judge"

    lines.append(f"**Caught by:** {caught} (trust order: state > trace > judge)")
    lines.append("")
    lines.append("### State checks")
    lines.append(f"- failures: {state_f or 'none'}")
    lines.append("")
    lines.append("### Trace checks")
    lines.append(f"- failures: {trace_f or 'none'}")
    lines.append("")
    lines.append("### Judge verdicts")
    lines.append(f"- failures: {judge_f or 'none'}")
    if scores.get("failures"):
        for f in scores["failures"]:
            lines.append(f"  - {f}")
    lines.append("")

    transcript, tools = _load_trace_segment(run_dir, scenario_id, failing_idx)
    lines.append("### Tool calls")
    if tools:
        for t in tools:
            lines.append(f"- `{t}`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### Transcript")
    lines.append("```")
    lines.append(transcript or "(empty)")
    lines.append("```")
    lines.append("")
    return lines


def _load_trace_segment(
    run_dir: Path, scenario_id: str, run_index: int | None
) -> tuple[str, list[str]]:
    trace_file = run_dir / "traces" / f"{scenario_id}.jsonl"
    if not trace_file.exists() or run_index is None:
        return "", []

    segments: list[list[dict]] = []
    current: list[dict] = []
    with open(trace_file, encoding="utf-8") as f:
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

    if run_index >= len(segments):
        return "", []

    transcript_lines = []
    tools = []
    for event in segments[run_index]:
        et = event.get("event")
        if et == "patient_turn":
            transcript_lines.append(f"Patient: {event.get('response', '')}")
        elif et == "agent_turn":
            transcript_lines.append(f"Agent: {event.get('response', '')}")
        elif et == "tool_call":
            args = event.get("args", {})
            result = str(event.get("result", ""))[:200]
            tools.append(f"{event.get('tool', '')}({args}) -> {result}")
    return "\n".join(transcript_lines), tools
