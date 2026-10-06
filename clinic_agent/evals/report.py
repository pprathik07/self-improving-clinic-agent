"""Report — before/after markdown table from results.json ONLY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ReportError(Exception):
    pass


def load_results(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_report(
    before: dict[str, Any],
    after: dict[str, Any] | None = None,
    *,
    allow_mock: bool = False,
) -> str:
    """Render a markdown report from results.json dicts."""
    modes = {before.get("llm_mode", "")}
    if after:
        modes.add(after.get("llm_mode", ""))
    is_mock = "mock" in modes or any(m != "real" for m in modes)

    if is_mock and not allow_mock:
        raise ReportError(
            "Refusing mock/non-real results.json. Pass allow_mock=True / --allow-mock for tests."
        )

    lines: list[str] = []
    if is_mock:
        lines.append("> **MOCK, DO NOT PUBLISH**")
        lines.append("")

    lines.append("# Eval Report")
    lines.append("")
    lines.append("## Models / mode")
    lines.append("")
    lines.append("| Field | Before | After |")
    lines.append("|---|---|---|")
    for key in ("llm_mode", "agent_model", "judge_model", "sim_model", "policy_hash", "k"):
        b = before.get(key, "")
        a = after.get(key, "") if after else ""
        lines.append(f"| {key} | {b} | {a} |")
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Split | Before | After |")
    lines.append("|---|---|---|")
    for split in ("overall", "train", "heldout"):
        b = before.get(split, {})
        a = (after or {}).get(split, {}) if after else {}
        b_s = f"{b.get('passed', '?')}/{b.get('total', '?')} ({b.get('pass_rate', 0):.0%})"
        a_s = (
            f"{a.get('passed', '?')}/{a.get('total', '?')} ({a.get('pass_rate', 0):.0%})"
            if after
            else "—"
        )
        lines.append(f"| {split} | {b_s} | {a_s} |")
    lines.append("")

    # Turns / cost proxies
    b_turns = _total_turns(before)
    a_turns = _total_turns(after) if after else None
    lines.append("## Turns")
    lines.append("")
    lines.append(f"- before: {b_turns}")
    if a_turns is not None:
        lines.append(f"- after: {a_turns}")
    lines.append("")

    lines.append("## Per-scenario pass rate (over k)")
    lines.append("")
    lines.append("| Scenario | Split | Before | After |")
    lines.append("|---|---|---|---|")
    ids = sorted(
        set(before.get("per_scenario", {}))
        | set((after or {}).get("per_scenario", {}))
    )
    for sid in ids:
        b = before.get("per_scenario", {}).get(sid, {})
        a = (after or {}).get("per_scenario", {}).get(sid, {}) if after else {}
        split = b.get("split") or a.get("split") or "?"
        b_s = f"{b.get('pass_count', '?')}/{b.get('total_runs', '?')} ({b.get('pass_rate', 0):.0%})"
        a_s = (
            f"{a.get('pass_count', '?')}/{a.get('total_runs', '?')} ({a.get('pass_rate', 0):.0%})"
            if after
            else "—"
        )
        lines.append(f"| {sid} | {split} | {b_s} | {a_s} |")
    lines.append("")
    return "\n".join(lines)


def _total_turns(results: dict[str, Any]) -> int:
    total = 0
    for data in results.get("per_scenario", {}).values():
        for run in data.get("runs", []):
            total += int(run.get("turns", 0) or 0)
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit before/after markdown from results.json")
    parser.add_argument("before", help="Path to before results.json")
    parser.add_argument("after", nargs="?", help="Optional after results.json")
    parser.add_argument("--allow-mock", action="store_true", help="Allow mock llm_mode (stamps output)")
    parser.add_argument("-o", "--output", help="Write to file instead of stdout")
    args = parser.parse_args(argv)

    before = load_results(args.before)
    after = load_results(args.after) if args.after else None
    try:
        text = render_report(before, after, allow_mock=args.allow_mock)
    except ReportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
