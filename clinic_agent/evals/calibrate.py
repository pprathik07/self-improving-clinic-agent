"""Judge calibration — compute agree-rate from HUMAN-filled labels only.

Generate transcripts from a real (or allow-mock) eval run; never fill labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


class CalibrateError(Exception):
    pass


def load_labels(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_agree_rate(labels_dir: str | Path) -> dict[str, Any]:
    """Compute judge vs human agree-rate from filled label files.

    Empty templates (no human_labels) are skipped and reported as unlabeled.
    """
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.glob("labels_*.yaml"))
    if not files:
        raise CalibrateError(f"No label YAMLs in {labels_dir}")

    labeled = 0
    unlabeled = 0
    agreements = 0
    comparisons = 0
    details = []

    for path in files:
        data = load_labels(path)
        human = data.get("human_labels")
        judge = data.get("judge_labels") or {}
        if not human:
            unlabeled += 1
            details.append({"file": path.name, "status": "unlabeled"})
            continue
        labeled += 1
        for key, human_val in human.items():
            if key not in judge:
                continue
            comparisons += 1
            agree = bool(human_val) == bool(judge[key])
            if agree:
                agreements += 1
            details.append({
                "file": path.name,
                "rubric": key,
                "human": human_val,
                "judge": judge[key],
                "agree": agree,
            })

    rate = agreements / comparisons if comparisons else None
    return {
        "labeled_files": labeled,
        "unlabeled_files": unlabeled,
        "comparisons": comparisons,
        "agreements": agreements,
        "agree_rate": rate,
        "details": details,
    }


def _load_trace_segments(trace_file: Path) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
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
    return segments


def _segment_to_transcript(segment: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in segment:
        et = event.get("event")
        if et == "patient_turn":
            lines.append(f"Patient: {event.get('response', '')}")
        elif et == "agent_turn":
            lines.append(f"Agent: {event.get('response', '')}")
        elif et == "tool_call":
            lines.append(
                f"Tool: {event.get('tool', '')}({event.get('args', {})})"
            )
    return "\n".join(lines) + ("\n" if lines else "")


def _select_six_runs(results: dict[str, Any]) -> list[tuple[str, int, bool]]:
    """Deterministic mix of pass and fail runs: fails first (sorted), then passes."""
    fails: list[tuple[str, int]] = []
    passes: list[tuple[str, int]] = []
    for scenario_id in sorted(results.get("per_scenario", {})):
        data = results["per_scenario"][scenario_id]
        for run in data.get("runs", []):
            if run.get("error"):
                continue
            idx = int(run.get("run_index", 0))
            if run.get("passed"):
                passes.append((scenario_id, idx))
            else:
                fails.append((scenario_id, idx))
    fails.sort()
    passes.sort()
    selected: list[tuple[str, int, bool]] = []
    for sid, idx in fails:
        if len(selected) >= 6:
            break
        selected.append((sid, idx, False))
    for sid, idx in passes:
        if len(selected) >= 6:
            break
        selected.append((sid, idx, True))
    if len(selected) < 6:
        raise CalibrateError(
            f"Need at least 6 non-error runs to calibrate; found {len(selected)}"
        )
    return selected[:6]


def generate_from_run(
    run_dir: str | Path,
    out_dir: str | Path = "data/calibration",
    *,
    allow_mock: bool = False,
) -> list[Path]:
    """Select 6 runs from results.json + traces; write transcripts and EMPTY labels."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise CalibrateError(f"No results.json in {run_dir}")

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    mode = results.get("llm_mode", "")
    if mode != "real" and not allow_mock:
        raise CalibrateError(
            f"Refusing llm_mode={mode!r}. Pass --allow-mock only for tests "
            f"(output will be stamped MOCK, DO NOT LABEL)."
        )

    selected = _select_six_runs(results)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear prior generated pair files
    for old in out_dir.glob("transcript_*.txt"):
        old.unlink()
    for old in out_dir.glob("labels_*.yaml"):
        old.unlink()

    created: list[Path] = []
    stamp_mock = mode != "real"

    for i, (scenario_id, run_index, passed) in enumerate(selected, start=1):
        trace_file = run_dir / "traces" / f"{scenario_id}.jsonl"
        if not trace_file.exists():
            raise CalibrateError(f"Missing trace for {scenario_id}: {trace_file}")
        segments = _load_trace_segments(trace_file)
        if run_index >= len(segments):
            raise CalibrateError(
                f"run_index {run_index} out of range for {scenario_id} "
                f"({len(segments)} segments)"
            )
        body = _segment_to_transcript(segments[run_index])
        header_lines = [
            f"# Calibration transcript {i:02d}",
            f"# scenario_id: {scenario_id}",
            f"# run_index: {run_index}",
            f"# passed: {passed}",
            f"# source_run: {run_dir.as_posix()}",
        ]
        if stamp_mock:
            header_lines.insert(1, "# MOCK, DO NOT LABEL")
        transcript_path = out_dir / f"transcript_{i:02d}.txt"
        transcript_path.write_text(
            "\n".join(header_lines) + "\n\n" + body,
            encoding="utf-8",
        )
        created.append(transcript_path)

        labels_path = out_dir / f"labels_{i:02d}.yaml"
        label_lines = [
            f"# Calibration labels for transcript_{i:02d}.txt",
            "# HUMAN fills human_labels. Do not invent values.",
        ]
        if stamp_mock:
            label_lines.append("# MOCK, DO NOT LABEL")
        label_lines.extend([
            f"transcript: transcript_{i:02d}.txt",
            f"scenario_id: {scenario_id}",
            f"run_index: {run_index}",
            "judge_labels: {}",
            "human_labels: {}",
        ])
        labels_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        created.append(labels_path)

    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge calibration agree-rate")
    parser.add_argument(
        "--dir",
        default="data/calibration",
        help="Calibration directory with transcripts + label YAMLs",
    )
    parser.add_argument(
        "--from-run",
        dest="from_run",
        default=None,
        help="Run directory with results.json + traces/ to generate 6 transcripts",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow llm_mode!=real (stamps MOCK, DO NOT LABEL)",
    )
    args = parser.parse_args(argv)

    if args.from_run:
        try:
            created = generate_from_run(
                args.from_run, args.dir, allow_mock=args.allow_mock
            )
        except CalibrateError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {len(created)} files under {args.dir}")
        print("HUMAN: fill human_labels in each labels_XX.yaml, then re-run without --from-run.")
        return

    try:
        result = compute_agree_rate(args.dir)
    except CalibrateError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(
        {k: v for k, v in result.items() if k != "details"},
        indent=2,
    ))
    if result["unlabeled_files"]:
        print(
            f"\nNote: {result['unlabeled_files']} unlabeled file(s). "
            f"Agree-rate requires HUMAN-filled human_labels.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
