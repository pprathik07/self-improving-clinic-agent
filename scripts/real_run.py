"""Orchestrate a real baseline eval (human runs this).

Steps stop on failure with a clear message. Pure gate helpers are unit-tested.

Usage (human):
  uv run python scripts/real_run.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_mock_off_and_distinct_models(env: dict[str, str] | None = None) -> None:
    src = env if env is not None else os.environ
    if src.get("MOCK_LLM", "0") != "0":
        raise RuntimeError("Refuse: MOCK_LLM must be 0 for a real run")
    models = [src.get("AGENT_MODEL"), src.get("JUDGE_MODEL"), src.get("SIM_MODEL")]
    if not all(models) or len(set(models)) < 3:
        raise RuntimeError(f"Refuse: AGENT/JUDGE/SIM must be three different models; got {models}")


def baseline_quality_ok(results: dict) -> tuple[bool, str]:
    """Validate baseline quality. Returns (ok, message)."""
    if results.get("llm_mode") != "real":
        return False, f"llm_mode must be real, got {results.get('llm_mode')!r}"
    if int(results.get("k") or 0) < 3:
        return False, f"k must be >= 3, got {results.get('k')}"
    err = 0
    for d in results.get("per_scenario", {}).values():
        for run in d.get("runs", []):
            if run.get("error"):
                err += 1
    if err:
        return False, f"baseline has {err} error run(s)"
    train_fail = 0
    for d in results.get("per_scenario", {}).values():
        if d.get("split") == "train" and float(d.get("pass_rate", 1.0)) < 1.0:
            train_fail += 1
    if train_fail < 2:
        return False, (
            f"v1 passes too much: only {train_fail} train failures "
            f"(need >=2). Review scenarios_candidates/ before continuing"
        )
    return True, f"ok ({train_fail} train failures)"


def find_latest_run_dir(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    cands = []
    for child in sorted(runs_dir.iterdir()):
        if child.is_dir() and child.name not in ("_mock", "_invalid"):
            if (child / "results.json").exists():
                cands.append(child)
    return cands[-1] if cands else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke", action="store_true", help="tests only")
    parser.add_argument("--skip-eval", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    # 1) smoke
    if not args.skip_smoke:
        print("=== Step 1: smoke_llm ===")
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "smoke_llm.py")], cwd=str(ROOT))
        if rc != 0:
            print("STOP: smoke_llm failed")
            return rc

    # 2) freeze --check
    print("=== Step 2: freeze_manifest --check ===")
    rc = subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "freeze_manifest.py"), "--check"],
        cwd=str(ROOT),
    )
    if rc != 0:
        print("STOP: freeze_manifest --check failed (run freeze_manifest.py first)")
        return rc

    # 3) env gate
    print("=== Step 3: MOCK_LLM=0 and distinct models ===")
    try:
        require_mock_off_and_distinct_models()
    except RuntimeError as e:
        print(f"STOP: {e}")
        return 1

    # 4) eval
    if not args.skip_eval:
        print("=== Step 4: real eval k=3 ===")
        rc = subprocess.call(
            [
                "uv",
                "run",
                "python",
                "-m",
                "clinic_agent.evals",
                "policy/policy_v1.yaml",
                "3",
            ],
            cwd=str(ROOT),
        )
        if rc != 0:
            print("STOP: eval failed")
            return rc

    # 5) baseline quality
    print("=== Step 5: baseline quality gate ===")
    run_dir = find_latest_run_dir(ROOT / "runs")
    if run_dir is None:
        print("STOP: no results.json found under runs/")
        return 1
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    ok, msg = baseline_quality_ok(data)
    if not ok:
        print(f"STOP: {msg}")
        return 1
    print(msg)

    # 6) calibrate
    print("=== Step 6: calibrate --from-run ===")
    rc = subprocess.call(
        [
            "uv",
            "run",
            "python",
            "-m",
            "clinic_agent.evals.calibrate",
            "--from-run",
            str(run_dir),
        ],
        cwd=str(ROOT),
    )
    if rc != 0:
        print("STOP: calibrate failed")
        return rc

    # 7) next human steps
    print(
        """
=== Next human steps ===
1. Read {run}/failures.md
2. Fill data/calibration/labels_*.yaml
3. Run: uv run python -m clinic_agent.loop.improve
""".format(run=run_dir)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
