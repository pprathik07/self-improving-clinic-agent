"""Improvement loop — baseline -> failures.md -> reflector -> patch -> gate.

Entry point: python -m clinic_agent.loop.improve
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml
from dotenv import load_dotenv

from clinic_agent.agent.policy import load_policy, policy_hash
from clinic_agent.evals.failures import write_failures_md
from clinic_agent.evals.report import render_report
from clinic_agent.evals.run_eval import run_eval
from clinic_agent.loop.gate import Decision, decide
from clinic_agent.loop.patch import (
    FailureCategory,
    Patch,
    PatchRejected,
    apply_patch,
    write_new_policy,
)
from clinic_agent.loop.reflector import ReflectorInput, build_reflector_input, reflect

logger = logging.getLogger(__name__)


class ImproveError(Exception):
    pass


def find_latest_baseline(
    runs_dir: str | Path = "runs",
    *,
    allow_mock: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Load the latest valid baseline results.json.

    Refuses mock/invalid unless allow_mock=True (tests only).
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise ImproveError(f"No runs directory at {runs_dir}")

    candidates: list[Path] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name in ("_mock", "_invalid"):
            if allow_mock and name == "_mock":
                for sub in sorted(child.iterdir()):
                    if (sub / "results.json").exists():
                        candidates.append(sub)
            continue
        if (child / "results.json").exists():
            candidates.append(child)

    if not candidates:
        raise ImproveError(
            "No valid baseline found. Run a real eval first "
            "(or pass --allow-mock in tests)."
        )

    # Newest by directory name (timestamps sort lexicographically)
    run_dir = candidates[-1]
    with open(run_dir / "results.json", encoding="utf-8") as f:
        results = json.load(f)

    mode = results.get("llm_mode", "")
    if mode != "real" and not allow_mock:
        raise ImproveError(
            f"Baseline at {run_dir} has llm_mode={mode!r}. "
            f"Refusing mock/invalid baselines unless --allow-mock."
        )
    if results.get("invalid"):
        raise ImproveError(f"Baseline at {run_dir} is marked invalid.")

    # Refuse baselines that contain any error run (matches gate no-error-runs rule)
    for scenario_id, data in results.get("per_scenario", {}).items():
        for run in data.get("runs", []):
            if run.get("error"):
                raise ImproveError(
                    f"Baseline at {run_dir} has error run(s) "
                    f"(scenario {scenario_id}). Refusing invalid baseline."
                )

    return run_dir, results


def _prompt_yes_no(message: str, *, yes: bool, stdin_fn: Callable[[str], str]) -> bool:
    if yes:
        print(f"{message} y (--yes)")
        return True
    answer = stdin_fn(f"{message} ").strip().lower()
    return answer in ("y", "yes")


def _detect_version(policy_path: str | Path) -> int:
    match = re.search(r"v(\d+)", str(policy_path))
    return int(match.group(1)) if match else 1


def _next_version(policy_dir: Path) -> int:
    versions = []
    for p in policy_dir.glob("policy_v*.yaml"):
        m = re.search(r"v(\d+)", p.name)
        if m:
            versions.append(int(m.group(1)))
    return (max(versions) if versions else 1) + 1


def _unified_diff(old_text: str, new_text: str, old_name: str, new_name: str) -> str:
    import difflib
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
        )
    )


def _default_reflector_llm(system: str, message: str) -> str:
    """Call the real LLM via llm.py using REFLECTOR_MODEL."""
    from clinic_agent.llm import chat

    resp = chat(
        messages=[{"role": "user", "content": message}],
        system=system,
        model_key="REFLECTOR_MODEL",
        temperature=0.2,
    )
    return resp.content


def _append_changelog(
    *,
    changelog_path: Path,
    before: dict[str, Any],
    after: dict[str, Any] | None,
    patch: Patch | None,
    gate: Decision | None,
    continue_reflector: bool,
    apply_patch_decision: bool | None,
    old_version: int,
    new_version: int,
    status: str,
) -> None:
    """Append a CHANGELOG entry GENERATED FROM results.json (no hand-typed numbers)."""
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"\n## v{old_version} -> v{new_version} ({ts})\n",
        f"- status: {status}",
        f"- continue_to_reflector: {'y' if continue_reflector else 'n'}",
    ]
    if apply_patch_decision is not None:
        lines.append(f"- apply_patch: {'y' if apply_patch_decision else 'n'}")
    lines.append(f"- before_policy_hash: `{before.get('policy_hash', '')}`")
    if after:
        lines.append(f"- after_policy_hash: `{after.get('policy_hash', '')}`")
    lines.append(f"- before_llm_mode: `{before.get('llm_mode', '')}`")
    if after:
        lines.append(f"- after_llm_mode: `{after.get('llm_mode', '')}`")

    if patch:
        lines.append("- patch_json:")
        lines.append("```json")
        lines.append(patch.model_dump_json(indent=2))
        lines.append("```")

    # Before/after table from results.json via report.py
    try:
        table = render_report(
            before,
            after,
            allow_mock=(before.get("llm_mode") != "real")
            or (after is not None and after.get("llm_mode") != "real"),
        )
        lines.append("")
        lines.append(table)
    except Exception as e:
        lines.append(f"- report_error: {e}")

    if gate:
        lines.append("")
        lines.append(f"### Gate: {'ACCEPT' if gate.accept else 'REJECT'}")
        for r in gate.reasons:
            lines.append(f"- {r}")

    lines.append("")
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_improvement_loop(
    *,
    policy_path: str = "policy/policy_v1.yaml",
    runs_dir: str | Path = "runs",
    yes: bool = False,
    allow_mock: bool = False,
    reflector_llm: Callable[[str, str], str] | None = None,
    eval_fn: Callable[..., dict[str, Any]] | None = None,
    stdin_fn: Callable[[str], str] | None = None,
    changelog_path: Path | None = None,
) -> dict[str, Any]:
    """Run one improvement iteration. Returns a summary dict for tests."""
    stdin_fn = stdin_fn or input
    eval_fn = eval_fn or run_eval
    reflector_llm = reflector_llm or _default_reflector_llm
    changelog_path = changelog_path or Path("loop/CHANGELOG.md")
    policy_dir = Path(policy_path).parent

    run_dir, baseline = find_latest_baseline(runs_dir, allow_mock=allow_mock)

    # --yes is CI-only: refuse when baseline is real
    if yes and baseline.get("llm_mode") == "real":
        raise ImproveError(
            "--yes is CI-only and refused when baseline llm_mode is real. "
            "Review failures.md and answer the prompts interactively."
        )

    # Write / refresh failures.md
    failures_path = write_failures_md(run_dir, baseline)
    print(f"\nWrote {failures_path}")
    failing = [
        (sid, d)
        for sid, d in baseline.get("per_scenario", {}).items()
        if d.get("split") == "train" and d.get("pass_rate", 1.0) < 1.0
    ]
    print(f"Train failures: {len(failing)}")
    for sid, d in failing:
        print(f"  - {sid}: {d.get('pass_rate', 0):.0%}")

    if not _prompt_yes_no(
        "Review failures.md. Continue to reflector? [y/N]",
        yes=yes,
        stdin_fn=stdin_fn,
    ):
        _append_changelog(
            changelog_path=changelog_path,
            before=baseline,
            after=None,
            patch=None,
            gate=None,
            continue_reflector=False,
            apply_patch_decision=None,
            old_version=_detect_version(policy_path),
            new_version=_detect_version(policy_path),
            status="STOPPED_AT_REFLECTOR_PROMPT",
        )
        return {"status": "stopped_reflector", "run_dir": str(run_dir)}

    current_policy = load_policy(policy_path)
    reflector_input = build_reflector_input(baseline, run_dir, current_policy)
    if not reflector_input.failures:
        print("No train failures for reflector (after filters). Stopping.")
        return {"status": "no_failures", "run_dir": str(run_dir)}

    print(f"\nCalling reflector (REFLECTOR_MODEL={os.environ.get('REFLECTOR_MODEL', 'unset')})...")
    patch = reflect(reflector_input, reflector_llm)
    print("\nPatch proposed:")
    print(patch.model_dump_json(indent=2))

    # Validate via patch.py
    applied = apply_patch(current_policy, patch)

    # needs_code_fix with empty changes: human TODO, do not apply
    if isinstance(applied, Patch) or (
        patch.failure_category == FailureCategory.NEEDS_CODE_FIX and not patch.changes
    ):
        print("\nHUMAN TODO: failure_category=needs_code_fix with empty changes.")
        print("No policy file written. Fix in code, then re-run eval.")
        _append_changelog(
            changelog_path=changelog_path,
            before=baseline,
            after=None,
            patch=patch,
            gate=None,
            continue_reflector=True,
            apply_patch_decision=False,
            old_version=_detect_version(policy_path),
            new_version=_detect_version(policy_path),
            status="NEEDS_CODE_FIX",
        )
        return {"status": "needs_code_fix", "patch": patch}

    new_version = _next_version(policy_dir)
    old_version = _detect_version(policy_path)

    # Show unified diff before writing
    old_text = Path(policy_path).read_text(encoding="utf-8")
    new_yaml = yaml.safe_dump(applied, sort_keys=False, allow_unicode=True)
    print("\nUnified diff (proposed):")
    print(_unified_diff(old_text, new_yaml, f"policy_v{old_version}.yaml", f"policy_v{new_version}.yaml"))

    if not _prompt_yes_no("Apply patch? [y/N]", yes=yes, stdin_fn=stdin_fn):
        _append_changelog(
            changelog_path=changelog_path,
            before=baseline,
            after=None,
            patch=patch,
            gate=None,
            continue_reflector=True,
            apply_patch_decision=False,
            old_version=old_version,
            new_version=new_version,
            status="STOPPED_AT_APPLY_PROMPT",
        )
        return {"status": "stopped_apply", "patch": patch}

    new_policy_path = write_new_policy(applied, new_version, policy_dir=policy_dir)
    print(f"Wrote {new_policy_path}")

    # Never re-run the baseline policy path as if it were the patched version
    if Path(new_policy_path).resolve() == Path(policy_path).resolve():
        raise ImproveError(
            f"Refusing to re-eval the baseline policy path {policy_path} as v{new_version}."
        )
    if new_version == 1 or Path(new_policy_path).name == "policy_v1.yaml":
        raise ImproveError("Refusing to evaluate policy_v1.yaml as a patched policy.")

    # Re-run full eval with baseline k (required; no silent fallback)
    if "k" not in baseline or baseline["k"] is None:
        raise ImproveError(
            "Baseline results.json has no usable 'k'. "
            "Cannot re-run eval without the baseline k."
        )
    k = int(baseline["k"])
    print(f"\nRe-running eval with {new_policy_path} (k={k})...")
    new_results = eval_fn(policy_path=str(new_policy_path), k=k)
    if not new_results:
        print("Eval returned empty results (invalid/aborted). Rolling back.")
        rejected = new_policy_path.with_suffix(".yaml.rejected")
        new_policy_path.rename(rejected)
        _append_changelog(
            changelog_path=changelog_path,
            before=baseline,
            after=None,
            patch=patch,
            gate=None,
            continue_reflector=True,
            apply_patch_decision=True,
            old_version=old_version,
            new_version=new_version,
            status="EVAL_INVALID_ROLLBACK",
        )
        return {"status": "eval_invalid", "active_policy": policy_path}

    gate = decide(
        baseline,
        new_results,
        patch.expected_to_fix,
        allow_mock=allow_mock,
    )
    print("\nGate:")
    for r in gate.reasons:
        print(f"  {r}")

    if gate.accept:
        print(f"\nGATE ACCEPTED. Active policy: {new_policy_path}")
        status = "ACCEPTED"
        active = str(new_policy_path)
        kept_path = str(new_policy_path)
    else:
        print(f"\nGATE REJECTED. Rolling back; keeping rejected file.")
        rejected = Path(str(new_policy_path) + ".rejected")
        shutil.move(str(new_policy_path), str(rejected))
        print(f"Rejected policy saved as {rejected}")
        print(f"Active policy remains: {policy_path}")
        status = "REJECTED_ROLLBACK"
        active = policy_path
        kept_path = str(rejected)

    _append_changelog(
        changelog_path=changelog_path,
        before=baseline,
        after=new_results,
        patch=patch,
        gate=gate,
        continue_reflector=True,
        apply_patch_decision=True,
        old_version=old_version,
        new_version=new_version,
        status=status,
    )
    return {
        "status": status.lower(),
        "active_policy": active,
        "new_policy": kept_path,
        "gate": gate,
        "patch": patch,
    }


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Policy improvement loop")
    parser.add_argument("policy", nargs="?", default="policy/policy_v1.yaml")
    parser.add_argument("--yes", action="store_true", help="CI-only; refused for real baselines")
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow mock baselines (tests only)",
    )
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args(argv)

    try:
        summary = run_improvement_loop(
            policy_path=args.policy,
            runs_dir=args.runs_dir,
            yes=args.yes,
            allow_mock=args.allow_mock,
        )
        print(f"\nDone: {summary.get('status')}")
    except (ImproveError, PatchRejected, Exception) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
