"""Hand-mutation table runner for non-guard logic.

Applies ONE exact-string replacement per entry, runs named test files, restores
the file (verified by sha256). Human runs mutations; --dry-run only checks patterns.

Usage (from repo root):
  uv run python scripts/mutation_table.py --dry-run
  uv run python scripts/mutation_table.py --list
  uv run python scripts/mutation_table.py --only R1
  uv run python scripts/mutation_table.py --expect-hashes
"""

from __future__ import annotations

import argparse
import hashlib
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Entry:
    id: str
    file: str
    old_text: str
    new_text: str
    test_files: tuple[str, ...]
    expected_substring: str


# Exact old_text must appear once in the file. Keep strings unique.
ENTRIES: list[Entry] = [
    # --- reflector ---
    Entry(
        "R1",
        "clinic_agent/loop/reflector.py",
        '        if data.get("split") != "train":\n            continue\n',
        '        if False and data.get("split") != "train":\n            continue\n',
        ("tests/test_reflector.py",),
        "heldout",
    ),
    Entry(
        "R2",
        "clinic_agent/loop/reflector.py",
        '            if scenario.get("is_infra_error", False):\n                continue\n',
        '            if False and scenario.get("is_infra_error", False):\n                continue\n',
        ("tests/test_reflector.py",),
        "infra",
    ),
    Entry(
        "R3",
        "clinic_agent/loop/reflector.py",
        "    _check_for_heldout_leak(prompt, all_scenarios_meta)\n",
        "    pass  # mutated: skipped heldout leak check\n",
        ("tests/test_reflector.py",),
        "heldout",
    ),
    Entry(
        "R4",
        "clinic_agent/loop/reflector.py",
        '            if len(words) < 8:\n'
        '                continue\n'
        '            for i in range(len(words) - 7):\n'
        '                phrase = " ".join(words[i:i + 8])\n',
        '            if len(words) < 80:\n'
        '                continue\n'
        '            for i in range(len(words) - 79):\n'
        '                phrase = " ".join(words[i:i + 80])\n',
        ("tests/test_reflector.py",),
        "8",
    ),
    Entry(
        "R5",
        "clinic_agent/loop/reflector.py",
        '            if len(words) < 8:\n'
        '                continue\n'
        '            for i in range(len(words) - 7):\n'
        '                phrase = " ".join(words[i:i + 8])\n',
        '            if len(words) < 6:\n'
        '                continue\n'
        '            for i in range(len(words) - 5):\n'
        '                phrase = " ".join(words[i:i + 6])\n',
        ("tests/test_reflector.py",),
        "8",
    ),
    Entry(
        "R6",
        "clinic_agent/loop/reflector.py",
        "    for attempt in range(2):\n",
        "    for attempt in range(1):\n",
        ("tests/test_reflector.py",),
        "retry",
    ),
    Entry(
        "R7",
        "clinic_agent/loop/reflector.py",
        '            if not run.get("passed") and not run.get("error"):\n',
        '            if run.get("passed") and not run.get("error"):\n',
        ("tests/test_reflector.py",),
        "fail",
    ),
    Entry(
        "R8",
        "clinic_agent/loop/reflector.py",
        "        capped = _cap_entries(collapsed, limit=30)\n",
        "        capped = collapsed  # mutated: bypass 30-entry cap\n",
        ("tests/test_reflector.py",),
        "cap",
    ),
    Entry(
        "R9",
        "clinic_agent/loop/reflector.py",
        "    # 1) Decide reported scenarios from results.json alone\n",
        "    # 1) MUTATED open traces before filtering — force any traced scenario into report path\n"
        "    for _sid, _data in list(per_scenario.items()):\n"
        "        _tf = run_dir / \"traces\" / f\"{_sid}.jsonl\"\n"
        "        if _tf.exists() and _data.get(\"split\") != \"train\":\n"
        "            per_scenario[_sid] = {\n"
        "                **_data,\n"
        "                \"split\": \"train\",\n"
        "                \"pass_rate\": 0.0,\n"
        "            }\n"
        "    # 1) Decide reported scenarios from results.json alone\n",
        ("tests/test_reflector.py",),
        "heldout",
    ),
    Entry(
        "R10",
        "clinic_agent/loop/reflector.py",
        '            line = f"- {layer}: {key}: {detail}"\n',
        '            line = f"- {layer}: {key}"  # mutated: drop detail\n',
        ("tests/test_reflector.py",),
        "detail",
    ),
    Entry(
        "R11",
        "clinic_agent/loop/reflector.py",
        '        if runs and all(r.get("error") for r in runs):\n',
        '        if False and runs and all(r.get("error") for r in runs):\n',
        ("tests/test_reflector.py",),
        "all-error",
    ),
    # --- gate ---
    Entry(
        "G1",
        "clinic_agent/loop/gate.py",
        "        if new_rate <= old_rate:\n",
        "        if new_rate < old_rate:\n",
        ("tests/test_gate.py",),
        "improve",
    ),
    Entry(
        "G2",
        "clinic_agent/loop/gate.py",
        "        if new_rate < old_rate or new_pass < old_pass:\n",
        "        if (new_rate < old_rate or new_pass < old_pass) and (old_pass - new_pass) > 1:\n",
        ("tests/test_gate.py",),
        "drop",
    ),
    Entry(
        "G3",
        "clinic_agent/loop/gate.py",
        "    if new_heldout < old_heldout:\n",
        "    if False and new_heldout < old_heldout:\n",
        ("tests/test_gate.py",),
        "heldout",
    ),
    Entry(
        "G4",
        "clinic_agent/loop/gate.py",
        "    if before_errors > 0 or after_errors > 0:\n",
        "    if before_errors > 0:  # mutated: ignore after-side errors\n",
        ("tests/test_gate.py",),
        "error",
    ),
    Entry(
        "G5",
        "clinic_agent/loop/gate.py",
        '                f"ADVISORY: turn count increased by {turn_increase:.0%} "\n'
        '                f"({old_turns} -> {new_turns}), exceeds +20% (does not reject)"\n'
        "            )\n",
        '                f"REJECT: turn count increased by {turn_increase:.0%} "\n'
        '                f"({old_turns} -> {new_turns}), exceeds +20%"\n'
        "            )\n"
        "            accept = False\n",
        ("tests/test_gate.py",),
        "advisory",
    ),
    Entry(
        "G6",
        "clinic_agent/loop/gate.py",
        "    if not allow_mock:\n"
        '        if before_mode != "real" or after_mode != "real":\n'
        "            raise GateError(\n"
        '                f"Gate refuses non-real results "\n'
        '                f"(before.llm_mode={before_mode!r}, after.llm_mode={after_mode!r}). "\n'
        '                f"Pass allow_mock=True only in tests."\n'
        "            )\n",
        "    if False and not allow_mock:\n"
        '        if before_mode != "real" or after_mode != "real":\n'
        "            raise GateError(\n"
        '                f"Gate refuses non-real results "\n'
        '                f"(before.llm_mode={before_mode!r}, after.llm_mode={after_mode!r}). "\n'
        '                f"Pass allow_mock=True only in tests."\n'
        "            )\n",
        ("tests/test_gate.py",),
        "mock",
    ),
    Entry(
        "G7",
        "clinic_agent/loop/gate.py",
        '        if before_mode != "real" or after_mode != "real":\n'
        "            raise GateError(\n"
        '                f"Gate refuses non-real results "\n'
        '                f"(before.llm_mode={before_mode!r}, after.llm_mode={after_mode!r}). "\n'
        '                f"Pass allow_mock=True only in tests."\n'
        "            )\n",
        '        if before_mode != "real":\n'
        "            raise GateError(\n"
        '                f"Gate refuses non-real results "\n'
        '                f"(before.llm_mode={before_mode!r}, after.llm_mode={after_mode!r}). "\n'
        '                f"Pass allow_mock=True only in tests."\n'
        "            )\n",
        ("tests/test_gate.py",),
        "mock",
    ),
    Entry(
        "G8",
        "clinic_agent/loop/gate.py",
        '    if results.get("invalid"):\n'
        '        return max(1, int(results.get("error_runs", 1)))\n',
        "    if False:  # mutated: ignore top-level invalid\n"
        '        return max(1, int(results.get("error_runs", 1)))\n',
        ("tests/test_gate.py",),
        "invalid",
    ),
    Entry(
        "G9",
        "clinic_agent/loop/gate.py",
        "        if old_total <= 0 or old_rate < 1.0:\n",
        "        if old_total <= 0 or old_rate < 0.5:\n",
        ("tests/test_gate.py",),
        "fully",
    ),
    Entry(
        "G10",
        "clinic_agent/loop/gate.py",
        "    # Empty targets reject\n"
        "    if not targets:\n"
        '        reasons.append("REJECT: no target scenarios")\n'
        "        return Decision(accept=False, reasons=reasons, metrics=metrics)\n",
        "    # Empty targets reject REMOVED\n"
        "    if False and not targets:\n"
        '        reasons.append("REJECT: no target scenarios")\n'
        "        return Decision(accept=False, reasons=reasons, metrics=metrics)\n",
        ("tests/test_gate.py",),
        "target",
    ),
    Entry(
        "G11",
        "clinic_agent/loop/gate.py",
        "    # k mismatch\n"
        '    before_has_k = "k" in before\n'
        '    after_has_k = "k" in after\n'
        "    if before_has_k and after_has_k:\n"
        '        if before.get("k") != after.get("k"):\n'
        "            raise GateError(\n"
        '                f"k mismatch: before.k={before.get(\'k\')!r} after.k={after.get(\'k\')!r}"\n'
        "            )\n"
        "    elif before_has_k or after_has_k:\n"
        "        raise GateError(\n"
        '            f"k present on only one side "\n'
        '            f"(before_has_k={before_has_k}, after_has_k={after_has_k})"\n'
        "        )\n",
        "    # k mismatch REMOVED\n",
        ("tests/test_gate.py",),
        "k mismatch",
    ),
    Entry(
        "G12",
        "clinic_agent/loop/gate.py",
        "    # Scenario-set mismatch\n"
        '    before_ids = set(before.get("per_scenario", {}))\n'
        '    after_ids = set(after.get("per_scenario", {}))\n'
        "    if before_ids != after_ids:\n"
        "        only_before = sorted(before_ids - after_ids)\n"
        "        only_after = sorted(after_ids - before_ids)\n"
        "        raise GateError(\n"
        '            f"Scenario set mismatch: only_before={only_before}, only_after={only_after}"\n'
        "        )\n",
        "    # Scenario-set mismatch REMOVED\n",
        ("tests/test_gate.py",),
        "Scenario set",
    ),
    # --- improve ---
    Entry(
        "I1",
        "clinic_agent/loop/improve.py",
        "    if gate.accept:\n",
        "    if True:  # mutated: skip gate, always accept\n",
        ("tests/test_improve.py",),
        "gate",
    ),
    Entry(
        "I2",
        "clinic_agent/loop/improve.py",
        '        shutil.move(str(new_policy_path), str(rejected))\n',
        "        pass  # mutated: skip rollback move\n",
        ("tests/test_improve.py",),
        "rollback",
    ),
    Entry(
        "I3",
        "clinic_agent/loop/improve.py",
        '    if not _prompt_yes_no(\n'
        '        "Review failures.md. Continue to reflector? [y/N]",\n'
        "        yes=yes,\n"
        "        stdin_fn=stdin_fn,\n"
        "    ):\n",
        "    if False:  # mutated: drop first y/N prompt\n",
        ("tests/test_improve.py",),
        "reflector",
    ),
    Entry(
        "I4",
        "clinic_agent/loop/improve.py",
        '    if mode != "real" and not allow_mock:\n',
        '    if False and mode != "real" and not allow_mock:\n',
        ("tests/test_improve.py",),
        "mock",
    ),
    Entry(
        "I5",
        "clinic_agent/loop/improve.py",
        "    new_version = _next_version(policy_dir)\n",
        "    new_version = 1  # mutated: overwrite v1\n",
        ("tests/test_improve.py",),
        "v1",
    ),
    Entry(
        "I6",
        "clinic_agent/loop/improve.py",
        '    if "k" not in baseline or baseline["k"] is None:\n'
        "        raise ImproveError(\n"
        '            "Baseline results.json has no usable \'k\'. "\n'
        '            "Cannot re-run eval without the baseline k."\n'
        "        )\n"
        '    k = int(baseline["k"])\n',
        "    k = int(baseline.get(\"k\") or 3)  # mutated: silent k fallback\n",
        ("tests/test_improve.py",),
        "usable",
    ),
    Entry(
        "I7",
        "clinic_agent/loop/improve.py",
        "    # Refuse baselines that contain any error run (matches gate no-error-runs rule)\n"
        '    for scenario_id, data in results.get("per_scenario", {}).items():\n'
        '        for run in data.get("runs", []):\n'
        '            if run.get("error"):\n'
        "                raise ImproveError(\n"
        '                    f"Baseline at {run_dir} has error run(s) "\n'
        '                    f"(scenario {scenario_id}). Refusing invalid baseline."\n'
        "                )\n",
        "    # Refuse error baselines REMOVED\n",
        ("tests/test_improve.py",),
        "error run",
    ),
    # --- llm / run_eval ---
    Entry(
        "L1",
        "clinic_agent/llm.py",
        "_RATE_LIMIT_SLEEPS = (10, 30, 60)\n",
        "_RATE_LIMIT_SLEEPS = (2, 4, 8)\n",
        ("tests/test_llm_errors.py", "tests/test_run_eval.py"),
        "10",
    ),
    Entry(
        "L2",
        "clinic_agent/evals/run_eval.py",
        "            except RuntimeError as e:\n"
        "                # Non-fatal infra RuntimeError — mark run, continue\n",
        "            except RuntimeError as e:\n"
        '                if "404" in str(e) or "QUOTA" in str(e).upper():\n'
        "                    print(f\"\\n\\nFATAL (mutated substring): {e}\")\n"
        "                    import shutil as _sh\n"
        "                    _sh.rmtree(run_dir, ignore_errors=True)\n"
        "                    return {}\n"
        "                # Non-fatal infra RuntimeError — mark run, continue\n",
        ("tests/test_llm_errors.py", "tests/test_run_eval.py"),
        "404",
    ),
    # --- patch (temporary mutate only; do not edit permanently) ---
    Entry(
        "P1",
        "clinic_agent/loop/patch.py",
        '                new_policy[change.policy_section] = {"text": f"{existing_text.rstrip()}\\n{change.text}"}\n',
        '                new_policy[change.policy_section] = {"text": change.text}  # mutated: add overwrites\n',
        ("tests/test_patch.py",),
        "append",
    ),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_newlines(text: str) -> str:
    """LF-normalize so patterns written with \\n match CRLF working trees."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def encode_with_original_newlines(text_lf: str, original: bytes) -> bytes:
    """Re-encode LF text using CRLF if the original file used CRLF."""
    if b"\r\n" in original:
        return text_lf.replace("\n", "\r\n").encode("utf-8")
    return text_lf.encode("utf-8")


def count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    haystack = normalize_newlines(haystack)
    needle = normalize_newlines(needle)
    count = 0
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        count += 1
        start = idx + len(needle)
    return count


def recorded_hashes() -> dict[str, str]:
    """sha256 of every unique target file as currently on disk."""
    out: dict[str, str] = {}
    for e in ENTRIES:
        p = ROOT / e.file
        out[e.file] = sha256_file(p)
    return out


EXPECT_HASHES: dict[str, str] | None = None  # filled at --expect-hashes from disk at start


def parse_failed_tests(stdout: str, stderr: str) -> list[str]:
    failed: list[str] = []
    for stream in (stdout, stderr):
        for ln in stream.splitlines():
            if ln.startswith("FAILED "):
                name = ln[len("FAILED ") :].split(" - ")[0].strip()
                if name:
                    failed.append(name)
    return failed


def run_pytest(test_files: tuple[str, ...]) -> tuple[str, list[str], str]:
    """Returns (status, failed_names, raw_output).

    status: CAUGHT | UNTESTED | INVALID | COLLECTION_ERROR
    """
    cmd = [
        "uv",
        "run",
        "pytest",
        *test_files,
        "-q",
        "--tb=no",
        "-rf",
    ]
    env = {**os.environ, "MOCK_LLM": "1"}
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    raw = (proc.stdout or "") + (proc.stderr or "")
    # Collection / import errors
    if "ERROR collecting" in raw or "Interrupted:" in raw and "error" in raw.lower():
        return "INVALID", [], raw
    if proc.returncode == 2:
        return "INVALID", [], raw
    failed = parse_failed_tests(proc.stdout or "", proc.stderr or "")
    if failed:
        return "CAUGHT", failed, raw
    if proc.returncode != 0:
        # Non-zero but no FAILED lines — treat as invalid/collection
        return "INVALID", [], raw
    return "UNTESTED", [], raw


def ensure_suite_green() -> None:
    env = {**os.environ, "MOCK_LLM": "1"}
    proc = subprocess.run(
        ["uv", "run", "pytest", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print("REFUSE: suite is not fully green before mutating.")
        print(out)
        raise SystemExit(2)
    print("Pre-check: uv run pytest -q is green.")


def dry_run(entries: list[Entry]) -> int:
    ok = True
    for e in entries:
        path = ROOT / e.file
        text = normalize_newlines(path.read_text(encoding="utf-8"))
        n = count_occurrences(text, e.old_text)
        if n == 1:
            print(f"OK  {e.id}: exact match once in {e.file}")
        else:
            ok = False
            print(f"PATTERN NOT FOUND/AMBIGUOUS  {e.id}: count={n} in {e.file}")
    return 0 if ok else 1


def run_entries(entries: list[Entry], *, expect_hashes: bool) -> int:
    ensure_suite_green()
    if expect_hashes:
        current = recorded_hashes()
        # Compare to hashes recorded at module load time via --expect-hashes file?
        # Spec: refuse if any target file does not match its recorded sha256 when
        # --expect-hashes is passed. We treat "recorded" as hashes computed at the
        # start of this run (after green check) and re-check before each mutation.
        baseline_hashes = current
        print("Recorded expect-hashes:")
        for f, h in sorted(baseline_hashes.items()):
            print(f"  {f}: {h}")
    else:
        baseline_hashes = recorded_hashes()

    rows: list[dict[str, str]] = []
    temp_dir = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "mutation_table_backups"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for e in entries:
            path = ROOT / e.file
            original = path.read_bytes()
            original_hash = sha256_bytes(original)
            if expect_hashes and baseline_hashes.get(e.file) != original_hash:
                print(f"REFUSE: {e.file} hash mismatch vs recorded expect-hashes.")
                return 2

            text = normalize_newlines(original.decode("utf-8"))
            old = normalize_newlines(e.old_text)
            new = normalize_newlines(e.new_text)
            n = count_occurrences(text, old)
            if n != 1:
                print(f"PATTERN NOT FOUND/AMBIGUOUS  {e.id}: count={n} — skip")
                rows.append({
                    "id": e.id,
                    "file": e.file,
                    "change": e.id,
                    "result": "SKIPPED_AMBIGUOUS",
                    "failing": "",
                    "restored": "n/a",
                })
                continue

            backup_path = temp_dir / f"{e.id}_{path.name}.bak"
            backup_path.write_bytes(original)

            mutated_text = text.replace(old, new, 1)
            result = "INVALID"
            failing: list[str] = []
            caught_flag = ""
            try:
                path.write_bytes(encode_with_original_newlines(mutated_text, original))
                try:
                    py_compile.compile(str(path), doraise=True)
                except py_compile.PyCompileError as err:
                    print(f"{e.id}: INVALID (syntax): {err}")
                    result = "INVALID"
                else:
                    status, failing, _raw = run_pytest(e.test_files)
                    result = status
                    if status == "CAUGHT":
                        hit = any(e.expected_substring.lower() in f.lower() for f in failing)
                        caught_flag = "yes" if hit else "no"
                        print(
                            f"{e.id}: CAUGHT ({len(failing)} failures; "
                            f"expected_substring={e.expected_substring!r} matched={hit})"
                        )
                        for f in failing:
                            print(f"  FAILED {f}")
                    else:
                        print(f"{e.id}: {status}")
            except KeyboardInterrupt:
                print("Interrupted — restoring...")
                raise
            except Exception:
                print(f"{e.id}: exception during mutation run:")
                traceback.print_exc()
                result = "INVALID"
            finally:
                path.write_bytes(original)
                restored_hash = sha256_file(path)
                if restored_hash != original_hash:
                    # Restore from temp backup
                    path.write_bytes(backup_path.read_bytes())
                    restored_hash = sha256_file(path)
                    if restored_hash != original_hash:
                        print("RESTORE FAILED")
                        rows.append({
                            "id": e.id,
                            "file": e.file,
                            "change": e.id,
                            "result": "RESTORE_FAILED",
                            "failing": "; ".join(failing),
                            "restored": "NO",
                        })
                        _write_report(rows)
                        return 3
                restored_ok = "yes" if restored_hash == original_hash else "NO"

            fail_s = "; ".join(failing)
            if result == "CAUGHT" and caught_flag:
                fail_s = f"{fail_s} (substr_match={caught_flag})"
            rows.append({
                "id": e.id,
                "file": e.file,
                "change": e.id,
                "result": result,
                "failing": fail_s,
                "restored": restored_ok,
            })
    finally:
        # Final verify all targets
        pass

    _write_report(rows)
    print("\nFinal sha256 of every target file:")
    seen: set[str] = set()
    for e in ENTRIES:
        if e.file in seen:
            continue
        seen.add(e.file)
        print(f"  {e.file}: {sha256_file(ROOT / e.file)}")
    return 0


def _write_report(rows: list[dict[str, str]]) -> None:
    lines = [
        "| id | file | change | result | failing tests | restored hash OK |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['file']} | {r['change']} | {r['result']} | "
            f"{r['failing']} | {r['restored']} |"
        )
    md = "\n".join(lines) + "\n"
    print("\n" + md)
    (ROOT / "mutation_report.md").write_text(md, encoding="utf-8")
    print(f"Wrote {ROOT / 'mutation_report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only verify patterns match once")
    parser.add_argument("--only", type=str, help="Run a single entry id")
    parser.add_argument("--list", action="store_true", help="List entry ids")
    parser.add_argument(
        "--expect-hashes",
        action="store_true",
        help="Refuse if target file hashes change mid-run vs start",
    )
    args = parser.parse_args(argv)

    entries = list(ENTRIES)
    if args.list:
        for e in entries:
            print(f"{e.id}\t{e.file}\t{e.expected_substring}\t{','.join(e.test_files)}")
        return 0

    if args.only:
        entries = [e for e in entries if e.id == args.only]
        if not entries:
            print(f"Unknown id: {args.only}")
            return 1

    if args.dry_run:
        return dry_run(entries)

    return run_entries(entries, expect_hashes=args.expect_hashes)


if __name__ == "__main__":
    raise SystemExit(main())
