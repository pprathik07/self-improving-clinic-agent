"""Read-only preflight checks before real run / commit.

Prints: check | result | detail
Results: PASS, FAIL, PENDING, HUMAN
Exit non-zero on any FAIL.

Usage:
  uv run python scripts/preflight.py
  uv run python scripts/preflight.py --pre-real
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
MUTATION_CHECK_EXPECTED = (
    "ebdf0a7b28311ab396fd3a3cd1e4f81e23403c6db97b5e787c3d80fe4e045bce"
)
MOCK_NUMBER_PATTERNS = ("14/15", "93%", "89%", "6/6 (100%)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- Pure checks (testable) -------------------------------------------------

def check_pytest_green(
    run_pytest: Callable[[], tuple[int, str]] | None = None,
) -> tuple[str, str]:
    """a: pytest green with 0 skipped."""
    if run_pytest is None:
        def run_pytest() -> tuple[int, str]:
            env = {**os.environ, "MOCK_LLM": "1"}
            proc = subprocess.run(
                ["uv", "run", "pytest", "-q", "-rs"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    code, out = run_pytest()
    skipped = 0
    m = re.search(r"(\d+)\s+skipped", out)
    if m:
        skipped = int(m.group(1))
    # also -rs lines
    if re.search(r"\bSKIPPED\b", out) and skipped == 0:
        skipped = len(re.findall(r"SKIPPED", out))
    if code != 0:
        return "FAIL", f"pytest exit {code}"
    if skipped > 0:
        return "FAIL", f"{skipped} skipped"
    # extract pass count if present
    return "PASS", "0 failed, 0 skipped"


def check_patch_diff_empty(
    diff_text: str | None = None,
) -> tuple[str, str]:
    """b: git diff -- clinic_agent/loop/patch.py empty."""
    if diff_text is None:
        proc = subprocess.run(
            ["git", "diff", "--", "clinic_agent/loop/patch.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        diff_text = proc.stdout or ""
    if diff_text.strip():
        return "FAIL", "patch.py has local diff"
    return "PASS", "empty"


def check_mutation_check_hash(
    path: Path | None = None,
    expected: str = MUTATION_CHECK_EXPECTED,
) -> tuple[str, str]:
    """c: scripts/mutation_check.py sha256."""
    path = path or (ROOT / "scripts" / "mutation_check.py")
    if not path.exists():
        return "FAIL", "missing scripts/mutation_check.py"
    got = _sha256(path)
    if got != expected:
        return "FAIL", f"got {got}, expected {expected}"
    return "PASS", got


def check_mutation_report(
    report_path: Path | None = None,
) -> tuple[str, str]:
    """d: mutation_report.md exists, no UNTESTED/INVALID."""
    report_path = report_path or (ROOT / "mutation_report.md")
    if not report_path.exists():
        return "PENDING", "mutation_report.md missing (run scripts/mutation_table.py)"
    text = report_path.read_text(encoding="utf-8")
    if re.search(r"\|\s*UNTESTED\s*\|", text) or "UNTESTED" in text.split("|"):
        # simpler: any UNTESTED or INVALID as a result cell
        if "| UNTESTED |" in text or "| INVALID |" in text:
            return "PENDING", "mutation_report has UNTESTED or INVALID rows"
    if "| UNTESTED |" in text or "| INVALID |" in text:
        return "PENDING", "mutation_report has UNTESTED or INVALID rows"
    return "PASS", "no UNTESTED/INVALID rows"


def check_secrets(
    ls_files: list[str] | None = None,
    root: Path | None = None,
) -> tuple[str, str]:
    """e: no .env tracked; no AIza / AQ. keys; .env.example placeholders only."""
    root = root or ROOT
    if ls_files is None:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        ls_files = (proc.stdout or "").splitlines()
    if ".env" in ls_files:
        return "FAIL", ".env is tracked"
    if ".env.example" not in ls_files and not (root / ".env.example").exists():
        return "FAIL", ".env.example missing"
    # scan tracked files for key-like strings (sample readable text files)
    key_re = re.compile(r"AIza[0-9A-Za-z_-]{10,}|AQ\.[A-Za-z0-9_-]{20,}")
    for rel in ls_files:
        p = root / rel
        if not p.is_file():
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".db", ".lock"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if key_re.search(text):
            return "FAIL", f"secret-like pattern in {rel}"
    example = root / ".env.example"
    if example.exists():
        ex = example.read_text(encoding="utf-8")
        if key_re.search(ex) or re.search(r"GEMINI_API_KEY=\s*AIza", ex):
            return "FAIL", ".env.example looks like a real key"
        # placeholders: empty or your_ or CHANGE_ME style
        for line in ex.splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and not re.search(r"(your_|changeme|placeholder|xxx|<)", val, re.I):
                    if len(val) > 20 and re.match(r"^[A-Za-z0-9_.-]+$", val):
                        # still allow short placeholders
                        pass
    return "PASS", "no tracked .env; placeholders only"


def check_gitignore(text: str | None = None, root: Path | None = None) -> tuple[str, str]:
    """f: uv.lock NOT ignored; clinic.db, runs/_mock/, runs/_invalid/, .env ignored."""
    root = root or ROOT
    if text is None:
        text = (root / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if "uv.lock" in lines or "/uv.lock" in lines:
        return "FAIL", "uv.lock is ignored"
    needed = [".env", "clinic.db", "runs/_mock/", "runs/_invalid/"]
    missing = [n for n in needed if not any(n in ln or ln == n.rstrip("/") for ln in lines)]
    # more precise
    def ignored(pat: str) -> bool:
        return any(ln == pat or ln.endswith(pat) or ln == pat.rstrip("/") for ln in lines)

    bad = []
    if not ignored(".env"):
        bad.append(".env not ignored")
    if not ignored("clinic.db"):
        bad.append("clinic.db not ignored")
    if not any("runs/_mock" in ln for ln in lines):
        bad.append("runs/_mock/ not ignored")
    if not any("runs/_invalid" in ln for ln in lines):
        bad.append("runs/_invalid/ not ignored")
    if bad:
        return "FAIL", "; ".join(bad)
    return "PASS", "gitignore ok"


def check_runs_llm_mode(runs_dir: Path | None = None) -> tuple[str, str]:
    """g: every top-level folder with results.json has llm_mode real; mock/invalid outside _mock/_invalid = FAIL."""
    runs_dir = runs_dir or (ROOT / "runs")
    if not runs_dir.exists():
        return "PASS", "no runs/ yet"
    problems = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name in ("_mock", "_invalid"):
            continue
        # mock-named or invalid outside
        if name.startswith("_mock") or name.startswith("_invalid"):
            problems.append(f"unexpected {name}")
            continue
        results = child / "results.json"
        if not results.exists():
            # nested? also check if this looks like mock dumped at top
            continue
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"{name}: bad results.json")
            continue
        mode = data.get("llm_mode")
        if mode != "real":
            problems.append(f"{name}: llm_mode={mode!r}")
        if data.get("invalid"):
            problems.append(f"{name}: invalid=true outside _invalid")
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "runs llm_mode ok"


def check_frozen_hashes(
    manifest_path: Path | None = None,
    root: Path | None = None,
) -> tuple[str, str]:
    """h: scenario/scorer/policy_v1 hashes match frozen_hashes.json."""
    root = root or ROOT
    manifest_path = manifest_path or (root / "data" / "frozen_hashes.json")
    if not manifest_path.exists():
        return "PENDING", "data/frozen_hashes.json missing"
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import freeze_manifest as fm  # type: ignore
    except ImportError:
        # load by path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "freeze_manifest", ROOT / "scripts" / "freeze_manifest.py"
        )
        assert spec and spec.loader
        fm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fm)
    ok, problems = fm.check_manifest(manifest_path, root=root)
    if ok:
        return "PASS", "frozen hashes match"
    return "FAIL", "; ".join(problems)


def check_no_mock_numbers(
    root: Path | None = None,
    patterns: tuple[str, ...] = MOCK_NUMBER_PATTERNS,
) -> tuple[str, str]:
    """i: no mock numbers in README/design-note/decisions/CHANGELOG/docs."""
    root = root or ROOT
    files = [
        root / "README.md",
        root / "design-note.md",
        root / "decisions.md",
        root / "loop" / "CHANGELOG.md",
    ]
    docs = root / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.glob("*.md")))
    hits = []
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for pat in patterns:
            if pat in text:
                hits.append(f"{f.relative_to(root).as_posix()}:{pat}")
    if hits:
        return "FAIL", "; ".join(hits)
    return "PASS", "no mock numbers"


def check_checklist(path: Path | None = None) -> tuple[str, str]:
    """j: no '<failing test names>' placeholder; no [DONE marks."""
    path = path or (ROOT / "CHECKLIST.md")
    if not path.exists():
        return "FAIL", "CHECKLIST.md missing"
    text = path.read_text(encoding="utf-8")
    if "<failing test names>" in text:
        return "FAIL", "placeholder <failing test names> present"
    if "[DONE" in text:
        return "FAIL", "has [DONE mark"
    return "PASS", "checklist clean"


def check_decisions_ai_usage(
    decisions_path: Path | None = None,
    ai_usage_path: Path | None = None,
) -> tuple[str, str]:
    """k: no duplicate decision titles; no truncated '...'; no AI_USAGE 2/4/8 backoff."""
    decisions_path = decisions_path or (ROOT / "decisions.md")
    ai_usage_path = ai_usage_path or (ROOT / "AI_USAGE.md")
    problems = []
    if decisions_path.exists():
        titles = []
        for ln in decisions_path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^20\d\d-\d\d-\d\d \|", ln):
                # title = second field
                parts = ln.split("|")
                if len(parts) >= 2:
                    titles.append(parts[1].strip())
                if ln.rstrip().endswith("..."):
                    problems.append(f"truncated entry: {ln[:60]}")
        seen = set()
        for t in titles:
            if t in seen:
                problems.append(f"duplicate title: {t}")
            seen.add(t)
    if ai_usage_path.exists():
        ai = ai_usage_path.read_text(encoding="utf-8")
        if re.search(r"\b2\s*/\s*4\s*/\s*8\b", ai) or "sleeps 2/4/8" in ai or "2s, 4s, 8s" in ai:
            problems.append("AI_USAGE mentions old backoff 2/4/8")
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "decisions/AI_USAGE ok"


def check_readme_commands(text: str | None = None, root: Path | None = None) -> tuple[str, str]:
    """l: README top section contains agent, eval, improve, test commands."""
    root = root or ROOT
    if text is None:
        text = (root / "README.md").read_text(encoding="utf-8")
    # take first ~80 lines as "top"
    top = "\n".join(text.splitlines()[:80])
    needed = [
        ("agent", r"clinic_agent(?!\.)|make agent|-m clinic_agent\b"),
        ("eval", r"clinic_agent\.evals|make eval"),
        ("improve", r"clinic_agent\.loop\.improve|make improve"),
        ("test", r"pytest|make test"),
    ]
    missing = []
    for name, pat in needed:
        if not re.search(pat, top):
            missing.append(name)
    if missing:
        return "FAIL", f"missing commands: {missing}"
    return "PASS", "run commands present"


def check_real_baseline(runs_dir: Path | None = None) -> tuple[str, str]:
    """m: latest non-mock results.json real, 3 different models, zero errors, k>=3."""
    runs_dir = runs_dir or (ROOT / "runs")
    baseline = _latest_real_results(runs_dir)
    if baseline is None:
        return "PENDING", "no real baseline"
    path, data = baseline
    problems = []
    if data.get("llm_mode") != "real":
        problems.append(f"llm_mode={data.get('llm_mode')!r}")
    models = [data.get("agent_model"), data.get("judge_model"), data.get("sim_model")]
    if len([m for m in models if m]) < 3 or len(set(models)) < 3:
        problems.append(f"models not distinct: {models}")
    if int(data.get("k") or 0) < 3:
        problems.append(f"k={data.get('k')}")
    err = _error_run_count(data)
    if err:
        problems.append(f"{err} error runs")
    if problems:
        return "FAIL", f"{path}: " + "; ".join(problems)
    return "PASS", str(path)


def check_v1_fails_train(runs_dir: Path | None = None) -> tuple[str, str]:
    """n: baseline v1 fails >=2 TRAIN scenarios."""
    runs_dir = runs_dir or (ROOT / "runs")
    baseline = _latest_real_results(runs_dir)
    if baseline is None:
        return "PENDING", "no baseline"
    path, data = baseline
    failing = 0
    for sid, d in data.get("per_scenario", {}).items():
        if d.get("split") == "train" and float(d.get("pass_rate", 1.0)) < 1.0:
            # exclude all-error? still counts as failing for this gate
            failing += 1
    if failing < 2:
        return "FAIL", f"only {failing} train failures (need >=2) at {path}"
    return "PASS", f"{failing} train failures"


def check_changelog_and_v2(
    changelog: Path | None = None,
    policy_dir: Path | None = None,
) -> tuple[str, str]:
    """o: CHANGELOG has >=1 entry and policy_v2.yaml or .rejected exists."""
    changelog = changelog or (ROOT / "loop" / "CHANGELOG.md")
    policy_dir = policy_dir or (ROOT / "policy")
    if not changelog.exists():
        return "PENDING", "loop/CHANGELOG.md missing"
    text = changelog.read_text(encoding="utf-8")
    # entry: ## or dated line
    entries = [ln for ln in text.splitlines() if ln.startswith("## ") or re.match(r"^20\d\d-", ln)]
    if not entries:
        # also accept non-empty with status lines
        if len(text.strip()) < 20:
            return "PENDING", "CHANGELOG empty"
    v2 = policy_dir / "policy_v2.yaml"
    v2r = policy_dir / "policy_v2.yaml.rejected"
    if not v2.exists() and not v2r.exists():
        return "PENDING", "no policy_v2.yaml or .rejected yet"
    if not entries and len(text.strip()) < 20:
        return "PENDING", "CHANGELOG has no entries"
    return "PASS", "changelog + v2 present"


def check_design_note(path: Path | None = None) -> tuple[str, str]:
    """p: word count <=600 and no unfilled HUMAN: marker."""
    path = path or (ROOT / "design-note.md")
    if not path.exists():
        return "PENDING", "design-note.md missing"
    text = path.read_text(encoding="utf-8")
    words = re.findall(r"\b\w+\b", text)
    if "HUMAN:" in text or "HUMAN WRITES" in text:
        return "PENDING", "unfilled HUMAN marker"
    if len(words) > 600:
        return "FAIL", f"word count {len(words)} > 600"
    return "PASS", f"{len(words)} words"


def check_ai_usage_todo(path: Path | None = None) -> tuple[str, str]:
    """q: count rows with TODO (human fills) — HUMAN."""
    path = path or (ROOT / "AI_USAGE.md")
    if not path.exists():
        return "HUMAN", "AI_USAGE.md missing"
    text = path.read_text(encoding="utf-8")
    n = text.count("TODO (human fills)")
    return "HUMAN", f"{n} rows still TODO (human fills)"


def _error_run_count(results: dict) -> int:
    if results.get("invalid"):
        return max(1, int(results.get("error_runs", 1)))
    n = 0
    for d in results.get("per_scenario", {}).values():
        for run in d.get("runs", []):
            if run.get("error"):
                n += 1
    return n


def _latest_real_results(runs_dir: Path) -> tuple[Path, dict] | None:
    if not runs_dir.exists():
        return None
    candidates: list[Path] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or child.name in ("_mock", "_invalid"):
            continue
        r = child / "results.json"
        if r.exists():
            candidates.append(r)
    for r in reversed(candidates):
        try:
            data = json.loads(r.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("llm_mode") == "real":
            return r, data
    return None


CHECKS: list[tuple[str, Callable[..., tuple[str, str]]]] = [
    ("a pytest green", check_pytest_green),
    ("b patch.py diff empty", check_patch_diff_empty),
    ("c mutation_check hash", check_mutation_check_hash),
    ("d mutation_report", check_mutation_report),
    ("e secrets", check_secrets),
    ("f gitignore", check_gitignore),
    ("g runs llm_mode", check_runs_llm_mode),
    ("h frozen hashes", check_frozen_hashes),
    ("i no mock numbers", check_no_mock_numbers),
    ("j checklist", check_checklist),
    ("k decisions/AI_USAGE", check_decisions_ai_usage),
    ("l README commands", check_readme_commands),
    ("m real baseline", check_real_baseline),
    ("n v1 fails >=2 train", check_v1_fails_train),
    ("o changelog+v2", check_changelog_and_v2),
    ("p design-note", check_design_note),
    ("q AI_USAGE TODOs", check_ai_usage_todo),
]


def run_preflight(*, pre_real: bool = False) -> int:
    rows: list[tuple[str, str, str]] = []
    real_run_ids = {"m real baseline", "n v1 fails >=2 train", "o changelog+v2", "p design-note"}
    for name, fn in CHECKS:
        result, detail = fn()
        if pre_real and name in real_run_ids and result == "FAIL":
            result = "PENDING"
            detail = f"(pre-real) {detail}"
        if pre_real and name in real_run_ids and result == "PENDING":
            pass
        # also treat missing real-run artifacts as PENDING under --pre-real (already)
        rows.append((name, result, detail))

    print("| check | result | detail |")
    print("|---|---|---|")
    for name, result, detail in rows:
        detail_esc = detail.replace("|", "\\|")
        print(f"| {name} | {result} | {detail_esc} |")

    fails = [r for r in rows if r[1] == "FAIL"]
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-real",
        action="store_true",
        help="Treat real-run checks as PENDING instead of FAIL",
    )
    args = parser.parse_args(argv)
    return run_preflight(pre_real=args.pre_real)


if __name__ == "__main__":
    raise SystemExit(main())
