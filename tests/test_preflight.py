"""Tests for scripts/preflight.py pure checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

EXPECTED_MUT = "ebdf0a7b28311ab396fd3a3cd1e4f81e23403c6db97b5e787c3d80fe4e045bce"


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
    spec = importlib.util.spec_from_file_location("preflight", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load()


def test_pytest_green_pass_fail():
    r, _ = pf.check_pytest_green(run_pytest=lambda: (0, "10 passed in 1s"))
    assert r == "PASS"
    r, _ = pf.check_pytest_green(run_pytest=lambda: (1, "1 failed"))
    assert r == "FAIL"
    r, d = pf.check_pytest_green(run_pytest=lambda: (0, "9 passed, 1 skipped"))
    assert r == "FAIL"
    assert "skipped" in d


def test_patch_diff():
    assert pf.check_patch_diff_empty("")[0] == "PASS"
    assert pf.check_patch_diff_empty("diff --git a/x")[0] == "FAIL"


def test_mutation_check_hash(tmp_path):
    p = tmp_path / "mutation_check.py"
    p.write_text("x", encoding="utf-8")
    got = hashlib.sha256(b"x").hexdigest()
    assert pf.check_mutation_check_hash(p, expected=got)[0] == "PASS"
    assert pf.check_mutation_check_hash(p, expected=EXPECTED_MUT)[0] == "FAIL"


def test_mutation_report(tmp_path):
    assert pf.check_mutation_report(tmp_path / "missing.md")[0] == "PENDING"
    good = tmp_path / "r.md"
    good.write_text("| id | result |\n|---|---|\n| R1 | CAUGHT |\n", encoding="utf-8")
    assert pf.check_mutation_report(good)[0] == "PASS"
    bad = tmp_path / "b.md"
    bad.write_text("| id | result |\n| R1 | UNTESTED |\n", encoding="utf-8")
    # our checker looks for "| UNTESTED |"
    bad.write_text("| id | file | change | result | failing tests | restored hash OK |\n| R1 | f | c | UNTESTED | | yes |\n", encoding="utf-8")
    assert pf.check_mutation_report(bad)[0] == "PENDING"


def test_secrets(tmp_path):
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=your_key_here\n", encoding="utf-8")
    r, _ = pf.check_secrets(ls_files=[".env.example", "README.md"], root=tmp_path)
    assert r == "PASS"
    r, _ = pf.check_secrets(ls_files=[".env", ".env.example"], root=tmp_path)
    assert r == "FAIL"


def test_gitignore():
    good = ".env\nclinic.db\nruns/_mock/\nruns/_invalid/\n"
    assert pf.check_gitignore(good)[0] == "PASS"
    assert pf.check_gitignore(good + "uv.lock\n")[0] == "FAIL"


def test_runs_llm_mode(tmp_path):
    runs = tmp_path / "runs"
    (runs / "20260101").mkdir(parents=True)
    (runs / "20260101" / "results.json").write_text(
        json.dumps({"llm_mode": "real"}), encoding="utf-8"
    )
    assert pf.check_runs_llm_mode(runs)[0] == "PASS"
    (runs / "bad").mkdir()
    (runs / "bad" / "results.json").write_text(
        json.dumps({"llm_mode": "mock"}), encoding="utf-8"
    )
    assert pf.check_runs_llm_mode(runs)[0] == "FAIL"


def test_no_mock_numbers(tmp_path):
    (tmp_path / "README.md").write_text("ok\n", encoding="utf-8")
    (tmp_path / "design-note.md").write_text("ok\n", encoding="utf-8")
    (tmp_path / "decisions.md").write_text("ok\n", encoding="utf-8")
    (tmp_path / "loop").mkdir()
    (tmp_path / "loop" / "CHANGELOG.md").write_text("ok\n", encoding="utf-8")
    assert pf.check_no_mock_numbers(tmp_path)[0] == "PASS"
    (tmp_path / "README.md").write_text("score 14/15\n", encoding="utf-8")
    assert pf.check_no_mock_numbers(tmp_path)[0] == "FAIL"


def test_checklist(tmp_path):
    p = tmp_path / "CHECKLIST.md"
    p.write_text("- [ ] item\n", encoding="utf-8")
    assert pf.check_checklist(p)[0] == "PASS"
    p.write_text("- [DONE ] x\n", encoding="utf-8")
    assert pf.check_checklist(p)[0] == "FAIL"


def test_decisions_dupes(tmp_path):
    d = tmp_path / "decisions.md"
    d.write_text(
        "2026-10-06 | Title A | o | w | c\n"
        "2026-10-06 | Title A | o | w | c\n",
        encoding="utf-8",
    )
    a = tmp_path / "AI_USAGE.md"
    a.write_text("| x | did | TODO (human fills) |\n", encoding="utf-8")
    assert pf.check_decisions_ai_usage(d, a)[0] == "FAIL"
    d.write_text("2026-10-06 | Title A | o | w | c\n", encoding="utf-8")
    assert pf.check_decisions_ai_usage(d, a)[0] == "PASS"
    a.write_text("| llm | sleeps 2/4/8 | TODO (human fills) |\n", encoding="utf-8")
    assert pf.check_decisions_ai_usage(d, a)[0] == "FAIL"


def test_readme_commands():
    top = """
## Quick Start
uv run python -m clinic_agent
uv run python -m clinic_agent.evals
uv run python -m clinic_agent.loop.improve
uv run pytest -q
"""
    assert pf.check_readme_commands(top)[0] == "PASS"
    assert pf.check_readme_commands("hello")[0] == "FAIL"


def test_baseline_gates(tmp_path):
    runs = tmp_path / "runs"
    assert pf.check_real_baseline(runs)[0] == "PENDING"
    d = runs / "20260101_120000"
    d.mkdir(parents=True)
    results = {
        "llm_mode": "real",
        "k": 3,
        "agent_model": "a",
        "judge_model": "b",
        "sim_model": "c",
        "per_scenario": {
            "t1": {"split": "train", "pass_rate": 0.0, "runs": [{"error": None}]},
            "t2": {"split": "train", "pass_rate": 0.0, "runs": [{"error": None}]},
        },
    }
    (d / "results.json").write_text(json.dumps(results), encoding="utf-8")
    assert pf.check_real_baseline(runs)[0] == "PASS"
    assert pf.check_v1_fails_train(runs)[0] == "PASS"
    results["per_scenario"] = {
        "t1": {"split": "train", "pass_rate": 1.0, "runs": [{"error": None}]},
    }
    (d / "results.json").write_text(json.dumps(results), encoding="utf-8")
    assert pf.check_v1_fails_train(runs)[0] == "FAIL"


def test_changelog_v2(tmp_path):
    cl = tmp_path / "CHANGELOG.md"
    pol = tmp_path / "policy"
    pol.mkdir()
    assert pf.check_changelog_and_v2(cl, pol)[0] == "PENDING"
    cl.write_text("## 2026-10-06 ACCEPTED\n", encoding="utf-8")
    (pol / "policy_v2.yaml").write_text("x\n", encoding="utf-8")
    assert pf.check_changelog_and_v2(cl, pol)[0] == "PASS"


def test_design_note(tmp_path):
    p = tmp_path / "design-note.md"
    p.write_text("HUMAN: write this\n" + ("word " * 10), encoding="utf-8")
    assert pf.check_design_note(p)[0] == "PENDING"
    p.write_text("word " * 50, encoding="utf-8")
    assert pf.check_design_note(p)[0] == "PASS"


def test_ai_usage_todo(tmp_path):
    p = tmp_path / "AI_USAGE.md"
    p.write_text("| a | x | TODO (human fills) |\n| b | y | TODO (human fills) |\n", encoding="utf-8")
    r, d = pf.check_ai_usage_todo(p)
    assert r == "HUMAN"
    assert "2 rows" in d
