"""Tests for scripts/freeze_manifest.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_manifest.py"
    spec = importlib.util.spec_from_file_location("freeze_manifest", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fm = _load()


def test_write_and_check_pass(tmp_path):
    scenarios = tmp_path / "clinic_agent" / "evals" / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "a.yaml").write_text("id: a\n", encoding="utf-8")
    scorers = tmp_path / "clinic_agent" / "evals" / "scorers.py"
    scorers.parent.mkdir(parents=True, exist_ok=True)
    scorers.write_text("# scorers\n", encoding="utf-8")
    policy = tmp_path / "policy" / "policy_v1.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text("identity:\n  text: t\n", encoding="utf-8")

    out = tmp_path / "data" / "frozen_hashes.json"
    fm.write_manifest(out, root=tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "clinic_agent/evals/scenarios/a.yaml" in data
    ok, problems = fm.check_manifest(out, root=tmp_path)
    assert ok is True
    assert problems == []


def test_check_detects_change(tmp_path):
    scenarios = tmp_path / "clinic_agent" / "evals" / "scenarios"
    scenarios.mkdir(parents=True)
    f = scenarios / "a.yaml"
    f.write_text("id: a\n", encoding="utf-8")
    (tmp_path / "clinic_agent" / "evals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "clinic_agent" / "evals" / "scorers.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy" / "policy_v1.yaml").write_text("y\n", encoding="utf-8")
    out = tmp_path / "frozen.json"
    fm.write_manifest(out, root=tmp_path)
    f.write_text("id: a\nchanged\n", encoding="utf-8")
    ok, problems = fm.check_manifest(out, root=tmp_path)
    assert ok is False
    assert any("CHANGED" in p for p in problems)
