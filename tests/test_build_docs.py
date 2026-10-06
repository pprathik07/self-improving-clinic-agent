"""Tests for scripts/build_docs.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_docs.py"
    spec = importlib.util.spec_from_file_location("build_docs", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bd = _load()


def test_pending_when_no_paths(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# T\n\n{bd.START}\nold\n{bd.END}\n",
        encoding="utf-8",
    )
    bd.update_readme(readme, None, None)
    text = readme.read_text(encoding="utf-8")
    assert "PENDING REAL RUN" in text
    assert bd.START in text and bd.END in text


def test_refuses_mock(tmp_path):
    before = tmp_path / "before.json"
    before.write_text(json.dumps({
        "llm_mode": "mock",
        "k": 3,
        "overall": {"pass_rate": 1.0, "passed": 1, "total": 1},
        "train": {"pass_rate": 1.0, "passed": 1, "total": 1},
        "heldout": {"pass_rate": 1.0, "passed": 0, "total": 0},
        "per_scenario": {},
    }), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"{bd.START}\nx\n{bd.END}\n", encoding="utf-8")
    with pytest.raises(Exception):
        bd.update_readme(readme, before, None)


def test_replace_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"pre\n{bd.START}\nold\n{bd.END}\npost\n", encoding="utf-8")
    new = bd.replace_results_block(readme.read_text(encoding="utf-8"), "NEW BODY")
    assert "NEW BODY" in new
    assert "old" not in new
    assert new.startswith("pre")
