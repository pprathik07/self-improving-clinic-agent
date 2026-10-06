"""Tests for scripts/real_run.py pure gate functions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "real_run.py"
    spec = importlib.util.spec_from_file_location("real_run", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load()


def _base(**kwargs):
    data = {
        "llm_mode": "real",
        "k": 3,
        "agent_model": "a",
        "judge_model": "b",
        "sim_model": "c",
        "per_scenario": {
            "t1": {"split": "train", "pass_rate": 0.0, "runs": [{"error": None}]},
            "t2": {"split": "train", "pass_rate": 0.3, "runs": [{"error": None}]},
            "h1": {"split": "heldout", "pass_rate": 1.0, "runs": [{"error": None}]},
        },
    }
    data.update(kwargs)
    return data


def test_valid_baseline():
    ok, msg = rr.baseline_quality_ok(_base())
    assert ok is True


def test_error_runs():
    d = _base()
    d["per_scenario"]["t1"]["runs"] = [{"error": "boom"}]
    ok, msg = rr.baseline_quality_ok(d)
    assert ok is False
    assert "error" in msg


def test_mock_mode():
    ok, msg = rr.baseline_quality_ok(_base(llm_mode="mock"))
    assert ok is False


def test_k1():
    ok, msg = rr.baseline_quality_ok(_base(k=1))
    assert ok is False
    assert "k" in msg


def test_too_easy():
    d = _base()
    d["per_scenario"] = {
        "t1": {"split": "train", "pass_rate": 1.0, "runs": [{"error": None}]},
        "t2": {"split": "train", "pass_rate": 1.0, "runs": [{"error": None}]},
    }
    ok, msg = rr.baseline_quality_ok(d)
    assert ok is False
    assert "scenarios_candidates" in msg


def test_require_models():
    with pytest.raises(RuntimeError):
        rr.require_mock_off_and_distinct_models({"MOCK_LLM": "1", "AGENT_MODEL": "a", "JUDGE_MODEL": "b", "SIM_MODEL": "c"})
    with pytest.raises(RuntimeError):
        rr.require_mock_off_and_distinct_models({"MOCK_LLM": "0", "AGENT_MODEL": "a", "JUDGE_MODEL": "a", "SIM_MODEL": "c"})
    rr.require_mock_off_and_distinct_models({"MOCK_LLM": "0", "AGENT_MODEL": "a", "JUDGE_MODEL": "b", "SIM_MODEL": "c"})
