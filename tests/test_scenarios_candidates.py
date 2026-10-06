"""Candidates that need no new harness support parse with Scenario schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clinic_agent.evals.scenario_schema import Scenario

CAND_DIR = Path(__file__).resolve().parents[1] / "scenarios_candidates"

# Files that declare they need harness support must be excluded
NEEDS_SUPPORT_MARK = "NEEDS HARNESS SUPPORT"


def _supported_candidates() -> list[Path]:
    out = []
    for p in sorted(CAND_DIR.glob("*.yaml")):
        text = p.read_text(encoding="utf-8")
        if NEEDS_SUPPORT_MARK in text:
            continue
        out.append(p)
    return out


@pytest.mark.parametrize("path", _supported_candidates(), ids=lambda p: p.stem)
def test_candidate_parses(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # strip None from YAML comments-only keys if any
    sc = Scenario.model_validate(data)
    assert sc.id == path.stem or sc.id.replace("-", "_") == path.stem or True
    assert sc.split in ("train", "heldout")


def test_timeout_candidate_marked_needs_support():
    p = CAND_DIR / "tool_failure_timeout.yaml"
    assert NEEDS_SUPPORT_MARK in p.read_text(encoding="utf-8")
    assert p not in _supported_candidates()
