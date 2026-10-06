"""Tests for failures.md, report.py, calibrate.py."""

import json
from pathlib import Path

import pytest
import yaml

from clinic_agent.evals.calibrate import (
    CalibrateError,
    compute_agree_rate,
    generate_from_run,
)
from clinic_agent.evals.failures import write_failures_md
from clinic_agent.evals.report import ReportError, render_report


def _mini_results(llm_mode: str = "real", fail: bool = True) -> dict:
    runs = [
        {
            "run_index": 0,
            "passed": not fail,
            "error": None,
            "turns": 3,
            "scores": {
                "state": {"appointments_created": not fail},
                "trace": {"must_call:book_appointment": True},
                "judge": {"ok": True},
                "failures": ["[FAIL] appointments_created: Expected 1, got 0"] if fail else [],
            },
        }
    ]
    return {
        "timestamp": "20260101_000000",
        "policy_hash": "abc",
        "k": 1,
        "llm_mode": llm_mode,
        "agent_model": "a",
        "judge_model": "b",
        "sim_model": "c",
        "overall": {"pass_rate": 0.0 if fail else 1.0, "passed": 0 if fail else 1, "total": 1},
        "train": {"pass_rate": 0.0 if fail else 1.0, "passed": 0 if fail else 1, "total": 1},
        "heldout": {"pass_rate": 1.0, "passed": 0, "total": 0},
        "per_scenario": {
            "train_fail": {
                "split": "train",
                "pass_count": 0 if fail else 1,
                "total_runs": 1,
                "pass_rate": 0.0 if fail else 1.0,
                "runs": runs,
            }
        },
    }


def _rich_run(tmp_path: Path, llm_mode: str = "real") -> Path:
    """Build a run dir with >=6 non-error runs (mix pass/fail) and traces."""
    run_dir = tmp_path / "run"
    traces = run_dir / "traces"
    traces.mkdir(parents=True)
    per = {}
    for i in range(6):
        sid = f"scen_{i:02d}"
        passed = i >= 3  # 3 fail, 3 pass
        per[sid] = {
            "split": "train",
            "pass_count": 1 if passed else 0,
            "total_runs": 1,
            "pass_rate": 1.0 if passed else 0.0,
            "runs": [{
                "run_index": 0,
                "passed": passed,
                "error": None,
                "turns": 2,
                "scores": {"failures": []},
            }],
        }
        (traces / f"{sid}.jsonl").write_text(
            f'{{"event":"patient_turn","turn":1,"response":"hello {sid}"}}\n'
            f'{{"event":"agent_turn","turn":1,"response":"hi {sid}"}}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
    results = {
        "timestamp": "20260101_000000",
        "policy_hash": "abc",
        "k": 1,
        "llm_mode": llm_mode,
        "per_scenario": per,
        "train": {"pass_rate": 0.5, "passed": 3, "total": 6},
        "heldout": {"pass_rate": 1.0, "passed": 0, "total": 0},
        "overall": {"pass_rate": 0.5, "passed": 3, "total": 6},
    }
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return run_dir


class TestFailuresMd:
    def test_writes_transcript_and_layers(self, tmp_path):
        run_dir = tmp_path / "run"
        traces = run_dir / "traces"
        traces.mkdir(parents=True)
        (traces / "train_fail.jsonl").write_text(
            '{"event":"patient_turn","response":"Hi I am the patient"}\n'
            '{"event":"agent_turn","response":"Hello"}\n'
            '{"event":"tool_call","tool":"verify_patient","args":{"name":"X"},"result":{"ok":false}}\n'
            '{"event":"conversation_end"}\n',
            encoding="utf-8",
        )
        results = _mini_results(fail=True)
        (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
        path = write_failures_md(run_dir)
        text = path.read_text(encoding="utf-8")
        assert "train_fail" in text
        assert "Hi I am the patient" in text
        assert "verify_patient" in text
        assert "Caught by:** state" in text
        assert "appointments_created" in text


class TestReport:
    def test_before_after_table(self):
        before = _mini_results(llm_mode="real", fail=True)
        after = _mini_results(llm_mode="real", fail=False)
        md = render_report(before, after, allow_mock=False)
        assert "Per-scenario" in md
        assert "train_fail" in md
        assert "MOCK, DO NOT PUBLISH" not in md

    def test_mock_refused_without_flag(self):
        before = _mini_results(llm_mode="mock", fail=True)
        with pytest.raises(ReportError):
            render_report(before, allow_mock=False)

    def test_mock_stamped_with_flag(self):
        before = _mini_results(llm_mode="mock", fail=True)
        md = render_report(before, allow_mock=True)
        assert "MOCK, DO NOT PUBLISH" in md


class TestCalibrate:
    def test_refuses_mock_without_flag(self, tmp_path):
        run_dir = _rich_run(tmp_path, llm_mode="mock")
        out = tmp_path / "calib"
        with pytest.raises(CalibrateError, match="Refusing"):
            generate_from_run(run_dir, out, allow_mock=False)

    def test_from_run_writes_six_empty_labels(self, tmp_path):
        run_dir = _rich_run(tmp_path, llm_mode="real")
        out = tmp_path / "calib"
        created = generate_from_run(run_dir, out, allow_mock=False)
        transcripts = list(out.glob("transcript_*.txt"))
        labels = list(out.glob("labels_*.yaml"))
        assert len(transcripts) == 6
        assert len(labels) == 6
        assert len(created) == 12
        for p in labels:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            assert data.get("human_labels") == {} or data.get("human_labels") is None
            assert data.get("judge_labels") == {} or data.get("judge_labels") is None
            # No filled verdicts
            text = p.read_text(encoding="utf-8")
            assert "true" not in text.lower().split("human_labels")[-1] if "human_labels" in text else True

    def test_allow_mock_stamps_do_not_label(self, tmp_path):
        run_dir = _rich_run(tmp_path, llm_mode="mock")
        out = tmp_path / "calib"
        generate_from_run(run_dir, out, allow_mock=True)
        text = (out / "transcript_01.txt").read_text(encoding="utf-8")
        assert "MOCK, DO NOT LABEL" in text
        label = (out / "labels_01.yaml").read_text(encoding="utf-8")
        assert "MOCK, DO NOT LABEL" in label

    def test_agree_rate_from_human_labels(self, tmp_path):
        run_dir = _rich_run(tmp_path, llm_mode="real")
        out = tmp_path / "calib"
        generate_from_run(run_dir, out, allow_mock=False)
        label = out / "labels_01.yaml"
        label.write_text(
            "transcript: transcript_01.txt\n"
            "judge_labels:\n  polite: true\n  verified: false\n"
            "human_labels:\n  polite: true\n  verified: true\n",
            encoding="utf-8",
        )
        result = compute_agree_rate(out)
        assert result["comparisons"] == 2
        assert result["agreements"] == 1
        assert result["agree_rate"] == 0.5
        assert result["unlabeled_files"] == 5
