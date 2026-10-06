"""End-to-end tests for clinic_agent.loop.improve with fake LLM / fake eval."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from clinic_agent.loop.improve import ImproveError, find_latest_baseline, run_improvement_loop
from clinic_agent.loop.patch import FailureCategory


def _write_baseline(
    runs_dir: Path,
    *,
    llm_mode: str = "mock",
    target_rate: float = 0.0,
    ok_rate: float = 1.0,
    heldout_rate: float = 1.0,
    stamp: str = "20260101_120000",
) -> Path:
    if llm_mode == "mock":
        run_dir = runs_dir / "_mock" / stamp
    else:
        run_dir = runs_dir / stamp
    traces = run_dir / "traces"
    traces.mkdir(parents=True)

    def scenario(sid: str, split: str, rate: float) -> dict:
        total = 3
        passed = round(rate * total)
        runs = []
        lines = []
        for i in range(total):
            ok = i < passed
            runs.append({
                "scenario_id": sid,
                "run_index": i,
                "passed": ok,
                "error": None,
                "turns": 4,
                "scores": {
                    "state": {"appointments_created": ok},
                    "trace": {"must_call:book_appointment": ok},
                    "judge": {},
                    "failures": [] if ok else ["[FAIL] state:appointments_created: got 0"],
                },
            })
            lines.append(f'{{"event":"patient_turn","response":"{sid} run {i}"}}')
            lines.append(f'{{"event":"agent_turn","response":"ack {i}"}}')
            lines.append('{"event":"conversation_end"}')
        (traces / f"{sid}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "split": split,
            "pass_count": passed,
            "total_runs": total,
            "pass_rate": passed / total,
            "runs": runs,
        }

    per = {
        "wrong_dob_then_retry": scenario("wrong_dob_then_retry", "train", target_rate),
        "happy_path_book": scenario("happy_path_book", "train", ok_rate),
        "are_you_a_robot": scenario("are_you_a_robot", "heldout", heldout_rate),
    }
    results = {
        "timestamp": stamp,
        "policy_hash": "baselinehash",
        "k": 3,
        "llm_mode": llm_mode,
        "agent_model": "a",
        "judge_model": "b",
        "sim_model": "c",
        "overall": {"pass_rate": 0.5, "passed": 1, "total": 2},
        "train": {"pass_rate": 0.5, "passed": 1, "total": 2},
        "heldout": {"pass_rate": heldout_rate, "passed": 1, "total": 1},
        "per_scenario": per,
    }
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return run_dir


def _good_patch_json(category: str = "tone") -> str:
    return json.dumps({
        "failure_category": category,
        "evidence": [{"scenario_id": "wrong_dob_then_retry", "turn": 2, "observation": "looped"}],
        "root_cause": "Agent did not confirm slot before booking after DOB retry.",
        "changes": [{
            "policy_section": "tone",
            "op": "add",
            "text": "After a corrected date of birth verification, restate the booking goal and ask one clear next question.",
        }],
        "expected_to_fix": ["wrong_dob_then_retry"],
        "regression_risk": "Might add an extra turn on happy paths.",
    })


def _needs_code_fix_json() -> str:
    return json.dumps({
        "failure_category": "needs_code_fix",
        "evidence": [{"scenario_id": "wrong_dob_then_retry", "turn": 1, "observation": "tool bug"}],
        "root_cause": "Guard rejects valid confirmation hash.",
        "changes": [],
        "expected_to_fix": ["wrong_dob_then_retry"],
        "regression_risk": "None — code fix required.",
    })


def _fake_eval_accept(policy_path: str, k: int = 3, **kwargs) -> dict:
    """After patch: target improves, no regressions."""
    return {
        "timestamp": "20260101_130000",
        "policy_hash": "afterhash",
        "k": k,
        "llm_mode": "mock",
        "agent_model": "a",
        "judge_model": "b",
        "sim_model": "c",
        "overall": {"pass_rate": 1.0, "passed": 3, "total": 3},
        "train": {"pass_rate": 1.0, "passed": 2, "total": 2},
        "heldout": {"pass_rate": 1.0, "passed": 1, "total": 1},
        "per_scenario": {
            "wrong_dob_then_retry": {
                "split": "train", "pass_count": 3, "total_runs": 3, "pass_rate": 1.0,
                "runs": [{"turns": 4, "passed": True, "error": None}] * 3,
            },
            "happy_path_book": {
                "split": "train", "pass_count": 3, "total_runs": 3, "pass_rate": 1.0,
                "runs": [{"turns": 4, "passed": True, "error": None}] * 3,
            },
            "are_you_a_robot": {
                "split": "heldout", "pass_count": 3, "total_runs": 3, "pass_rate": 1.0,
                "runs": [{"turns": 4, "passed": True, "error": None}] * 3,
            },
        },
    }


def _fake_eval_reject(policy_path: str, k: int = 3, **kwargs) -> dict:
    """Target improves but previously fully-passing train drops."""
    out = _fake_eval_accept(policy_path, k=k)
    out["per_scenario"]["happy_path_book"] = {
        "split": "train", "pass_count": 2, "total_runs": 3, "pass_rate": 2 / 3,
        "runs": [
            {"turns": 4, "passed": True, "error": None},
            {"turns": 4, "passed": True, "error": None},
            {"turns": 4, "passed": False, "error": None},
        ],
    }
    out["train"] = {"pass_rate": 0.8, "passed": 5, "total": 6}
    return out


@pytest.fixture
def policy_tree(tmp_path):
    """Copy real policy_v1 into tmp policy dir."""
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    src = Path("policy/policy_v1.yaml")
    shutil.copy(src, policy_dir / "policy_v1.yaml")
    return policy_dir


class TestImprovePaths:
    def test_accept_path(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock", target_rate=0.0)
        answers = iter(["y", "y"])
        summary = run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=_fake_eval_accept,
            stdin_fn=lambda prompt: next(answers),
            changelog_path=tmp_path / "loop" / "CHANGELOG.md",
        )
        assert summary["status"] == "accepted"
        assert (policy_tree / "policy_v2.yaml").exists()
        assert (policy_tree / "policy_v1.yaml").exists()
        changelog = (tmp_path / "loop" / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "ACCEPTED" in changelog
        assert "baselinehash" in changelog
        assert "afterhash" in changelog

    def test_rollback_path(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock", target_rate=0.0)
        answers = iter(["y", "y"])
        summary = run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=_fake_eval_reject,
            stdin_fn=lambda prompt: next(answers),
            changelog_path=tmp_path / "loop" / "CHANGELOG.md",
        )
        assert "rejected" in summary["status"]
        assert summary["active_policy"].endswith("policy_v1.yaml")
        assert not (policy_tree / "policy_v2.yaml").exists()
        assert list(policy_tree.glob("policy_v2.yaml.rejected"))
        # v1 untouched
        assert (policy_tree / "policy_v1.yaml").read_bytes() == Path("policy/policy_v1.yaml").read_bytes()

    def test_mock_baseline_refused(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        with pytest.raises(ImproveError, match="mock"):
            run_improvement_loop(
                policy_path=str(policy_tree / "policy_v1.yaml"),
                runs_dir=runs,
                allow_mock=False,
                reflector_llm=lambda s, m: _good_patch_json(),
                eval_fn=_fake_eval_accept,
                stdin_fn=lambda p: "y",
            )

    def test_n_at_reflector_prompt(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        summary = run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: (_ for _ in ()).throw(AssertionError("should not call")),
            eval_fn=_fake_eval_accept,
            stdin_fn=lambda p: "n",
            changelog_path=tmp_path / "CHANGELOG.md",
        )
        assert summary["status"] == "stopped_reflector"
        assert not (policy_tree / "policy_v2.yaml").exists()

    def test_n_at_apply_prompt(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        answers = iter(["y", "n"])
        summary = run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=_fake_eval_accept,
            stdin_fn=lambda p: next(answers),
            changelog_path=tmp_path / "CHANGELOG.md",
        )
        assert summary["status"] == "stopped_apply"
        assert not (policy_tree / "policy_v2.yaml").exists()

    def test_needs_code_fix_path(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        summary = run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _needs_code_fix_json(),
            eval_fn=_fake_eval_accept,
            stdin_fn=lambda p: "y",
            changelog_path=tmp_path / "CHANGELOG.md",
        )
        assert summary["status"] == "needs_code_fix"
        assert not (policy_tree / "policy_v2.yaml").exists()
        assert summary["patch"].failure_category == FailureCategory.NEEDS_CODE_FIX

    def test_v1_never_overwritten(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        v1_before = (policy_tree / "policy_v1.yaml").read_bytes()
        answers = iter(["y", "y"])
        run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=_fake_eval_accept,
            stdin_fn=lambda p: next(answers),
            changelog_path=tmp_path / "CHANGELOG.md",
        )
        assert (policy_tree / "policy_v1.yaml").read_bytes() == v1_before
        assert (policy_tree / "policy_v2.yaml").exists()

    def test_changelog_produced_by_code(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        answers = iter(["y", "y"])
        cl = tmp_path / "loop" / "CHANGELOG.md"
        run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=_fake_eval_accept,
            stdin_fn=lambda p: next(answers),
            changelog_path=cl,
        )
        text = cl.read_text(encoding="utf-8")
        assert "patch_json" in text
        assert "continue_to_reflector: y" in text
        assert "apply_patch: y" in text
        assert "Gate:" in text or "ACCEPT" in text

    def test_gate_rejection_leaves_v1_active(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        answers = iter(["y", "y"])
        summary = run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=_fake_eval_reject,
            stdin_fn=lambda p: next(answers),
            changelog_path=tmp_path / "CHANGELOG.md",
        )
        assert summary["active_policy"].endswith("policy_v1.yaml")

    def test_yes_refused_for_real_baseline(self, tmp_path, policy_tree):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="real", stamp="20260102_120000")
        with pytest.raises(ImproveError, match="CI-only"):
            run_improvement_loop(
                policy_path=str(policy_tree / "policy_v1.yaml"),
                runs_dir=runs,
                yes=True,
                allow_mock=False,
                reflector_llm=lambda s, m: _good_patch_json(),
                eval_fn=_fake_eval_accept,
                stdin_fn=lambda p: "y",
            )

    def test_find_latest_refuses_mock(self, tmp_path):
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        with pytest.raises(ImproveError):
            find_latest_baseline(runs, allow_mock=False)

    def test_find_latest_refuses_mock_mode_outside_mock_dir(self, tmp_path):
        """llm_mode=mock in a normal timestamp dir (not runs/_mock/) must be refused.

        Covers the mode check that I4 mutates; _mock/ directory skip alone is not enough.
        """
        runs = tmp_path / "runs"
        # Write under a real-looking stamp path, but mark llm_mode mock
        run_dir = _write_baseline(runs, llm_mode="real", stamp="20260103_120000")
        results_path = run_dir / "results.json"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        data["llm_mode"] = "mock"
        results_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ImproveError, match="llm_mode"):
            find_latest_baseline(runs, allow_mock=False)

    def test_rerun_uses_baseline_k(self, tmp_path, policy_tree):
        """Re-run eval must receive the baseline's results['k']."""
        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")  # baseline k=3
        # Overwrite k to a distinctive value
        run_dir = sorted((runs / "_mock").iterdir())[-1]
        results_path = run_dir / "results.json"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        data["k"] = 7
        results_path.write_text(json.dumps(data), encoding="utf-8")

        seen = {}

        def capturing_eval(policy_path: str, k: int = 3, **kwargs):
            seen["k"] = k
            return _fake_eval_accept(policy_path, k=k)

        answers = iter(["y", "y"])
        run_improvement_loop(
            policy_path=str(policy_tree / "policy_v1.yaml"),
            runs_dir=runs,
            allow_mock=True,
            reflector_llm=lambda s, m: _good_patch_json(),
            eval_fn=capturing_eval,
            stdin_fn=lambda p: next(answers),
            changelog_path=tmp_path / "CHANGELOG.md",
        )
        assert seen["k"] == 7

    def test_find_latest_refuses_error_runs(self, tmp_path):
        """Any error run in baseline → ImproveError (even with allow_mock)."""
        runs = tmp_path / "runs"
        run_dir = _write_baseline(runs, llm_mode="mock")
        results_path = run_dir / "results.json"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        data["per_scenario"]["happy_path_book"]["runs"][0]["error"] = "infra boom"
        results_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ImproveError, match="error run"):
            find_latest_baseline(runs, allow_mock=True)

    def test_missing_baseline_k_raises(self, tmp_path, policy_tree):
        """Baseline without usable k → ImproveError (no silent fallback)."""
        runs = tmp_path / "runs"
        run_dir = _write_baseline(runs, llm_mode="mock")
        results_path = run_dir / "results.json"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        del data["k"]
        results_path.write_text(json.dumps(data), encoding="utf-8")

        answers = iter(["y", "y"])
        with pytest.raises(ImproveError, match="no usable 'k'"):
            run_improvement_loop(
                policy_path=str(policy_tree / "policy_v1.yaml"),
                runs_dir=runs,
                allow_mock=True,
                reflector_llm=lambda s, m: _good_patch_json(),
                eval_fn=_fake_eval_accept,
                stdin_fn=lambda p: next(answers),
                changelog_path=tmp_path / "CHANGELOG.md",
            )

    def test_refuse_reeval_same_policy_path(self, tmp_path, policy_tree, monkeypatch):
        """Never re-eval the baseline policy path / policy_v1 as the patch."""
        from clinic_agent.loop import improve as improve_mod

        runs = tmp_path / "runs"
        _write_baseline(runs, llm_mode="mock")
        v1 = str(policy_tree / "policy_v1.yaml")

        def fake_write(policy, version, policy_dir="policy"):
            # Point "new" policy at the same path as baseline → must refuse
            return Path(v1)

        monkeypatch.setattr(improve_mod, "write_new_policy", fake_write)
        answers = iter(["y", "y"])
        with pytest.raises(ImproveError, match="Refusing to re-eval|policy_v1"):
            run_improvement_loop(
                policy_path=v1,
                runs_dir=runs,
                allow_mock=True,
                reflector_llm=lambda s, m: _good_patch_json(),
                eval_fn=_fake_eval_accept,
                stdin_fn=lambda p: next(answers),
                changelog_path=tmp_path / "CHANGELOG.md",
            )
