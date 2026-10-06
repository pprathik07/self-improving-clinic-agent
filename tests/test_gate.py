"""Tests for loop/gate.py — decide() accept/reject rules."""

import pytest

from clinic_agent.loop.gate import Decision, GateError, decide


def _results(
    *,
    llm_mode: str = "real",
    train_rate: float = 1.0,
    heldout_rate: float = 1.0,
    per_scenario: dict | None = None,
    turns_per_run: int = 5,
    with_error: bool = False,
) -> dict:
    """Build a minimal results.json-shaped dict."""
    if per_scenario is None:
        per_scenario = {
            "target_a": {
                "split": "train",
                "pass_count": 2 if train_rate < 1 else 3,
                "total_runs": 3,
                "pass_rate": train_rate if "target" in "target_a" else train_rate,
                "runs": [
                    {"turns": turns_per_run, "passed": True, "error": None},
                    {"turns": turns_per_run, "passed": True, "error": None},
                    {"turns": turns_per_run, "passed": train_rate >= 1.0, "error": None},
                ],
            },
            "ok_train": {
                "split": "train",
                "pass_count": 3,
                "total_runs": 3,
                "pass_rate": 1.0,
                "runs": [
                    {"turns": turns_per_run, "passed": True, "error": None},
                    {"turns": turns_per_run, "passed": True, "error": None},
                    {"turns": turns_per_run, "passed": True, "error": None},
                ],
            },
            "ok_heldout": {
                "split": "heldout",
                "pass_count": 3,
                "total_runs": 3,
                "pass_rate": 1.0,
                "runs": [
                    {"turns": turns_per_run, "passed": True, "error": None},
                    {"turns": turns_per_run, "passed": True, "error": None},
                    {"turns": turns_per_run, "passed": True, "error": None},
                ],
            },
        }
    if with_error:
        # Inject an error run into first scenario
        first = next(iter(per_scenario.values()))
        first["error_runs"] = 1
        first["runs"] = list(first.get("runs", [])) + [
            {"turns": 1, "passed": False, "error": "infra boom"}
        ]

    return {
        "llm_mode": llm_mode,
        "train": {"pass_rate": train_rate, "passed": 0, "total": 0},
        "heldout": {"pass_rate": heldout_rate, "passed": 0, "total": 0},
        "overall": {"pass_rate": (train_rate + heldout_rate) / 2, "passed": 0, "total": 0},
        "per_scenario": per_scenario,
    }


def _pair(
    before_target_rate: float = 0.0,
    after_target_rate: float = 1.0,
    *,
    before_heldout: float = 1.0,
    after_heldout: float = 1.0,
    drop_ok_train: bool = False,
    drop_ok_heldout: bool = False,
    before_mode: str = "real",
    after_mode: str = "real",
    turns_before: int = 5,
    turns_after: int = 5,
    before_error: bool = False,
    after_error: bool = False,
) -> tuple[dict, dict]:
    def scenario(rate: float, split: str, turns: int) -> dict:
        total = 3
        passed = round(rate * total)
        return {
            "split": split,
            "pass_count": passed,
            "total_runs": total,
            "pass_rate": passed / total,
            "runs": [
                {"turns": turns, "passed": i < passed, "error": None}
                for i in range(total)
            ],
        }

    before_ps = {
        "target_a": scenario(before_target_rate, "train", turns_before),
        "ok_train": scenario(0.0 if False else 1.0, "train", turns_before),
        "ok_heldout": scenario(1.0, "heldout", turns_before),
    }
    after_ps = {
        "target_a": scenario(after_target_rate, "train", turns_after),
        "ok_train": scenario(2 / 3 if drop_ok_train else 1.0, "train", turns_after),
        "ok_heldout": scenario(2 / 3 if drop_ok_heldout else 1.0, "heldout", turns_after),
    }
    before = _results(
        llm_mode=before_mode,
        train_rate=before_target_rate,
        heldout_rate=before_heldout,
        per_scenario=before_ps,
        turns_per_run=turns_before,
        with_error=before_error,
    )
    after = _results(
        llm_mode=after_mode,
        train_rate=after_target_rate,
        heldout_rate=after_heldout,
        per_scenario=after_ps,
        turns_per_run=turns_after,
        with_error=after_error,
    )
    # Override heldout aggregate explicitly
    before["heldout"] = {"pass_rate": before_heldout, "passed": 0, "total": 0}
    after["heldout"] = {"pass_rate": after_heldout, "passed": 0, "total": 0}
    return before, after


class TestGateAccept:
    def test_accept_when_all_rules_hold(self):
        before, after = _pair(0.0, 1.0)
        d = decide(before, after, ["target_a"], allow_mock=False)
        assert isinstance(d, Decision)
        assert d.accept is True
        assert any("improved" in r.lower() for r in d.reasons)


class TestGateRejectTargetNotImproved:
    def test_reject_target_equal_not_strict(self):
        """Equal-not-strict improvement rejects."""
        before, after = _pair(0.333, 0.333)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("did not strictly improve" in r for r in d.reasons)

    def test_reject_target_lower(self):
        before, after = _pair(0.666, 0.333)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("did not strictly improve" in r for r in d.reasons)


class TestGateRejectRegression:
    def test_reject_one_run_drop_on_fully_passing_train(self):
        """Exactly one run dropped on a fully-passing train scenario rejects."""
        before, after = _pair(0.0, 1.0, drop_ok_train=True)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("ok_train" in r and "dropped" in r for r in d.reasons)

    def test_reject_one_run_drop_on_fully_passing_heldout(self):
        before, after = _pair(0.0, 1.0, drop_ok_heldout=True)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("ok_heldout" in r and "dropped" in r for r in d.reasons)


class TestGateHeldoutAggregate:
    def test_accept_heldout_equal(self):
        before, after = _pair(0.0, 1.0, before_heldout=0.8, after_heldout=0.8)
        d = decide(before, after, ["target_a"])
        assert d.accept is True

    def test_reject_heldout_lower(self):
        before, after = _pair(0.0, 1.0, before_heldout=0.9, after_heldout=0.8)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("heldout aggregate dropped" in r.lower() for r in d.reasons)


class TestGateErrorRuns:
    def test_reject_error_runs_on_before(self):
        before, after = _pair(0.0, 1.0, before_error=True)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("error/invalid" in r.lower() for r in d.reasons)

    def test_reject_error_runs_on_after(self):
        before, after = _pair(0.0, 1.0, after_error=True)
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("error/invalid" in r.lower() for r in d.reasons)


class TestGateMockRefused:
    def test_mock_input_refused_without_allow_mock(self):
        before, after = _pair(0.0, 1.0, before_mode="mock", after_mode="mock")
        with pytest.raises(GateError):
            decide(before, after, ["target_a"], allow_mock=False)

    def test_mock_allowed_in_tests(self):
        before, after = _pair(0.0, 1.0, before_mode="mock", after_mode="mock")
        d = decide(before, after, ["target_a"], allow_mock=True)
        assert d.accept is True


class TestGateCostAdvisory:
    def test_plus_20_percent_cost_still_accepts_with_advisory(self):
        """+20% turns is advisory: still accepts, reason mentions advisory."""
        before, after = _pair(0.0, 1.0, turns_before=5, turns_after=7)  # +40%
        d = decide(before, after, ["target_a"])
        assert d.accept is True
        assert any("ADVISORY" in r and "20%" in r for r in d.reasons)


class TestGateMutationCoverage:
    """Named tests that must fail under the hand-mutation table (G1–G12)."""

    def test_after_only_mock_refused(self):
        """G7: only the after side is mock — must raise without allow_mock."""
        before, after = _pair(0.0, 1.0, before_mode="real", after_mode="mock")
        with pytest.raises(GateError):
            decide(before, after, ["target_a"], allow_mock=False)

    def test_reject_top_level_invalid_marker(self):
        """G8: top-level invalid: true rejects even with no per-run errors."""
        before, after = _pair(0.0, 1.0)
        after = dict(after)
        after["invalid"] = True
        d = decide(before, after, ["target_a"])
        assert d.accept is False
        assert any("error/invalid" in r.lower() for r in d.reasons)

    def test_partial_pass_drop_still_accepted(self):
        """G9: a previously partially-passing scenario may drop without rejecting."""
        before, after = _pair(0.0, 1.0)
        before["per_scenario"]["partial"] = {
            "split": "train",
            "pass_count": 2,
            "total_runs": 3,
            "pass_rate": 2 / 3,
            "runs": [
                {"turns": 5, "passed": True, "error": None},
                {"turns": 5, "passed": True, "error": None},
                {"turns": 5, "passed": False, "error": None},
            ],
        }
        after["per_scenario"]["partial"] = {
            "split": "train",
            "pass_count": 1,
            "total_runs": 3,
            "pass_rate": 1 / 3,
            "runs": [
                {"turns": 5, "passed": True, "error": None},
                {"turns": 5, "passed": False, "error": None},
                {"turns": 5, "passed": False, "error": None},
            ],
        }
        d = decide(before, after, ["target_a"])
        assert d.accept is True
        assert not any("partial" in r and "dropped" in r for r in d.reasons)

    def test_one_run_drop_is_exactly_one(self):
        """G2b: regression fires on a drop of exactly one run (3/3 -> 2/3)."""
        before, after = _pair(0.0, 1.0, drop_ok_train=True)
        before_sc = before["per_scenario"]["ok_train"]
        after_sc = after["per_scenario"]["ok_train"]
        assert before_sc["pass_count"] - after_sc["pass_count"] == 1
        d = decide(before, after, ["target_a"])
        assert d.accept is False

    def test_empty_targets_rejected(self):
        """G10: empty targets → accept=False with REJECT: no target scenarios."""
        before, after = _pair(0.0, 1.0)
        d = decide(before, after, [])
        assert d.accept is False
        assert any("no target scenarios" in r for r in d.reasons)

    def test_k_mismatch_raises(self):
        """G11: differing k raises GateError."""
        before, after = _pair(0.0, 1.0)
        before = dict(before)
        after = dict(after)
        before["k"] = 3
        after["k"] = 5
        with pytest.raises(GateError, match="k mismatch"):
            decide(before, after, ["target_a"])

    def test_scenario_set_mismatch_raises(self):
        """G12: different per_scenario id sets raise GateError."""
        before, after = _pair(0.0, 1.0)
        after = dict(after)
        after["per_scenario"] = dict(after["per_scenario"])
        after["per_scenario"]["extra_only_after"] = {
            "split": "train",
            "pass_count": 3,
            "total_runs": 3,
            "pass_rate": 1.0,
            "runs": [{"turns": 1, "passed": True, "error": None}] * 3,
        }
        with pytest.raises(GateError, match="Scenario set mismatch"):
            decide(before, after, ["target_a"])
