"""Gate — decide whether a policy patch should be accepted or rolled back.

Pure function over parsed results.json dicts. Cost/turns +20% is advisory only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """Result of the gate check."""
    accept: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class GateError(Exception):
    """Raised when inputs are invalid for gating (e.g. mock without allow_mock)."""
    pass


def decide(
    before: dict[str, Any],
    after: dict[str, Any],
    targets: list[str],
    *,
    allow_mock: bool = False,
) -> Decision:
    """Accept only if all hard conditions hold.

    Hard rules:
    1. Every target scenario's pass rate strictly improves.
    2. No previously fully-passing scenario drops (train or heldout; 1-run drop counts).
    3. Heldout aggregate does not drop.
    4. No error/invalid runs on either side.
    5. Both sides are llm_mode == "real" (unless allow_mock=True for tests).

    Advisory (never rejects):
    - Cost/turns increase above +20% is reported in reasons only.
    """
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    # 5. llm_mode must be real (unless allow_mock)
    before_mode = before.get("llm_mode", "")
    after_mode = after.get("llm_mode", "")
    if not allow_mock:
        if before_mode != "real" or after_mode != "real":
            raise GateError(
                f"Gate refuses non-real results "
                f"(before.llm_mode={before_mode!r}, after.llm_mode={after_mode!r}). "
                f"Pass allow_mock=True only in tests."
            )
    else:
        if before_mode != "real" or after_mode != "real":
            reasons.append(
                f"ADVISORY: allow_mock=True; modes before={before_mode}, after={after_mode}"
            )

    # Empty targets reject
    if not targets:
        reasons.append("REJECT: no target scenarios")
        return Decision(accept=False, reasons=reasons, metrics=metrics)

    # k mismatch
    before_has_k = "k" in before
    after_has_k = "k" in after
    if before_has_k and after_has_k:
        if before.get("k") != after.get("k"):
            raise GateError(
                f"k mismatch: before.k={before.get('k')!r} after.k={after.get('k')!r}"
            )
    elif before_has_k or after_has_k:
        raise GateError(
            f"k present on only one side "
            f"(before_has_k={before_has_k}, after_has_k={after_has_k})"
        )

    # Scenario-set mismatch
    before_ids = set(before.get("per_scenario", {}))
    after_ids = set(after.get("per_scenario", {}))
    if before_ids != after_ids:
        only_before = sorted(before_ids - after_ids)
        only_after = sorted(after_ids - before_ids)
        raise GateError(
            f"Scenario set mismatch: only_before={only_before}, only_after={only_after}"
        )

    # 4. No error / invalid runs on either side
    before_errors = _error_run_count(before)
    after_errors = _error_run_count(after)
    metrics["before_error_runs"] = before_errors
    metrics["after_error_runs"] = after_errors
    if before_errors > 0 or after_errors > 0:
        reasons.append(
            f"REJECT: error/invalid runs present "
            f"(before={before_errors}, after={after_errors})"
        )
        return Decision(accept=False, reasons=reasons, metrics=metrics)

    accept = True

    # 1. Every target strictly improves
    target_metrics = {}
    for scenario_id in targets:
        old_rate = _get_pass_rate(before, scenario_id)
        new_rate = _get_pass_rate(after, scenario_id)
        target_metrics[scenario_id] = {"before": old_rate, "after": new_rate}
        if new_rate <= old_rate:
            reasons.append(
                f"REJECT: target '{scenario_id}' did not strictly improve: "
                f"{old_rate:.0%} -> {new_rate:.0%}"
            )
            accept = False
        else:
            reasons.append(
                f"Target '{scenario_id}' improved: {old_rate:.0%} -> {new_rate:.0%}"
            )
    metrics["targets"] = target_metrics

    # 2. No previously fully-passing scenario drops (even 1 run)
    regressions = []
    for scenario_id, data in before.get("per_scenario", {}).items():
        old_rate = data.get("pass_rate", 0.0)
        old_pass = data.get("pass_count", 0)
        old_total = data.get("total_runs", 0)
        # Fully-passing: pass_rate == 1.0 (or pass_count == total_runs > 0)
        if old_total <= 0 or old_rate < 1.0:
            continue
        new_data = after.get("per_scenario", {}).get(scenario_id, {})
        new_rate = new_data.get("pass_rate", 0.0)
        new_pass = new_data.get("pass_count", 0)
        if new_rate < old_rate or new_pass < old_pass:
            msg = (
                f"REJECT: previously fully-passing '{scenario_id}' dropped: "
                f"{old_pass}/{old_total} -> {new_pass}/{new_data.get('total_runs', 0)}"
            )
            reasons.append(msg)
            regressions.append(scenario_id)
            accept = False
    metrics["regressions"] = regressions

    # 3. Heldout aggregate does not drop (equal OK)
    old_heldout = before.get("heldout", {}).get("pass_rate", 0.0)
    new_heldout = after.get("heldout", {}).get("pass_rate", 0.0)
    metrics["heldout"] = {"before": old_heldout, "after": new_heldout}
    if new_heldout < old_heldout:
        reasons.append(
            f"REJECT: heldout aggregate dropped: {old_heldout:.0%} -> {new_heldout:.0%}"
        )
        accept = False
    else:
        reasons.append(f"Heldout aggregate: {old_heldout:.0%} -> {new_heldout:.0%}")

    # Advisory: cost/turns +20% — never rejects
    old_turns = _total_turns(before)
    new_turns = _total_turns(after)
    metrics["turns"] = {"before": old_turns, "after": new_turns}
    if old_turns > 0:
        turn_increase = (new_turns - old_turns) / old_turns
        metrics["turn_increase"] = turn_increase
        if turn_increase > 0.20:
            reasons.append(
                f"ADVISORY: turn count increased by {turn_increase:.0%} "
                f"({old_turns} -> {new_turns}), exceeds +20% (does not reject)"
            )
        else:
            reasons.append(f"Turn count: {old_turns} -> {new_turns} ({turn_increase:+.0%})")

    return Decision(accept=accept, reasons=reasons, metrics=metrics)


def _get_pass_rate(results: dict[str, Any], scenario_id: str) -> float:
    per_scenario = results.get("per_scenario", {})
    scenario_data = per_scenario.get(scenario_id, {})
    return float(scenario_data.get("pass_rate", 0.0))


def _total_turns(results: dict[str, Any]) -> int:
    total = 0
    for scenario_data in results.get("per_scenario", {}).values():
        for run in scenario_data.get("runs", []):
            total += int(run.get("turns", 0) or 0)
    return total


def _error_run_count(results: dict[str, Any]) -> int:
    """Count runs with error set, plus top-level invalid markers."""
    if results.get("invalid"):
        return max(1, int(results.get("error_runs", 1)))
    count = 0
    for scenario_data in results.get("per_scenario", {}).values():
        for run in scenario_data.get("runs", []):
            if run.get("error"):
                count += 1
    return count
