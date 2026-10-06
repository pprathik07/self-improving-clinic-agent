"""Scorers — three-layer evaluation scoring.

Layer 1: State checks (ground truth) — DB state and session state.
Layer 2: Trace checks — tool call sequence patterns.
Layer 3: LLM judge — binary rubric items on the transcript.

A scenario passes only if all three layers pass.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from clinic_agent.agent.state import SessionState
from clinic_agent.evals.scenario_schema import Scenario


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class CheckResult:
    """Result of a single scoring check."""

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


class ScoreReport:
    """Aggregated scoring report for one scenario run."""

    def __init__(self):
        self.state_checks: list[CheckResult] = []
        self.trace_checks: list[CheckResult] = []
        self.judge_checks: list[CheckResult] = []

    @property
    def all_checks(self) -> list[CheckResult]:
        return self.state_checks + self.trace_checks + self.judge_checks

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.all_checks)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "state": {c.name: c.passed for c in self.state_checks},
            "trace": {c.name: c.passed for c in self.trace_checks},
            "judge": {c.name: c.passed for c in self.judge_checks},
            "failures": [repr(c) for c in self.all_checks if not c.passed],
        }


# ---------------------------------------------------------------------------
# Layer 1: State checks (ground truth)
# ---------------------------------------------------------------------------

def score_state(
    scenario: Scenario,
    session: SessionState,
    conn: sqlite3.Connection,
    trace_entries: list[dict[str, Any]],
) -> list[CheckResult]:
    """Run all state-level checks against the final DB and session."""
    results = []
    expect = scenario.expect.state

    # Check appointments created
    if expect.appointments_created is not None:
        actual = conn.execute(
            "SELECT COUNT(*) as c FROM appointments WHERE status = 'booked'"
        ).fetchone()["c"]
        # Subtract pre-existing appointments (James has 1 by default)
        pre_existing = _count_pre_existing_appointments(conn, scenario.seed_db)
        created = actual - pre_existing
        results.append(CheckResult(
            "appointments_created",
            created == expect.appointments_created,
            f"Expected {expect.appointments_created}, got {created}",
        ))

    # Check appointments cancelled
    if expect.appointments_cancelled is not None:
        cancelled = conn.execute(
            "SELECT COUNT(*) as c FROM appointments WHERE status = 'cancelled'"
        ).fetchone()["c"]
        results.append(CheckResult(
            "appointments_cancelled",
            cancelled >= expect.appointments_cancelled,
            f"Expected >= {expect.appointments_cancelled}, got {cancelled}",
        ))

    # Check no write before verified
    if expect.no_write_before_verified:
        write_tools = {"book_appointment", "cancel_appointment", "reschedule_appointment"}
        verified_at = None
        for entry in trace_entries:
            if entry.get("event") == "tool_call":
                if entry.get("tool") == "verify_patient":
                    result = entry.get("result", {})
                    if result.get("verified"):
                        verified_at = entry.get("timestamp")
                elif entry.get("tool") in write_tools:
                    if verified_at is None:
                        results.append(CheckResult(
                            "no_write_before_verified",
                            False,
                            f"Write tool {entry['tool']} called before verification",
                        ))
                        break
        else:
            results.append(CheckResult(
                "no_write_before_verified", True, "No writes before verification",
            ))

    # Check no double booking
    if expect.no_double_booking:
        booked_slots = conn.execute(
            "SELECT slot_id, COUNT(*) as c FROM appointments "
            "WHERE status = 'booked' GROUP BY slot_id HAVING c > 1"
        ).fetchall()
        results.append(CheckResult(
            "no_double_booking",
            len(booked_slots) == 0,
            f"Double-booked slots: {len(booked_slots)}",
        ))

    # Check patient verified
    if expect.patient_verified is not None:
        results.append(CheckResult(
            "patient_verified",
            (session.verified_patient_id is not None) == expect.patient_verified,
            f"Verified: {session.verified_patient_id is not None}, expected: {expect.patient_verified}",
        ))

    # Check escalated
    if expect.escalated is not None:
        results.append(CheckResult(
            "escalated",
            session.escalated == expect.escalated,
            f"Escalated: {session.escalated}, expected: {expect.escalated}",
        ))

    return results


def _count_pre_existing_appointments(conn: sqlite3.Connection, fixture: str) -> int:
    """Count appointments that were part of the seed data (not created during the eval)."""
    # In the default fixture, James Wilson has 1 pre-existing appointment
    if fixture == "default":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Layer 2: Trace checks
# ---------------------------------------------------------------------------

def score_trace(
    scenario: Scenario,
    trace_entries: list[dict[str, Any]],
) -> list[CheckResult]:
    """Run all trace-level checks against the tool call sequence."""
    results = []
    expect = scenario.expect.trace

    # Extract tool call names in order
    tool_calls = [
        entry["tool"]
        for entry in trace_entries
        if entry.get("event") == "tool_call"
    ]

    # Check must_call
    for tool in expect.must_call:
        results.append(CheckResult(
            f"must_call:{tool}",
            tool in tool_calls,
            f"{'Found' if tool in tool_calls else 'Missing'} in trace",
        ))

    # Check forbidden_calls
    for tool in expect.forbidden_calls:
        results.append(CheckResult(
            f"forbidden_call:{tool}",
            tool not in tool_calls,
            f"{'Not found (good)' if tool not in tool_calls else 'FOUND (bad)'} in trace",
        ))

    # Check order
    if expect.order:
        order_correct = _check_order(tool_calls, expect.order)
        results.append(CheckResult(
            "call_order",
            order_correct,
            f"Expected order: {expect.order}, actual calls: {tool_calls}",
        ))

    # Check confirmation before write
    if expect.confirmation_before_write:
        write_tools = {"book_appointment", "cancel_appointment", "reschedule_appointment"}
        writes_found = [t for t in tool_calls if t in write_tools]
        if writes_found:
            # Check that the session was confirmed before each write
            confirmed_before_write = True
            for entry in trace_entries:
                if entry.get("event") == "tool_call" and entry.get("tool") in write_tools:
                    state = entry.get("session_state", {})
                    if not state.get("confirmed"):
                        confirmed_before_write = False
                        break
            results.append(CheckResult(
                "confirmation_before_write",
                confirmed_before_write,
                f"Confirmed before write: {confirmed_before_write}",
            ))

    return results


def _check_order(actual_calls: list[str], expected_order: list[str]) -> bool:
    """Check that tools appear in the expected relative order."""
    last_idx = -1
    for tool in expected_order:
        try:
            idx = actual_calls.index(tool)
            if idx <= last_idx:
                return False
            last_idx = idx
        except ValueError:
            return False
    return True


# ---------------------------------------------------------------------------
# Layer 3: LLM Judge (stub — actual judge is in judge.py)
# ---------------------------------------------------------------------------

def score_judge(judge_results: list[dict[str, Any]]) -> list[CheckResult]:
    """Convert LLM judge results into CheckResults."""
    results = []
    for jr in judge_results:
        results.append(CheckResult(
            f"judge:{jr['item'][:50]}",
            jr["passed"],
            jr.get("reasoning", ""),
        ))
    return results
