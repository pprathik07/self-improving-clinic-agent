"""Code-enforced safety guards for tools.

These guards are the primary safety layer. They are NOT prompt-editable
and NOT modifiable by the improvement loop. They enforce:
- Identity verification before any patient-specific action
- Explicit confirmation before any write action
- Ownership checks (a patient can only act on their own records)
- Verify-attempt lockout after 3 failures
- State-machine enforcement (tools only callable in allowed states)
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from clinic_agent.clinic.db import get_appointment_by_id

if TYPE_CHECKING:
    from clinic_agent.agent.state import SessionState


class GuardError(Exception):
    """Raised when a code-enforced guard blocks a tool call.

    The message is safe to show to the patient (no internal details leaked).
    """

    def __init__(self, message: str, internal_reason: str = ""):
        super().__init__(message)
        self.patient_message = message
        self.internal_reason = internal_reason or message


# ---------------------------------------------------------------------------
# Individual guards
# ---------------------------------------------------------------------------

def require_verified(session: SessionState) -> None:
    """Block if the patient has not been verified."""
    if session.verified_patient_id is None:
        raise GuardError(
            "I need to verify your identity first. Could you please provide your full name and date of birth?",
            internal_reason="Tool called without verified_patient_id",
        )


def require_confirmed(session: SessionState) -> None:
    """Block if the pending action has not been explicitly confirmed."""
    if not session.confirmed:
        raise GuardError(
            "I need you to confirm the action before I can proceed. Would you like to go ahead?",
            internal_reason="Tool called without confirmation",
        )


def require_ownership(
    session: SessionState, appointment_id: int, conn: sqlite3.Connection
) -> None:
    """Block if the appointment does not belong to the verified patient."""
    appt = get_appointment_by_id(conn, appointment_id)
    if appt is None:
        raise GuardError(
            "I couldn't find that appointment. Could you double-check the details?",
            internal_reason=f"Appointment {appointment_id} not found",
        )
    if appt.patient_id != session.verified_patient_id:
        raise GuardError(
            "I can only help you with your own appointments.",
            internal_reason=(
                f"Ownership mismatch: appointment {appointment_id} belongs to "
                f"patient {appt.patient_id}, not {session.verified_patient_id}"
            ),
        )


def check_verify_lockout(session: SessionState) -> bool:
    """Return True if the patient has exceeded the verification attempt limit."""
    return session.verify_attempts >= 3


def require_state(session: SessionState, allowed_states: set[str]) -> None:
    """Block if the session is not in one of the allowed states."""
    if session.state.value not in {s if isinstance(s, str) else s.value for s in allowed_states}:
        raise GuardError(
            "I can't do that right now. Let me help you with the current step first.",
            internal_reason=(
                f"State guard: current state {session.state.value} "
                f"not in allowed {allowed_states}"
            ),
        )


def require_slot_open(conn: sqlite3.Connection, slot_id: int) -> None:
    """Block if the slot is no longer open (race condition guard)."""
    from clinic_agent.clinic.db import get_slot_by_id
    from clinic_agent.clinic.models import SlotStatus

    slot = get_slot_by_id(conn, slot_id)
    if slot is None:
        raise GuardError(
            "That time slot doesn't exist. Let me help you find another one.",
            internal_reason=f"Slot {slot_id} not found",
        )
    if slot.status != SlotStatus.OPEN:
        raise GuardError(
            "I'm sorry, that time slot is no longer available. Let me find another one for you.",
            internal_reason=f"Slot {slot_id} status is {slot.status}, not open",
        )
