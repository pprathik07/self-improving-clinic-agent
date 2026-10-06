"""Tool implementations — the actual logic behind each tool.

Each function takes (args, session, conn) and returns a typed result.
Guards are applied before the function body; the function trusts that
guards have already passed.

Key safety invariant: tools never take patient_id from the LLM.
It always comes from session.verified_patient_id.
"""

from __future__ import annotations

import sqlite3
import logging
from datetime import datetime

from clinic_agent.agent.state import AgentState, SessionState
from clinic_agent.clinic import db
from clinic_agent.clinic.models import SlotStatus
from clinic_agent.tools.guards import (
    GuardError,
    check_verify_lockout,
    require_confirmed,
    require_ownership,
    require_slot_open,
    require_state,
    require_verified,
)
from clinic_agent.tools.schemas import (
    AppointmentInfo,
    AvailabilityResult,
    BookArgs,
    BookResult,
    CancelArgs,
    CancelResult,
    CheckAvailabilityArgs,
    EscalateArgs,
    EscalateResult,
    GetAppointmentsArgs,
    GetAppointmentsResult,
    RescheduleArgs,
    RescheduleResult,
    SlotInfoResponse,
    VerifyPatientArgs,
    VerifyResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# verify_patient
# ---------------------------------------------------------------------------

def verify_patient(
    args: VerifyPatientArgs, session: SessionState, conn: sqlite3.Connection
) -> VerifyResult:
    """Verify a patient's identity by name + DOB.

    Guards: must be in IDENTIFY state. Locks out after 3 failures.
    """
    require_state(session, {AgentState.IDENTIFY})

    # Check lockout first
    if check_verify_lockout(session):
        session.escalated = True
        session.state = AgentState.ESCALATED
        return VerifyResult(
            verified=False,
            message="Too many failed verification attempts. Let me connect you with a staff member who can help.",
        )

    patient = db.get_patient_by_name_dob(conn, args.name, args.dob)

    if patient is None:
        session.verify_attempts += 1
        remaining = 3 - session.verify_attempts
        if remaining <= 0:
            session.escalated = True
            session.state = AgentState.ESCALATED
            return VerifyResult(
                verified=False,
                message="I wasn't able to verify your identity. Let me connect you with a staff member who can help.",
            )
        return VerifyResult(
            verified=False,
            message=f"I couldn't find a match. Please double-check your name and date of birth. You have {remaining} attempt(s) remaining.",
        )

    # Successful verification
    session.verified_patient_id = patient.id
    session.state = AgentState.VERIFIED
    return VerifyResult(
        verified=True,
        patient_id=patient.id,
        message=f"Identity verified. Welcome, {patient.name}! How can I help you today?",
    )


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------

def check_availability(
    args: CheckAvailabilityArgs, session: SessionState, conn: sqlite3.Connection
) -> AvailabilityResult:
    """Check available appointment slots.

    Guards: allowed in INTENT and SLOT_SELECTION states.
    Does NOT require verification (availability is public info).
    """
    require_state(session, {AgentState.INTENT, AgentState.SLOT_SELECTION})

    slots = db.get_available_slots(
        conn,
        specialty=args.specialty,
        provider_id=args.provider_id,
        date_start=args.date_start,
        date_end=args.date_end,
    )

    if not slots:
        return AvailabilityResult(
            slots=[],
            message="I don't see any available slots matching those criteria. Would you like to try different dates or a different specialty?",
        )

    slot_responses = [
        SlotInfoResponse(
            slot_id=s.slot_id,
            provider_name=s.provider_name,
            specialty=s.specialty,
            start=s.start.isoformat(),
            end=s.end.isoformat(),
        )
        for s in slots
    ]

    # Advance state: if we found slots, move to SLOT_SELECTION
    if session.state == AgentState.INTENT:
        session.state = AgentState.SLOT_SELECTION

    return AvailabilityResult(
        slots=slot_responses,
        message=f"I found {len(slot_responses)} available slot(s).",
    )


# ---------------------------------------------------------------------------
# book_appointment
# ---------------------------------------------------------------------------

def book_appointment(
    args: BookArgs, session: SessionState, conn: sqlite3.Connection
) -> BookResult:
    """Book an appointment.

    Guards: must be verified, confirmed, in EXECUTE state, and slot must be open.
    patient_id comes from session, NOT from args.
    """
    require_state(session, {AgentState.EXECUTE})
    require_verified(session)
    require_confirmed(session)
    require_slot_open(conn, args.slot_id)

    appointment_id = db.book_slot(
        conn,
        patient_id=session.verified_patient_id,  # NEVER from LLM
        slot_id=args.slot_id,
        reason=args.reason,
    )

    if appointment_id is None:
        return BookResult(
            success=False,
            message="I'm sorry, that slot was just taken. Let me find another one for you.",
        )

    # Get slot details for the confirmation message
    slot = db.get_slot_by_id(conn, args.slot_id)
    provider = db.get_provider_by_id(conn, slot.provider_id) if slot else None

    session.state = AgentState.DONE
    session.confirmed = False  # Reset after use

    return BookResult(
        success=True,
        appointment_id=appointment_id,
        message=(
            f"Your appointment has been booked! "
            f"Appointment #{appointment_id} with {provider.name if provider else 'the provider'} "
            f"on {slot.start.strftime('%A, %B %d at %I:%M %p') if slot else 'the selected time'}. "
            f"Is there anything else I can help you with?"
        ),
    )


# ---------------------------------------------------------------------------
# cancel_appointment
# ---------------------------------------------------------------------------

def cancel_appointment(
    args: CancelArgs, session: SessionState, conn: sqlite3.Connection
) -> CancelResult:
    """Cancel an appointment.

    Guards: must be verified, confirmed, in EXECUTE state,
    and appointment must belong to the verified patient.
    """
    require_state(session, {AgentState.EXECUTE})
    require_verified(session)
    require_confirmed(session)
    require_ownership(session, args.appointment_id, conn)

    success = db.cancel_appointment(conn, args.appointment_id)

    if not success:
        return CancelResult(
            success=False,
            message="I wasn't able to cancel that appointment. It may have already been cancelled.",
        )

    session.state = AgentState.DONE
    session.confirmed = False

    return CancelResult(
        success=True,
        message=f"Appointment #{args.appointment_id} has been cancelled. Is there anything else I can help you with?",
    )


# ---------------------------------------------------------------------------
# reschedule_appointment
# ---------------------------------------------------------------------------

def reschedule_appointment(
    args: RescheduleArgs, session: SessionState, conn: sqlite3.Connection
) -> RescheduleResult:
    """Reschedule an appointment to a new slot.

    Guards: must be verified, confirmed, in EXECUTE state,
    appointment must belong to patient, new slot must be open.
    Implemented as cancel + book (transactional).
    """
    require_state(session, {AgentState.EXECUTE})
    require_verified(session)
    require_confirmed(session)
    require_ownership(session, args.appointment_id, conn)
    require_slot_open(conn, args.new_slot_id)

    # Get the old appointment to preserve the reason
    old_appt = db.get_appointment_by_id(conn, args.appointment_id)
    if old_appt is None:
        return RescheduleResult(
            success=False,
            message="I couldn't find that appointment to reschedule.",
        )

    # Cancel old
    cancelled = db.cancel_appointment(conn, args.appointment_id)
    if not cancelled:
        return RescheduleResult(
            success=False,
            message="I wasn't able to reschedule. The original appointment may have already been cancelled.",
        )

    # Book new
    new_id = db.book_slot(
        conn,
        patient_id=session.verified_patient_id,
        slot_id=args.new_slot_id,
        reason=old_appt.reason,
    )

    if new_id is None:
        # New slot was taken — this is a partial failure. The old appointment
        # is already cancelled, so we need to inform the patient.
        return RescheduleResult(
            success=False,
            message="The new time slot was no longer available. Your original appointment has been cancelled. Let me help you find another time.",
        )

    session.state = AgentState.DONE
    session.confirmed = False

    slot = db.get_slot_by_id(conn, args.new_slot_id)

    return RescheduleResult(
        success=True,
        new_appointment_id=new_id,
        message=(
            f"Your appointment has been rescheduled. "
            f"New appointment #{new_id} on "
            f"{slot.start.strftime('%A, %B %d at %I:%M %p') if slot else 'the new time'}. "
            f"Is there anything else I can help you with?"
        ),
    )


# ---------------------------------------------------------------------------
# escalate_to_human
# ---------------------------------------------------------------------------

def escalate_to_human(
    args: EscalateArgs, session: SessionState, conn: sqlite3.Connection
) -> EscalateResult:
    """Escalate the conversation to a human staff member.

    Always allowed regardless of state.
    """
    session.escalated = True
    session.state = AgentState.ESCALATED

    return EscalateResult(
        message="I'm connecting you with a staff member who can help. Please stay on the line. Thank you for your patience.",
    )


# ---------------------------------------------------------------------------
# get_patient_appointments
# ---------------------------------------------------------------------------

def get_patient_appointments(
    args: GetAppointmentsArgs, session: SessionState, conn: sqlite3.Connection
) -> GetAppointmentsResult:
    """Get the verified patient's appointments.

    Guards: must be verified. Patient ID comes from session.
    """
    require_verified(session)

    appointments = db.get_patient_appointments(conn, session.verified_patient_id)

    if not appointments:
        return GetAppointmentsResult(
            appointments=[],
            message="You don't have any upcoming appointments.",
        )

    infos = []
    for appt in appointments:
        slot = db.get_slot_by_id(conn, appt.slot_id)
        provider = db.get_provider_by_id(conn, slot.provider_id) if slot else None
        infos.append(
            AppointmentInfo(
                appointment_id=appt.id,
                provider_name=provider.name if provider else "Unknown",
                specialty=provider.specialty if provider else "Unknown",
                start=slot.start.isoformat() if slot else "",
                end=slot.end.isoformat() if slot else "",
                reason=appt.reason,
                status=appt.status.value,
            )
        )

    # Advance state for cancel/reschedule flow
    if session.state == AgentState.INTENT:
        session.state = AgentState.CONFIRM

    return GetAppointmentsResult(
        appointments=infos,
        message=f"You have {len(infos)} upcoming appointment(s).",
    )


# ---------------------------------------------------------------------------
# Tool registry — maps names to (function, args_model) for the runner
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, tuple[callable, type]] = {
    "verify_patient": (verify_patient, VerifyPatientArgs),
    "check_availability": (check_availability, CheckAvailabilityArgs),
    "book_appointment": (book_appointment, BookArgs),
    "cancel_appointment": (cancel_appointment, CancelArgs),
    "reschedule_appointment": (reschedule_appointment, RescheduleArgs),
    "escalate_to_human": (escalate_to_human, EscalateArgs),
    "get_patient_appointments": (get_patient_appointments, GetAppointmentsArgs),
}
