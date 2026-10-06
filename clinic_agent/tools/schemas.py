"""Pydantic schemas for tool arguments and return values.

Every tool boundary is typed — the LLM's JSON is validated against
the input model, and every return is a typed output model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# verify_patient
# ---------------------------------------------------------------------------

class VerifyPatientArgs(BaseModel):
    """Arguments the LLM supplies to verify a patient."""
    name: str = Field(description="Patient's full name")
    dob: str = Field(description="Patient's date of birth in YYYY-MM-DD format")


class VerifyResult(BaseModel):
    """Result of a verification attempt."""
    verified: bool
    patient_id: int | None = None
    message: str


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------

class CheckAvailabilityArgs(BaseModel):
    """Arguments for checking available appointment slots."""
    specialty: str | None = Field(default=None, description="Medical specialty to filter by")
    provider_id: int | None = Field(default=None, description="Specific provider ID")
    date_start: str | None = Field(default=None, description="Start of date range (YYYY-MM-DD or ISO)")
    date_end: str | None = Field(default=None, description="End of date range (YYYY-MM-DD or ISO)")


class AvailabilityResult(BaseModel):
    """Available slots returned to the agent."""
    slots: list[SlotInfoResponse]
    message: str


class SlotInfoResponse(BaseModel):
    """A single available slot for display."""
    slot_id: int
    provider_name: str
    specialty: str
    start: str  # ISO formatted for LLM consumption
    end: str


# ---------------------------------------------------------------------------
# book_appointment
# ---------------------------------------------------------------------------

class BookArgs(BaseModel):
    """Arguments for booking an appointment."""
    slot_id: int = Field(description="The slot ID to book")
    reason: str = Field(description="Reason for the appointment")


class BookResult(BaseModel):
    """Result of a booking attempt."""
    success: bool
    appointment_id: int | None = None
    message: str


# ---------------------------------------------------------------------------
# reschedule_appointment
# ---------------------------------------------------------------------------

class RescheduleArgs(BaseModel):
    """Arguments for rescheduling an existing appointment."""
    appointment_id: int = Field(description="The appointment ID to reschedule")
    new_slot_id: int = Field(description="The new slot ID to move to")


class RescheduleResult(BaseModel):
    """Result of a reschedule attempt."""
    success: bool
    new_appointment_id: int | None = None
    message: str


# ---------------------------------------------------------------------------
# cancel_appointment
# ---------------------------------------------------------------------------

class CancelArgs(BaseModel):
    """Arguments for cancelling an appointment."""
    appointment_id: int = Field(description="The appointment ID to cancel")


class CancelResult(BaseModel):
    """Result of a cancellation attempt."""
    success: bool
    message: str


# ---------------------------------------------------------------------------
# escalate_to_human
# ---------------------------------------------------------------------------

class EscalateArgs(BaseModel):
    """Arguments for escalating to a human operator."""
    reason: str = Field(description="Why the conversation is being escalated")


class EscalateResult(BaseModel):
    """Result of an escalation."""
    message: str


# ---------------------------------------------------------------------------
# get_patient_appointments (read tool for verified patients)
# ---------------------------------------------------------------------------

class GetAppointmentsArgs(BaseModel):
    """Arguments for retrieving a patient's appointments. No args from LLM —
    patient_id comes from session."""
    pass


class AppointmentInfo(BaseModel):
    """A single appointment for display."""
    appointment_id: int
    provider_name: str
    specialty: str
    start: str
    end: str
    reason: str
    status: str


class GetAppointmentsResult(BaseModel):
    """Patient's appointments."""
    appointments: list[AppointmentInfo]
    message: str
