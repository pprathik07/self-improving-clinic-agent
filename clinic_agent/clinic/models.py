"""Pydantic models for clinic domain objects.

These are the shared data types used across DB, tools, and the agent.
They are NOT the SQLite schema (that lives in db.py), but they mirror it
so that every boundary passes typed data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SlotStatus(str, Enum):
    """Status of a time slot."""
    OPEN = "open"
    HELD = "held"
    BOOKED = "booked"


class AppointmentStatus(str, Enum):
    """Status of an appointment."""
    BOOKED = "booked"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Provider(BaseModel):
    """A healthcare provider (doctor)."""
    id: int
    name: str
    specialty: str


class Slot(BaseModel):
    """A bookable time slot belonging to a provider."""
    id: int
    provider_id: int
    start: datetime
    end: datetime
    status: SlotStatus = SlotStatus.OPEN


class Patient(BaseModel):
    """A registered patient."""
    id: int
    name: str
    dob: str  # ISO date string YYYY-MM-DD for easy comparison
    phone: str


class Appointment(BaseModel):
    """A booked or cancelled appointment."""
    id: int
    patient_id: int
    slot_id: int
    status: AppointmentStatus = AppointmentStatus.BOOKED
    reason: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Convenience model for availability display
# ---------------------------------------------------------------------------

class SlotInfo(BaseModel):
    """A slot enriched with provider details, for display to the patient."""
    slot_id: int
    provider_name: str
    specialty: str
    start: datetime
    end: datetime
