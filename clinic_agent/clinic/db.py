"""SQLite database layer — schema, seed data, reset.

All timestamps are relative to FROZEN_NOW (from env), never wall-clock time.
Supports in-memory DBs for eval isolation and file DBs for demo mode.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

from clinic_agent.clinic.models import (
    Appointment,
    AppointmentStatus,
    Patient,
    Provider,
    Slot,
    SlotInfo,
    SlotStatus,
)

# ---------------------------------------------------------------------------
# Clock — always frozen, never datetime.now()
# ---------------------------------------------------------------------------

def frozen_now() -> datetime:
    """Return the frozen clock from env, defaulting to a Monday morning."""
    raw = os.environ.get("FROZEN_NOW", "2025-01-13T09:00:00")
    return datetime.fromisoformat(raw)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES providers(id),
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    dob TEXT NOT NULL,
    phone TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    slot_id INTEGER NOT NULL REFERENCES slots(id),
    status TEXT NOT NULL DEFAULT 'booked',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Init / Reset
# ---------------------------------------------------------------------------

def init_db(in_memory: bool = True) -> sqlite3.Connection:
    """Create a new database and apply the schema.

    Args:
        in_memory: If True, use ':memory:'. Otherwise use 'clinic.db'.
    """
    db_path = ":memory:" if in_memory else "clinic.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def reset_db(conn: sqlite3.Connection, fixture: str = "default") -> None:
    """Drop all data and re-seed. Used before each eval run."""
    conn.executescript("""
        DELETE FROM appointments;
        DELETE FROM slots;
        DELETE FROM patients;
        DELETE FROM providers;
    """)
    seed_db(conn, fixture)


# ---------------------------------------------------------------------------
# Seed data (deterministic)
# ---------------------------------------------------------------------------

# Named fixtures let scenarios use different data setups.
_FIXTURES: dict[str, callable] = {}


def _register_fixture(name: str):
    """Decorator to register a seed fixture by name."""
    def decorator(fn):
        _FIXTURES[name] = fn
        return fn
    return decorator


def seed_db(conn: sqlite3.Connection, fixture: str = "default") -> None:
    """Seed the database with a named fixture."""
    fn = _FIXTURES.get(fixture)
    if fn is None:
        raise ValueError(f"Unknown fixture: {fixture!r}. Available: {list(_FIXTURES)}")
    fn(conn)


@_register_fixture("default")
def _seed_default(conn: sqlite3.Connection) -> None:
    """Default fixture: 3 providers, weekday slots for the week, 3 patients."""
    now = frozen_now()

    # --- Providers ---
    providers = [
        (1, "Dr. Sarah Chen", "General Practice"),
        (2, "Dr. Raj Patel", "Cardiology"),
        (3, "Dr. Emily Okafor", "Dermatology"),
    ]
    conn.executemany(
        "INSERT INTO providers (id, name, specialty) VALUES (?, ?, ?)",
        providers,
    )

    # --- Slots: weekday slots for the current week, 20 min each, 9am-5pm ---
    slot_id = 1
    # Find the Monday of the current week
    monday = now - timedelta(days=now.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    slots = []
    for day_offset in range(5):  # Mon-Fri
        day = monday + timedelta(days=day_offset)
        for provider_id, _, _ in providers:
            hour = 9
            while hour < 17:
                for minute_offset in [0, 20, 40]:
                    start = day.replace(hour=hour, minute=minute_offset)
                    end = start + timedelta(minutes=20)
                    # Only create slots that are in the future relative to FROZEN_NOW
                    if start >= now:
                        slots.append((slot_id, provider_id, start.isoformat(), end.isoformat(), "open"))
                        slot_id += 1
                hour += 1

    conn.executemany(
        "INSERT INTO slots (id, provider_id, start, end, status) VALUES (?, ?, ?, ?, ?)",
        slots,
    )

    # --- Patients ---
    patients = [
        (1, "Asha Rao", "1990-05-14", "555-0101"),
        (2, "James Wilson", "1985-11-22", "555-0102"),
        (3, "Maria Garcia", "1978-03-08", "555-0103"),
    ]
    conn.executemany(
        "INSERT INTO patients (id, name, dob, phone) VALUES (?, ?, ?, ?)",
        patients,
    )

    # --- Pre-existing appointment for James Wilson (for cancel/reschedule scenarios) ---
    # Book a slot on Wednesday at 10am with Dr. Chen
    wednesday_10am_slot = None
    for sid, pid, start, end, status in slots:
        if pid == 1:  # Dr. Chen
            dt = datetime.fromisoformat(start)
            if dt.weekday() == 2 and dt.hour == 10 and dt.minute == 0:  # Wednesday 10am
                wednesday_10am_slot = sid
                break

    if wednesday_10am_slot:
        conn.execute(
            "UPDATE slots SET status = 'booked' WHERE id = ?",
            (wednesday_10am_slot,),
        )
        conn.execute(
            "INSERT INTO appointments (patient_id, slot_id, status, reason, created_at) "
            "VALUES (?, ?, 'booked', 'Annual checkup', ?)",
            (2, wednesday_10am_slot, now.isoformat()),
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Query helpers (used by tools)
# ---------------------------------------------------------------------------

def get_patient_by_name_dob(conn: sqlite3.Connection, name: str, dob: str) -> Patient | None:
    """Look up a patient by name (case-insensitive) and DOB (exact match)."""
    row = conn.execute(
        "SELECT * FROM patients WHERE LOWER(name) = LOWER(?) AND dob = ?",
        (name.strip(), dob.strip()),
    ).fetchone()
    if row:
        return Patient(id=row["id"], name=row["name"], dob=row["dob"], phone=row["phone"])
    return None


def get_available_slots(
    conn: sqlite3.Connection,
    specialty: str | None = None,
    provider_id: int | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> list[SlotInfo]:
    """Return open slots, optionally filtered by specialty/provider/date range."""
    query = """
        SELECT s.id as slot_id, p.name as provider_name, p.specialty,
               s.start, s.end
        FROM slots s
        JOIN providers p ON s.provider_id = p.id
        WHERE s.status = 'open'
    """
    params: list = []

    if specialty:
        query += " AND LOWER(p.specialty) = LOWER(?)"
        params.append(specialty)
    if provider_id:
        query += " AND s.provider_id = ?"
        params.append(provider_id)
    if date_start:
        query += " AND s.start >= ?"
        params.append(date_start)
    if date_end:
        query += " AND s.start <= ?"
        params.append(date_end)

    query += " ORDER BY s.start LIMIT 20"

    rows = conn.execute(query, params).fetchall()
    return [
        SlotInfo(
            slot_id=r["slot_id"],
            provider_name=r["provider_name"],
            specialty=r["specialty"],
            start=datetime.fromisoformat(r["start"]),
            end=datetime.fromisoformat(r["end"]),
        )
        for r in rows
    ]


def get_slot_by_id(conn: sqlite3.Connection, slot_id: int) -> Slot | None:
    """Get a single slot by ID."""
    row = conn.execute("SELECT * FROM slots WHERE id = ?", (slot_id,)).fetchone()
    if row:
        return Slot(
            id=row["id"],
            provider_id=row["provider_id"],
            start=datetime.fromisoformat(row["start"]),
            end=datetime.fromisoformat(row["end"]),
            status=SlotStatus(row["status"]),
        )
    return None


def get_appointment_by_id(conn: sqlite3.Connection, appointment_id: int) -> Appointment | None:
    """Get a single appointment by ID."""
    row = conn.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,)).fetchone()
    if row:
        return Appointment(
            id=row["id"],
            patient_id=row["patient_id"],
            slot_id=row["slot_id"],
            status=AppointmentStatus(row["status"]),
            reason=row["reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
    return None


def get_patient_appointments(conn: sqlite3.Connection, patient_id: int) -> list[Appointment]:
    """Get all active (booked) appointments for a patient."""
    rows = conn.execute(
        "SELECT * FROM appointments WHERE patient_id = ? AND status = 'booked' ORDER BY created_at",
        (patient_id,),
    ).fetchall()
    return [
        Appointment(
            id=r["id"],
            patient_id=r["patient_id"],
            slot_id=r["slot_id"],
            status=AppointmentStatus(r["status"]),
            reason=r["reason"],
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in rows
    ]


def book_slot(
    conn: sqlite3.Connection,
    patient_id: int,
    slot_id: int,
    reason: str,
) -> int | None:
    """Atomically book a slot. Returns appointment ID or None if slot is no longer open."""
    # Check slot is still open (race-condition safe within single connection)
    slot = get_slot_by_id(conn, slot_id)
    if slot is None or slot.status != SlotStatus.OPEN:
        return None

    now = frozen_now()
    conn.execute("UPDATE slots SET status = 'booked' WHERE id = ?", (slot_id,))
    cursor = conn.execute(
        "INSERT INTO appointments (patient_id, slot_id, status, reason, created_at) "
        "VALUES (?, ?, 'booked', ?, ?)",
        (patient_id, slot_id, reason, now.isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def cancel_appointment(conn: sqlite3.Connection, appointment_id: int) -> bool:
    """Cancel an appointment and release the slot. Returns True on success."""
    appt = get_appointment_by_id(conn, appointment_id)
    if appt is None or appt.status != AppointmentStatus.BOOKED:
        return False

    conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,))
    conn.execute("UPDATE slots SET status = 'open' WHERE id = ?", (appt.slot_id,))
    conn.commit()
    return True


def get_provider_by_id(conn: sqlite3.Connection, provider_id: int) -> Provider | None:
    """Get a provider by ID."""
    row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
    if row:
        return Provider(id=row["id"], name=row["name"], specialty=row["specialty"])
    return None
