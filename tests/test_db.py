"""Tests for database layer — seed data, queries, booking."""

import pytest

from clinic_agent.clinic.db import (
    book_slot,
    cancel_appointment,
    frozen_now,
    get_available_slots,
    get_patient_appointments,
    get_patient_by_name_dob,
    get_slot_by_id,
    init_db,
    reset_db,
    seed_db,
)
from clinic_agent.clinic.models import SlotStatus


@pytest.fixture
def db_conn():
    """Create a fresh in-memory database for each test."""
    conn = init_db(in_memory=True)
    seed_db(conn, "default")
    return conn


class TestSeedData:
    """Verify the default seed data is deterministic and correct."""

    def test_patients_exist(self, db_conn):
        """Three patients should be seeded."""
        rows = db_conn.execute("SELECT COUNT(*) as c FROM patients").fetchone()
        assert rows["c"] == 3

    def test_providers_exist(self, db_conn):
        """Three providers should be seeded."""
        rows = db_conn.execute("SELECT COUNT(*) as c FROM providers").fetchone()
        assert rows["c"] == 3

    def test_slots_exist(self, db_conn):
        """Slots should be seeded for the week."""
        rows = db_conn.execute("SELECT COUNT(*) as c FROM slots").fetchone()
        assert rows["c"] > 0

    def test_james_has_appointment(self, db_conn):
        """James Wilson should have a pre-existing appointment."""
        appts = get_patient_appointments(db_conn, patient_id=2)
        assert len(appts) == 1
        assert appts[0].reason == "Annual checkup"

    def test_seed_is_deterministic(self, db_conn):
        """Two fresh seeds should produce the same data."""
        conn2 = init_db(in_memory=True)
        seed_db(conn2, "default")

        p1 = db_conn.execute("SELECT name FROM patients ORDER BY id").fetchall()
        p2 = conn2.execute("SELECT name FROM patients ORDER BY id").fetchall()
        assert [r["name"] for r in p1] == [r["name"] for r in p2]


class TestPatientLookup:
    """Test patient verification queries."""

    def test_exact_match(self, db_conn):
        patient = get_patient_by_name_dob(db_conn, "Asha Rao", "1990-05-14")
        assert patient is not None
        assert patient.id == 1

    def test_case_insensitive_name(self, db_conn):
        patient = get_patient_by_name_dob(db_conn, "asha rao", "1990-05-14")
        assert patient is not None

    def test_wrong_dob_no_match(self, db_conn):
        patient = get_patient_by_name_dob(db_conn, "Asha Rao", "1990-05-15")
        assert patient is None

    def test_wrong_name_no_match(self, db_conn):
        patient = get_patient_by_name_dob(db_conn, "Unknown Person", "1990-05-14")
        assert patient is None


class TestAvailability:
    """Test slot availability queries."""

    def test_open_slots_exist(self, db_conn):
        slots = get_available_slots(db_conn)
        assert len(slots) > 0

    def test_filter_by_specialty(self, db_conn):
        slots = get_available_slots(db_conn, specialty="Cardiology")
        assert all(s.specialty == "Cardiology" for s in slots)

    def test_no_results_for_fake_specialty(self, db_conn):
        slots = get_available_slots(db_conn, specialty="Underwater Basket Weaving")
        assert len(slots) == 0


class TestBooking:
    """Test booking and cancellation."""

    def test_book_open_slot(self, db_conn):
        slots = get_available_slots(db_conn)
        assert len(slots) > 0

        slot_id = slots[0].slot_id
        appt_id = book_slot(db_conn, patient_id=1, slot_id=slot_id, reason="Checkup")
        assert appt_id is not None

        # Slot should now be booked
        slot = get_slot_by_id(db_conn, slot_id)
        assert slot.status == SlotStatus.BOOKED

    def test_double_book_fails(self, db_conn):
        slots = get_available_slots(db_conn)
        slot_id = slots[0].slot_id

        # First booking succeeds
        appt_id1 = book_slot(db_conn, patient_id=1, slot_id=slot_id, reason="First")
        assert appt_id1 is not None

        # Second booking fails
        appt_id2 = book_slot(db_conn, patient_id=2, slot_id=slot_id, reason="Second")
        assert appt_id2 is None

    def test_cancel_releases_slot(self, db_conn):
        # James has a pre-existing appointment
        appts = get_patient_appointments(db_conn, patient_id=2)
        assert len(appts) == 1

        slot_id = appts[0].slot_id
        success = cancel_appointment(db_conn, appts[0].id)
        assert success

        # Slot should be open again
        slot = get_slot_by_id(db_conn, slot_id)
        assert slot.status == SlotStatus.OPEN

    def test_cancel_nonexistent_fails(self, db_conn):
        success = cancel_appointment(db_conn, 99999)
        assert not success


class TestReset:
    """Test database reset for eval isolation."""

    def test_reset_restores_state(self, db_conn):
        # Book a slot to change state
        slots = get_available_slots(db_conn)
        book_slot(db_conn, patient_id=1, slot_id=slots[0].slot_id, reason="Test")

        # Reset should restore to seed state
        reset_db(db_conn, "default")

        # Asha should have no appointments again
        appts = get_patient_appointments(db_conn, patient_id=1)
        assert len(appts) == 0

        # James should still have his original appointment
        appts = get_patient_appointments(db_conn, patient_id=2)
        assert len(appts) == 1
