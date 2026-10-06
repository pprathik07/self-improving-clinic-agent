"""Tests for code-enforced safety guards.

These tests verify that the primary safety layer (code guards) correctly
blocks unauthorized, unverified, and unconfirmed operations.
"""

import pytest

from clinic_agent.agent.state import AgentState, SessionState
from clinic_agent.clinic.db import init_db, seed_db, get_patient_appointments
from clinic_agent.tools.guards import (
    GuardError,
    check_verify_lockout,
    require_confirmed,
    require_ownership,
    require_slot_open,
    require_state,
    require_verified,
)


@pytest.fixture
def db_conn():
    """Create a fresh in-memory database for each test."""
    conn = init_db(in_memory=True)
    seed_db(conn, "default")
    return conn


@pytest.fixture
def unverified_session():
    """A session where the patient has not been verified."""
    return SessionState(state=AgentState.IDENTIFY)


@pytest.fixture
def verified_session():
    """A session where the patient has been verified."""
    return SessionState(
        state=AgentState.EXECUTE,
        verified_patient_id=1,  # Asha Rao
        confirmed=False,
    )


@pytest.fixture
def confirmed_session():
    """A session where the patient is verified and has confirmed an action."""
    return SessionState(
        state=AgentState.EXECUTE,
        verified_patient_id=1,
        confirmed=True,
    )


class TestRequireVerified:
    """Write operations must be blocked when patient is not verified."""

    def test_blocks_unverified(self, unverified_session):
        with pytest.raises(GuardError):
            require_verified(unverified_session)

    def test_allows_verified(self, verified_session):
        # Should not raise
        require_verified(verified_session)


class TestRequireConfirmed:
    """Write operations must be blocked when action is not confirmed."""

    def test_blocks_unconfirmed(self, verified_session):
        with pytest.raises(GuardError):
            require_confirmed(verified_session)

    def test_allows_confirmed(self, confirmed_session):
        # Should not raise
        require_confirmed(confirmed_session)


class TestRequireOwnership:
    """A patient can only act on their own appointments."""

    def test_blocks_wrong_owner(self, db_conn):
        """Patient 1 (Asha) cannot cancel patient 2's (James) appointment."""
        session = SessionState(
            state=AgentState.EXECUTE,
            verified_patient_id=1,  # Asha
            confirmed=True,
        )
        # James's appointment
        appts = get_patient_appointments(db_conn, patient_id=2)
        assert len(appts) > 0

        with pytest.raises(GuardError):
            require_ownership(session, appts[0].id, db_conn)

    def test_allows_own_appointment(self, db_conn):
        """Patient 2 (James) can act on his own appointment."""
        session = SessionState(
            state=AgentState.EXECUTE,
            verified_patient_id=2,  # James
            confirmed=True,
        )
        appts = get_patient_appointments(db_conn, patient_id=2)
        assert len(appts) > 0

        # Should not raise
        require_ownership(session, appts[0].id, db_conn)

    def test_blocks_nonexistent_appointment(self, db_conn):
        """Cannot act on an appointment that doesn't exist."""
        session = SessionState(
            state=AgentState.EXECUTE,
            verified_patient_id=1,
            confirmed=True,
        )
        with pytest.raises(GuardError):
            require_ownership(session, 99999, db_conn)


class TestVerifyLockout:
    """Identity verification locks out after 3 failed attempts."""

    def test_not_locked_initially(self):
        session = SessionState(verify_attempts=0)
        assert not check_verify_lockout(session)

    def test_not_locked_at_two(self):
        session = SessionState(verify_attempts=2)
        assert not check_verify_lockout(session)

    def test_locked_at_three(self):
        session = SessionState(verify_attempts=3)
        assert check_verify_lockout(session)

    def test_locked_beyond_three(self):
        session = SessionState(verify_attempts=5)
        assert check_verify_lockout(session)


class TestRequireState:
    """Tools must only be callable in their allowed states."""

    def test_blocks_wrong_state(self):
        session = SessionState(state=AgentState.GREETING)
        with pytest.raises(GuardError):
            require_state(session, {AgentState.EXECUTE})

    def test_allows_correct_state(self):
        session = SessionState(state=AgentState.EXECUTE)
        # Should not raise
        require_state(session, {AgentState.EXECUTE})

    def test_allows_one_of_multiple(self):
        session = SessionState(state=AgentState.INTENT)
        # Should not raise
        require_state(session, {AgentState.INTENT, AgentState.SLOT_SELECTION})


class TestRequireSlotOpen:
    """Booking must be blocked if the slot is already taken."""

    def test_blocks_booked_slot(self, db_conn):
        """Cannot book a slot that is already booked."""
        # James's appointment has a booked slot
        appts = get_patient_appointments(db_conn, patient_id=2)
        assert len(appts) > 0

        with pytest.raises(GuardError):
            require_slot_open(db_conn, appts[0].slot_id)

    def test_allows_open_slot(self, db_conn):
        """Can use an open slot."""
        from clinic_agent.clinic.db import get_available_slots
        slots = get_available_slots(db_conn)
        assert len(slots) > 0

        # Should not raise
        require_slot_open(db_conn, slots[0].slot_id)

    def test_blocks_nonexistent_slot(self, db_conn):
        """Cannot use a slot that doesn't exist."""
        with pytest.raises(GuardError):
            require_slot_open(db_conn, 99999)
