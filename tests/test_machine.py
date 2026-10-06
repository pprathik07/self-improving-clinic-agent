"""Tests for the state machine — transitions and tool allowlists."""

import pytest

from clinic_agent.agent.machine import (
    get_allowed_tools,
    is_terminal,
    is_valid_transition,
    transition,
)
from clinic_agent.agent.state import AgentState


class TestTransitions:
    """Every state has valid exits; illegal transitions stay put."""

    def test_greeting_to_identify(self):
        assert is_valid_transition(AgentState.GREETING, AgentState.IDENTIFY)

    def test_identify_to_verified(self):
        assert is_valid_transition(AgentState.IDENTIFY, AgentState.VERIFIED)

    def test_verified_to_intent(self):
        assert is_valid_transition(AgentState.VERIFIED, AgentState.INTENT)

    def test_intent_to_slot_selection(self):
        assert is_valid_transition(AgentState.INTENT, AgentState.SLOT_SELECTION)

    def test_slot_selection_to_confirm(self):
        assert is_valid_transition(AgentState.SLOT_SELECTION, AgentState.CONFIRM)

    def test_confirm_to_execute(self):
        assert is_valid_transition(AgentState.CONFIRM, AgentState.EXECUTE)

    def test_execute_to_done(self):
        assert is_valid_transition(AgentState.EXECUTE, AgentState.DONE)

    def test_any_state_can_escalate(self):
        """ESCALATED should be reachable from any non-terminal state."""
        non_terminal = [s for s in AgentState if not is_terminal(s)]
        for state in non_terminal:
            assert is_valid_transition(state, AgentState.ESCALATED), (
                f"{state} should allow transition to ESCALATED"
            )

    def test_illegal_transition_blocked(self):
        """Cannot jump from GREETING directly to EXECUTE."""
        assert not is_valid_transition(AgentState.GREETING, AgentState.EXECUTE)

    def test_transition_fn_stays_on_invalid(self):
        """transition() returns current state on invalid transition."""
        result = transition(AgentState.GREETING, AgentState.EXECUTE)
        assert result == AgentState.GREETING

    def test_transition_fn_moves_on_valid(self):
        result = transition(AgentState.GREETING, AgentState.IDENTIFY)
        assert result == AgentState.IDENTIFY

    def test_done_is_terminal(self):
        assert is_terminal(AgentState.DONE)

    def test_escalated_is_terminal(self):
        assert is_terminal(AgentState.ESCALATED)

    def test_greeting_is_not_terminal(self):
        assert not is_terminal(AgentState.GREETING)

    def test_terminal_states_have_no_exits(self):
        """DONE and ESCALATED should have no valid next states."""
        assert not is_valid_transition(AgentState.DONE, AgentState.GREETING)
        assert not is_valid_transition(AgentState.ESCALATED, AgentState.GREETING)


class TestAllowedTools:
    """Tools are only available in their designated states."""

    def test_verify_only_in_identify(self):
        assert "verify_patient" in get_allowed_tools(AgentState.IDENTIFY)
        assert "verify_patient" not in get_allowed_tools(AgentState.GREETING)
        assert "verify_patient" not in get_allowed_tools(AgentState.INTENT)

    def test_book_only_in_execute(self):
        assert "book_appointment" in get_allowed_tools(AgentState.EXECUTE)
        assert "book_appointment" not in get_allowed_tools(AgentState.IDENTIFY)

    def test_check_availability_in_intent_and_selection(self):
        assert "check_availability" in get_allowed_tools(AgentState.INTENT)
        assert "check_availability" in get_allowed_tools(AgentState.SLOT_SELECTION)
        assert "check_availability" not in get_allowed_tools(AgentState.EXECUTE)

    def test_escalate_available_in_non_terminal(self):
        """escalate_to_human should be available in all non-terminal states."""
        for state in AgentState:
            if is_terminal(state):
                continue
            assert "escalate_to_human" in get_allowed_tools(state), (
                f"escalate_to_human should be allowed in {state}"
            )

    def test_no_tools_in_done(self):
        assert get_allowed_tools(AgentState.DONE) == set()

    def test_no_tools_in_escalated(self):
        assert get_allowed_tools(AgentState.ESCALATED) == set()
