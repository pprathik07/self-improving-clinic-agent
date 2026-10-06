"""State machine — defines valid transitions and allowed tools per state.

Every state has an explicit set of valid next states and allowed tools.
An illegal transition results in a clarification, never a silent pass-through.
"""

from __future__ import annotations

from clinic_agent.agent.state import AgentState


# ---------------------------------------------------------------------------
# Transition table: state -> set of valid next states
# ---------------------------------------------------------------------------

TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.GREETING: {AgentState.IDENTIFY, AgentState.ESCALATED, AgentState.DONE},
    AgentState.IDENTIFY: {AgentState.VERIFIED, AgentState.ESCALATED, AgentState.DONE},
    AgentState.VERIFIED: {AgentState.INTENT, AgentState.ESCALATED, AgentState.DONE},
    AgentState.INTENT: {AgentState.SLOT_SELECTION, AgentState.CONFIRM, AgentState.DONE, AgentState.ESCALATED},
    AgentState.SLOT_SELECTION: {AgentState.CONFIRM, AgentState.INTENT, AgentState.DONE, AgentState.ESCALATED},
    AgentState.CONFIRM: {AgentState.EXECUTE, AgentState.SLOT_SELECTION, AgentState.INTENT, AgentState.DONE, AgentState.ESCALATED},
    AgentState.EXECUTE: {AgentState.DONE, AgentState.ESCALATED},
    AgentState.DONE: set(),  # terminal
    AgentState.ESCALATED: set(),  # terminal
}


# ---------------------------------------------------------------------------
# Allowed tools per state
# ---------------------------------------------------------------------------

ALLOWED_TOOLS: dict[AgentState, set[str]] = {
    AgentState.GREETING: {"escalate_to_human"},
    AgentState.IDENTIFY: {"verify_patient", "escalate_to_human"},
    AgentState.VERIFIED: {"escalate_to_human"},
    AgentState.INTENT: {"check_availability", "get_patient_appointments", "escalate_to_human"},
    AgentState.SLOT_SELECTION: {"check_availability", "escalate_to_human"},
    AgentState.CONFIRM: {"escalate_to_human"},
    AgentState.EXECUTE: {"book_appointment", "cancel_appointment", "reschedule_appointment", "escalate_to_human"},
    AgentState.DONE: set(),
    AgentState.ESCALATED: set(),
}


def is_valid_transition(current: AgentState, target: AgentState) -> bool:
    """Check if a state transition is valid."""
    return target in TRANSITIONS.get(current, set())


def get_allowed_tools(state: AgentState) -> set[str]:
    """Return the set of tool names allowed in the given state."""
    return ALLOWED_TOOLS.get(state, set())


def is_terminal(state: AgentState) -> bool:
    """Check if a state is terminal (DONE or ESCALATED)."""
    return state in {AgentState.DONE, AgentState.ESCALATED}


def transition(current: AgentState, target: AgentState) -> AgentState:
    """Attempt a state transition. Returns the new state or current if invalid.

    Never silently passes through — the caller should handle invalid
    transitions by issuing a clarification.
    """
    if is_valid_transition(current, target):
        return target
    return current  # stay put; caller issues clarification
