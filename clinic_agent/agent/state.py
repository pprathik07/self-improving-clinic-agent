"""Session state — explicit, typed, per-conversation.

Every piece of conversation state is tracked here, not in prompt history alone.
The state machine transitions are validated against this object.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Finite set of states the agent can be in."""
    GREETING = "GREETING"
    IDENTIFY = "IDENTIFY"
    VERIFIED = "VERIFIED"
    INTENT = "INTENT"
    SLOT_SELECTION = "SLOT_SELECTION"
    CONFIRM = "CONFIRM"
    EXECUTE = "EXECUTE"
    DONE = "DONE"
    ESCALATED = "ESCALATED"


class PendingAction(BaseModel):
    """The action that has been presented to the patient for confirmation."""
    action_type: str  # "book" | "cancel" | "reschedule"
    details: dict[str, Any] = Field(default_factory=dict)
    description: str = ""  # human-readable summary shown to the patient


class Message(BaseModel):
    """A single message in the conversation history."""
    role: str  # "system" | "user" | "assistant" | "tool_result"
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None


class SessionState(BaseModel):
    """Complete, serializable session state for one conversation.

    This is the single source of truth for where the conversation is
    and what has been verified/confirmed. Tools read from this, never
    from LLM-generated text.
    """
    state: AgentState = AgentState.GREETING
    verified_patient_id: int | None = None
    intent: str | None = None  # "book" | "reschedule" | "cancel" | None
    selected_slot_id: int | None = None
    pending_action: PendingAction | None = None
    confirmed: bool = False
    escalated: bool = False
    verify_attempts: int = 0
    messages: list[Message] = Field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot for logging (excludes message history)."""
        return {
            "state": self.state.value,
            "verified_patient_id": self.verified_patient_id,
            "intent": self.intent,
            "selected_slot_id": self.selected_slot_id,
            "pending_action": self.pending_action.model_dump() if self.pending_action else None,
            "confirmed": self.confirmed,
            "escalated": self.escalated,
            "verify_attempts": self.verify_attempts,
        }
