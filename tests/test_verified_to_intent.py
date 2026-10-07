"""VERIFIED → INTENT: intent may arrive with identity or on a later turn."""

from clinic_agent.agent.runner import (
    _maybe_transition_after_verification,
    _maybe_transition_from_user_input,
)
from clinic_agent.agent.state import AgentState, Message, SessionState


class TestVerifiedToIntent:
    def test_identity_and_request_in_one_message_moves_to_intent_after_verification(self):
        """Patient gives name/DOB and a booking request together; after verify → INTENT."""
        session = SessionState()
        session.state = AgentState.IDENTIFY
        session.messages.append(
            Message(
                role="user",
                content=(
                    "Hello, my name is Asha Rao, and my date of birth is May 14, 1990. "
                    "I'm calling to book a dermatology appointment for next Tuesday."
                ),
            )
        )
        # Successful verify_patient lands here (does not re-read the user message).
        session.state = AgentState.VERIFIED
        session.verified_patient_id = 1

        _maybe_transition_after_verification(session)

        assert session.state == AgentState.INTENT
        assert session.intent == "book"

    def test_request_only_after_verification_moves_to_intent(self):
        """Already verified; a later message that states intent → INTENT."""
        session = SessionState()
        session.state = AgentState.VERIFIED
        session.verified_patient_id = 1

        _maybe_transition_from_user_input(
            session, "I'd like to schedule an appointment please."
        )

        assert session.state == AgentState.INTENT
        assert session.intent == "book"

    def test_no_intent_message_stays_verified(self):
        """Verified with no booking/cancel/reschedule cue stays in VERIFIED."""
        session = SessionState()
        session.state = AgentState.VERIFIED
        session.verified_patient_id = 1
        session.messages.append(Message(role="user", content="Thanks, just a moment."))

        _maybe_transition_from_user_input(session, "Thanks, just a moment.")
        _maybe_transition_after_verification(session)

        assert session.state == AgentState.VERIFIED
        assert session.intent is None
