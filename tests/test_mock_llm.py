"""Tests for mock LLM behavior (harness bug regression tests).

These tests ensure the mock LLM correctly handles specific scenarios
without special-casing to make tests pass.
"""

import os

import pytest

from clinic_agent.llm import _mock_agent, _mock_simulator, LLMResponse


class TestMockSimulator:
    """Test mock simulator handles persona-specific intents after verification."""

    def test_simulator_wrong_dob_then_retry_book_cardiology(self):
        """After wrong DOB correction, simulator should book cardiology (not general practice)."""
        persona = (
            "You are Asha Rao. On your first verification attempt, you misremember your "
            "date of birth and say 'May 15, 1990' (the correct one is May 14, 1990). "
            "When the agent says verification failed, say 'Oh wait, I think it's actually "
            "May 14, 1990.' Then proceed to book a cardiology appointment for next week. "
            "Be cooperative after the correction."
        )

        # First, simulate the failed verification attempt
        agent_msg = "I wasn't able to verify those details. Could you please try again?"
        messages = [{"role": "user", "content": f"[Scheduling Assistant]: {agent_msg}"}]
        response = _mock_simulator(messages, system=persona)
        assert "may 14" in response.content.lower(), "Should correct DOB"

        # Then, after successful verification
        agent_msg = "Great, I've verified your identity. How can I help you today?"
        messages = [{"role": "user", "content": f"[Scheduling Assistant]: {agent_msg}"}]
        response = _mock_simulator(messages, system=persona)

        # Should request cardiology, not general practice
        assert "cardiolog" in response.content.lower(), \
            f"Expected cardiology request, got: {response.content}"

    def test_simulator_handles_generic_verified_state(self):
        """Generic 'how can I help' should still work for any persona."""
        persona = "You are Asha Rao, born May 14, 1990."
        agent_msg = "Great, I've verified your identity. How can I help you today?"
        messages = [{"role": "user", "content": f"[Scheduling Assistant]: {agent_msg}"}]

        response = _mock_simulator(messages, system=persona)

        # Should respond with booking intent
        assert "book" in response.content.lower(), \
            f"Expected booking intent, got: {response.content}"


class TestMockAgentIntent:
    """Test mock agent respects session intent when making tool calls."""

    def test_mock_agent_respects_cancel_intent_in_execute(self):
        """In EXECUTE state with cancel intent, should call cancel_appointment not book_appointment."""
        system = (
            "You are a clinic scheduling assistant.\n\n"
            "## Current State\n"
            "- State: EXECUTE\n"
            "- Intent: cancel\n"
            "- Confirmed: True\n"
        )

        tools = [
            {"name": "book_appointment", "input_schema": {}},
            {"name": "cancel_appointment", "input_schema": {}},
        ]

        messages = [{"role": "user", "content": "Yes, please cancel it."}]

        response = _mock_agent(messages, system, tools)

        # Should call cancel_appointment, not book_appointment
        assert response.tool_calls, "Expected tool call"
        assert response.tool_calls[0].name == "cancel_appointment", \
            f"Expected cancel_appointment, got: {response.tool_calls[0].name}"

    def test_mock_agent_books_when_intent_is_book(self):
        """In EXECUTE state with book intent, should call book_appointment."""
        system = (
            "You are a clinic scheduling assistant.\n\n"
            "## Current State\n"
            "- State: EXECUTE\n"
            "- Intent: book\n"
            "- Confirmed: True\n"
        )

        tools = [
            {"name": "book_appointment", "input_schema": {}},
            {"name": "cancel_appointment", "input_schema": {}},
        ]

        messages = [{"role": "user", "content": "Yes, please book that for me."}]

        response = _mock_agent(messages, system, tools)

        # Should call book_appointment
        assert response.tool_calls, "Expected tool call"
        assert response.tool_calls[0].name == "book_appointment", \
            f"Expected book_appointment, got: {response.tool_calls[0].name}"
