"""Agent runner — the main turn loop for interactive and simulated conversations.

Owns: building the system prompt, calling the LLM, dispatching tool calls,
evaluating state transitions, and logging every step to a JSONL trace.

Does NOT own: clinic business rules (those live in tools/guards), or policy
text (that lives in policy/*.yaml).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from clinic_agent.agent.machine import get_allowed_tools, is_terminal, is_valid_transition
from clinic_agent.agent.policy import load_policy, policy_to_prompt
from clinic_agent.agent.state import AgentState, Message, PendingAction, SessionState
from clinic_agent.clinic.db import frozen_now, init_db, seed_db
from clinic_agent.llm import LLMResponse, ToolCall, chat, pydantic_to_anthropic_tool
from clinic_agent.tools.guards import GuardError
from clinic_agent.tools.impl import TOOL_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions for the LLM (Anthropic format)
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "verify_patient": "Verify a patient's identity using their full name and date of birth.",
    "check_availability": "Check available appointment slots, optionally filtered by specialty, provider, or date range.",
    "book_appointment": "Book an appointment in a specific time slot.",
    "cancel_appointment": "Cancel an existing appointment.",
    "reschedule_appointment": "Reschedule an existing appointment to a new time slot.",
    "escalate_to_human": "Escalate the conversation to a human staff member.",
    "get_patient_appointments": "Retrieve the verified patient's current appointments.",
}


def _build_tool_definitions(allowed: set[str]) -> list[dict]:
    """Build Anthropic tool definitions for the currently allowed tools."""
    definitions = []
    for name in allowed:
        if name in TOOL_REGISTRY:
            _, args_model = TOOL_REGISTRY[name]
            desc = _TOOL_DESCRIPTIONS.get(name, name)
            definitions.append(pydantic_to_anthropic_tool(name, desc, args_model))
    return definitions


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(policy_text: str, session: SessionState) -> str:
    """Build the full system prompt from policy + current state context."""
    now = frozen_now()
    state_context = (
        f"\n\n## Current State\n"
        f"- State: {session.state.value}\n"
        f"- Patient verified: {'Yes' if session.verified_patient_id else 'No'}\n"
        f"- Intent: {session.intent or 'Not yet determined'}\n"
        f"- Confirmed: {session.confirmed}\n"
        f"- Today's date: {now.strftime('%A, %B %d, %Y')}\n"
        f"- Current time: {now.strftime('%I:%M %p')}\n"
    )

    return (
        "You are a clinic scheduling assistant. Help patients book, reschedule, "
        "and cancel appointments. Follow these rules strictly:\n\n"
        f"{policy_text}"
        f"{state_context}"
        "\n\nIMPORTANT: You can only use the tools provided. Do not make up information. "
        "If a tool is not available in your current state, guide the patient through "
        "the proper steps first."
    )


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------

class TraceLogger:
    """Logs every tool call and message to a JSONL file."""

    def __init__(self, trace_path: Path | None = None):
        self._entries: list[dict[str, Any]] = []
        self._path = trace_path

    def log(self, entry: dict[str, Any]) -> None:
        """Append a trace entry."""
        entry["timestamp"] = datetime.now().isoformat()
        self._entries.append(entry)
        if self._path:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self._entries


# ---------------------------------------------------------------------------
# Core turn loop
# ---------------------------------------------------------------------------

def run_turn(
    session: SessionState,
    conn: Any,
    policy_text: str,
    trace: TraceLogger,
) -> str:
    """Run one agent turn: LLM call -> tool dispatch -> state transition.

    Returns the agent's text response to show the patient.
    """
    # Build LLM inputs
    allowed = get_allowed_tools(session.state)
    tool_defs = _build_tool_definitions(allowed)
    system = _build_system_prompt(policy_text, session)

    # Convert session messages to Anthropic format
    api_messages = _to_api_messages(session.messages)

    # Call LLM
    response = chat(
        messages=api_messages,
        system=system,
        tools=tool_defs if tool_defs else None,
        model_key="AGENT_MODEL",
        temperature=0.2,
    )

    trace.log({
        "event": "llm_response",
        "content": response.content,
        "tool_calls": [tc.model_dump() for tc in response.tool_calls],
        "usage": response.usage,
        "session_state": session.snapshot(),
    })

    # Handle tool calls
    if response.tool_calls:
        return _handle_tool_calls(response, session, conn, policy_text, trace)

    # Text-only response — add to history and return
    if response.content:
        session.messages.append(Message(role="assistant", content=response.content))

    return response.content


def _handle_tool_calls(
    response: LLMResponse,
    session: SessionState,
    conn: Any,
    policy_text: str,
    trace: TraceLogger,
) -> str:
    """Process tool calls from the LLM response."""
    results = []
    allowed = get_allowed_tools(session.state)

    for tc in response.tool_calls:
        # Check if tool is allowed in current state
        if tc.name not in allowed:
            result_text = f"Tool '{tc.name}' is not available right now. Please follow the current step."
            trace.log({
                "event": "tool_blocked",
                "tool": tc.name,
                "reason": f"Not allowed in state {session.state.value}",
                "session_state": session.snapshot(),
            })
        else:
            result_text = _execute_tool(tc, session, conn, trace)

        results.append({
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": result_text,
        })

    # Add the assistant message with tool use to history
    # Build the content blocks for the assistant message
    assistant_content_parts = []
    if response.content:
        assistant_content_parts.append(response.content)
    for tc in response.tool_calls:
        assistant_content_parts.append(f"[Tool call: {tc.name}({json.dumps(tc.arguments)})]")

    session.messages.append(Message(
        role="assistant",
        content=response.content or "",
        tool_name=response.tool_calls[0].name if response.tool_calls else None,
    ))

    # Add tool results to history
    for i, result in enumerate(results):
        session.messages.append(Message(
            role="tool_result",
            content=result["content"],
            tool_call_id=result["tool_use_id"],
            tool_name=response.tool_calls[i].name,
        ))

    # After processing tool results, call the LLM again to get a natural language response
    api_messages = _to_api_messages(session.messages)
    system = _build_system_prompt(policy_text, session)
    allowed_now = get_allowed_tools(session.state)
    tool_defs = _build_tool_definitions(allowed_now)

    followup = chat(
        messages=api_messages,
        system=system,
        tools=tool_defs if tool_defs else None,
        model_key="AGENT_MODEL",
        temperature=0.2,
    )

    trace.log({
        "event": "llm_followup",
        "content": followup.content,
        "tool_calls": [tc.model_dump() for tc in followup.tool_calls],
        "usage": followup.usage,
        "session_state": session.snapshot(),
    })

    # If the followup also has tool calls, recurse (bounded by turn limit)
    if followup.tool_calls:
        # Add followup content to messages and process its tool calls
        if followup.content:
            session.messages.append(Message(role="assistant", content=followup.content))
        return _handle_tool_calls(followup, session, conn, policy_text, trace)

    if followup.content:
        session.messages.append(Message(role="assistant", content=followup.content))

    return followup.content


def _execute_tool(tc: ToolCall, session: SessionState, conn: Any, trace: TraceLogger) -> str:
    """Execute a single tool call with guards."""
    tool_fn, args_model = TOOL_REGISTRY[tc.name]

    try:
        # Validate args with Pydantic
        args = args_model(**tc.arguments)
        # Snapshot BEFORE execution (write tools reset confirmed/state)
        pre_state = session.snapshot()
        # Execute with guards (guards are inside the tool functions)
        result = tool_fn(args, session, conn)
        result_text = result.model_dump_json()

        trace.log({
            "event": "tool_call",
            "tool": tc.name,
            "args": tc.arguments,
            "result": result.model_dump(),
            "session_state": pre_state,
        })

        return result_text

    except GuardError as e:
        trace.log({
            "event": "tool_guard_blocked",
            "tool": tc.name,
            "args": tc.arguments,
            "guard_error": e.internal_reason,
            "session_state": session.snapshot(),
        })
        return json.dumps({"error": e.patient_message})

    except Exception as e:
        logger.exception("Tool execution error: %s", e)
        trace.log({
            "event": "tool_error",
            "tool": tc.name,
            "args": tc.arguments,
            "error": str(e),
            "session_state": session.snapshot(),
        })
        return json.dumps({
            "error": "Something went wrong on our end. Let me connect you with a staff member."
        })


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------

def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert session messages to Anthropic API format.

    Handles the Anthropic requirement that tool_result messages follow
    assistant messages with tool_use blocks.
    """
    api_msgs: list[dict[str, Any]] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "user":
            api_msgs.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            # Check if this assistant message has associated tool calls
            # by looking ahead for tool_result messages
            content_blocks: list[dict] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})

            # Look ahead for tool results that belong to this assistant turn
            j = i + 1
            tool_results = []
            while j < len(messages) and messages[j].role == "tool_result":
                tool_msg = messages[j]
                # Add a tool_use block to the assistant message
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_msg.tool_call_id,
                    "name": tool_msg.tool_name or "unknown",
                    "input": {},  # We don't store the full input in history
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_msg.tool_call_id,
                    "content": tool_msg.content,
                })
                j += 1

            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})

            api_msgs.append({"role": "assistant", "content": content_blocks})

            # Add tool results as a user message (Anthropic format)
            if tool_results:
                api_msgs.append({"role": "user", "content": tool_results})
                i = j
                continue

        elif msg.role == "tool_result":
            # Orphaned tool result (shouldn't happen, but handle gracefully)
            api_msgs.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id or "unknown", "content": msg.content}],
            })

        i += 1

    return api_msgs


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(policy_path: str | None = None) -> None:
    """Run the agent in interactive mode (stdin/stdout)."""
    from dotenv import load_dotenv
    load_dotenv()

    policy_path = policy_path or os.environ.get("POLICY_PATH", "policy/policy_v1.yaml")
    policy = load_policy(policy_path)
    policy_text = policy_to_prompt(policy)

    conn = init_db(in_memory=False)
    seed_db(conn, "default")

    session = SessionState()
    trace = TraceLogger()

    print("\nClinic Scheduling Agent")
    print("=" * 40)
    print("Type 'quit' to exit, 'state' to see current state.\n")

    # Initial greeting
    session.messages.append(Message(role="user", content="Hello"))
    session.state = AgentState.IDENTIFY  # Move to IDENTIFY immediately

    greeting = run_turn(session, conn, policy_text, trace)
    print(f"Agent: {greeting}\n")

    while not is_terminal(session.state):
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "state":
            print(f"  State: {session.state.value}")
            print(f"  Verified: {session.verified_patient_id}")
            print(f"  Intent: {session.intent}")
            print(f"  Confirmed: {session.confirmed}")
            continue

        session.messages.append(Message(role="user", content=user_input))

        # Detect state transitions from user input
        _maybe_transition_from_user_input(session, user_input)

        response_text = run_turn(session, conn, policy_text, trace)
        print(f"Agent: {response_text}\n")

    if is_terminal(session.state):
        print(f"\n[Conversation ended — state: {session.state.value}]")


def _maybe_transition_from_user_input(session: SessionState, user_input: str) -> None:
    """Evaluate if user input should trigger a state transition.

    This handles the conversational cues that don't involve tool calls,
    like the patient stating their intent or confirming an action.
    """
    lower = user_input.lower()

    # From VERIFIED -> INTENT when patient states what they want
    if session.state == AgentState.VERIFIED:
        if any(word in lower for word in ["book", "schedule", "appointment", "see", "visit",
                                           "cancel", "reschedule", "change", "move"]):
            session.state = AgentState.INTENT
            if any(w in lower for w in ["cancel"]):
                session.intent = "cancel"
            elif any(w in lower for w in ["reschedule", "change", "move"]):
                session.intent = "reschedule"
            else:
                session.intent = "book"

    # From GREETING -> IDENTIFY (patient starts talking)
    elif session.state == AgentState.GREETING:
        session.state = AgentState.IDENTIFY

    # Confirmation detection in CONFIRM state
    elif session.state == AgentState.CONFIRM:
        if any(word in lower for word in ["yes", "yeah", "yep", "sure", "go ahead",
                                           "confirm", "please do", "book it", "do it",
                                           "cancel it", "please cancel"]):
            session.confirmed = True
            session.state = AgentState.EXECUTE

    # SLOT_SELECTION → CONFIRM when patient picks a slot
    elif session.state == AgentState.SLOT_SELECTION:
        if any(word in lower for word in ["first", "that one", "book", "go ahead",
                                           "looks good", "please", "yes"]):
            session.state = AgentState.CONFIRM


# ---------------------------------------------------------------------------
# Simulated mode (for evals)
# ---------------------------------------------------------------------------

def run_simulated(
    conn: Any,
    opening: str,
    simulator_fn: Any,
    policy_path: str = "policy/policy_v1.yaml",
    max_turns: int = 14,
    trace_path: Path | None = None,
) -> tuple[SessionState, TraceLogger]:
    """Run the agent with a simulated patient for evaluation.

    Args:
        conn: Database connection (already seeded).
        opening: The patient's opening message.
        simulator_fn: Async function that generates patient responses.
        policy_path: Path to the policy YAML file.
        max_turns: Maximum number of turns before force-stopping.
        trace_path: Optional path to write the JSONL trace.

    Returns:
        (final_session_state, trace_logger)
    """
    policy = load_policy(policy_path)
    policy_text = policy_to_prompt(policy)

    session = SessionState()
    trace = TraceLogger(trace_path)

    # Start the conversation
    session.state = AgentState.IDENTIFY
    session.messages.append(Message(role="user", content=opening))

    trace.log({"event": "conversation_start", "opening": opening})

    turn = 0
    while not is_terminal(session.state) and turn < max_turns:
        turn += 1

        # Agent responds
        agent_response = run_turn(session, conn, policy_text, trace)

        trace.log({
            "event": "agent_turn",
            "turn": turn,
            "response": agent_response,
            "session_state": session.snapshot(),
        })

        if is_terminal(session.state):
            break

        # Patient simulator responds
        patient_response = simulator_fn(agent_response, turn)

        trace.log({
            "event": "patient_turn",
            "turn": turn,
            "response": patient_response,
        })

        session.messages.append(Message(role="user", content=patient_response))
        _maybe_transition_from_user_input(session, patient_response)

    if turn >= max_turns and not is_terminal(session.state):
        trace.log({"event": "max_turns_reached", "turn": turn})

    trace.log({
        "event": "conversation_end",
        "final_state": session.state.value,
        "total_turns": turn,
    })

    return session, trace
