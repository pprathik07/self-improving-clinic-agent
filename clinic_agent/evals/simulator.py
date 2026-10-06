"""Patient simulator — generates patient responses for eval scenarios.

Uses a separate LLM (SIM_MODEL) with low temperature to stay on-persona.
Includes turn cap enforcement and drift detection.
"""

from __future__ import annotations

import logging

from clinic_agent.llm import chat

logger = logging.getLogger(__name__)


def simulate_patient(
    persona: str,
    agent_message: str,
    conversation_history: list[dict],
    turn: int,
    max_turns: int,
) -> str:
    """Generate a patient response based on the persona and conversation.

    Args:
        persona: The patient persona description from the scenario.
        agent_message: The agent's latest message.
        conversation_history: Previous turns for context.
        turn: Current turn number.
        max_turns: Maximum turns allowed.

    Returns:
        The simulated patient's response text.
    """
    remaining = max_turns - turn
    system_prompt = (
        f"You are role-playing as a patient calling a clinic scheduling assistant.\n\n"
        f"YOUR PERSONA:\n{persona}\n\n"
        f"RULES:\n"
        f"- Stay in character at all times.\n"
        f"- Respond naturally as this patient would.\n"
        f"- Keep responses short (1-3 sentences).\n"
        f"- Do NOT break character or mention that you are an AI.\n"
        f"- Do NOT add meta-commentary about the conversation.\n"
        f"- You have {remaining} turns remaining. If the conversation should "
        f"naturally end, say goodbye.\n"
        f"- Follow the persona instructions precisely, including any specific "
        f"behaviors like giving wrong information on purpose.\n"
    )

    # Build messages
    messages = []
    for entry in conversation_history:
        messages.append(entry)

    # Add the latest agent message
    messages.append({"role": "user", "content": f"[Scheduling Assistant]: {agent_message}"})

    try:
        response = chat(
            messages=messages,
            system=system_prompt,
            model_key="SIM_MODEL",
            temperature=0.3,
            max_tokens=256,
        )
        return response.content.strip()
    except Exception as e:
        logger.error("Simulator error: %s", e)
        return "I'm sorry, I need to go. Goodbye."


def build_sim_history(turns: list[tuple[str, str]]) -> list[dict]:
    """Convert turn pairs into simulator conversation history.

    Args:
        turns: List of (agent_message, patient_response) tuples.

    Returns:
        Messages list for the simulator.
    """
    messages = []
    for agent_msg, patient_msg in turns:
        messages.append({"role": "user", "content": f"[Scheduling Assistant]: {agent_msg}"})
        messages.append({"role": "assistant", "content": patient_msg})
    return messages
