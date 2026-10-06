"""LLM Judge — binary pass/fail evaluation of conversation transcripts.

Uses a separate model (JUDGE_MODEL) from the agent to avoid self-evaluation bias.
Each rubric item is scored independently as pass/fail with a brief reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from clinic_agent.llm import chat

logger = logging.getLogger(__name__)


def judge_transcript(
    transcript: str,
    rubric_items: list[str],
) -> list[dict[str, Any]]:
    """Score a conversation transcript against rubric items.

    Args:
        transcript: The full conversation transcript text.
        rubric_items: List of rubric items to evaluate (from scenario YAML).

    Returns:
        List of dicts with {item, passed, reasoning} per rubric item.
    """
    if not rubric_items:
        return []

    rubric_text = "\n".join(f"- {item}" for item in rubric_items)

    system_prompt = (
        "You are a strict evaluator for a clinic scheduling AI assistant. "
        "You will be given a conversation transcript and a list of rubric items. "
        "For each rubric item, determine if the assistant's behavior PASSED or FAILED.\n\n"
        "Rules:\n"
        "- Be strict. If there is any doubt, mark as FAIL.\n"
        "- Judge based on the actual transcript, not what you think should have happened.\n"
        "- A pass means the assistant clearly demonstrated the required behavior.\n"
        "- A fail means the assistant did not demonstrate it, or violated it.\n\n"
        "Respond with a JSON array. Each element must have:\n"
        '- "item": the rubric item text (exact)\n'
        '- "passed": true or false\n'
        '- "reasoning": one sentence explaining your judgment\n\n'
        "Respond ONLY with the JSON array, no other text."
    )

    message = (
        f"## Conversation Transcript\n\n{transcript}\n\n"
        f"## Rubric Items to Evaluate\n\n{rubric_text}\n\n"
        f"Evaluate each rubric item. Respond with a JSON array."
    )

    try:
        response = chat(
            messages=[{"role": "user", "content": message}],
            system=system_prompt,
            model_key="JUDGE_MODEL",
            temperature=0.0,
            max_tokens=1024,
        )

        return _parse_judge_response(response.content, rubric_items)

    except Exception as e:
        logger.error("Judge error: %s", e)
        # On failure, mark all items as failed
        return [
            {"item": item, "passed": False, "reasoning": f"Judge error: {e}"}
            for item in rubric_items
        ]


def _parse_judge_response(
    response_text: str, rubric_items: list[str]
) -> list[dict[str, Any]]:
    """Parse the judge's JSON response, handling common format issues."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        results = json.loads(text)
        if isinstance(results, list):
            # Validate structure
            validated = []
            for r in results:
                if isinstance(r, dict) and "item" in r and "passed" in r:
                    validated.append({
                        "item": r["item"],
                        "passed": bool(r["passed"]),
                        "reasoning": r.get("reasoning", ""),
                    })
            # Ensure all rubric items are covered
            covered_items = {v["item"] for v in validated}
            for item in rubric_items:
                if item not in covered_items:
                    validated.append({
                        "item": item,
                        "passed": False,
                        "reasoning": "Not evaluated by judge",
                    })
            return validated
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract individual item results
    logger.warning("Could not parse judge response as JSON, falling back to all-fail")
    return [
        {"item": item, "passed": False, "reasoning": "Could not parse judge response"}
        for item in rubric_items
    ]


def format_transcript(messages: list[dict]) -> str:
    """Format conversation messages into a readable transcript for the judge."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Patient: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "tool_result":
            tool_name = msg.get("tool_name", "unknown")
            lines.append(f"[Tool Result ({tool_name})]: {content}")
    return "\n\n".join(lines)
