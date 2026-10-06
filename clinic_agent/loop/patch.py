"""Patch application — applies structured patches to policy YAML files."""

from __future__ import annotations

import copy
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from clinic_agent.agent.policy import load_policy, save_policy


class FailureCategory(str, Enum):
    """Categories of failures."""
    IDENTITY_LEAK = "identity_leak"
    MISSING_CONFIRMATION = "missing_confirmation"
    EMERGENCY_MISSED = "emergency_missed"
    INJECTION = "injection"
    AMBIGUITY = "ambiguity"
    HALLUCINATED_SUCCESS = "hallucinated_success"
    TONE = "tone"
    NEEDS_CODE_FIX = "needs_code_fix"
    OTHER = "other"


class Evidence(BaseModel):
    """A specific observation from a failing scenario."""
    scenario_id: str
    turn: int | None = None
    observation: str


class PolicyChange(BaseModel):
    """A single change to the policy file."""
    policy_section: str
    op: str  # "add" or "edit"
    text: str

    def __init__(self, **data):
        super().__init__(**data)
        if self.op not in ("add", "edit"):
            raise ValueError(f"op must be 'add' or 'edit', got '{self.op}'")


class Patch(BaseModel):
    """A structured policy patch proposed by the reflector."""
    failure_category: FailureCategory
    evidence: list[Evidence] = Field(min_length=1)
    root_cause: str = Field(max_length=200)
    changes: list[PolicyChange]  # max 3 changes validated in apply_patch
    expected_to_fix: list[str] = Field(min_length=1)
    regression_risk: str = Field(max_length=200)

logger = logging.getLogger(__name__)


# Protected sections that cannot be edited (only additions allowed)
_PROTECTED_SECTIONS = {
    "identity", "confirmation", "emergency", "injection"
}


class PatchRejectedReason(str, Enum):
    """Reason codes for patch rejection."""
    UNKNOWN_SECTION = "UNKNOWN_SECTION"
    SCENARIO_LEAK = "SCENARIO_LEAK"
    TOO_LONG = "TOO_LONG"
    TOO_MANY_CHANGES = "TOO_MANY_CHANGES"
    EMPTY_CHANGES = "EMPTY_CHANGES"
    CODE_FIX_WITH_CHANGES = "CODE_FIX_WITH_CHANGES"
    PROTECTED_SECTION_EDIT = "PROTECTED_SECTION_EDIT"
    V1_OVERWRITE = "V1_OVERWRITE"
    V_EXISTS_OVERWRITE = "V_EXISTS_OVERWRITE"


class PatchRejected(Exception):
    """Raised when a patch is rejected for safety reasons."""
    def __init__(self, reason: PatchRejectedReason, message: str):
        self.reason = reason
        self.message = message
        super().__init__(message)


def _load_all_scenario_data() -> tuple[set[str], set[str], set[str]]:
    """Load all scenario IDs, patient names, and quoted text from scenario files.

    Returns:
        (scenario_ids, patient_names, quoted_lines) - sets of strings for anti-overfitting
    """
    scenarios_dir = Path(__file__).resolve().parents[1] / "evals" / "scenarios"
    scenario_ids = set()
    patient_names = set()
    quoted_lines = set()

    for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Add scenario ID
        scenario_id = data.get("id", "")
        if scenario_id:
            scenario_ids.add(scenario_id.lower())

        # Extract patient names from persona (heuristic: look for common name patterns)
        persona = data.get("patient_persona", "")
        # Pattern: "You are NAME" or "My name is NAME" or "I am NAME"
        # Extract from original case to distinguish names from phrases
        name_patterns = re.findall(r"(?:You are|My name is|I am)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", persona)
        for name in name_patterns:
            patient_names.add(name.lower())

        # Extract quoted lines from persona and opening
        # Look for text in quotes that might be copied directly
        persona_text = data.get("patient_persona", "")
        opening_text = data.get("opening", "")
        for text in [persona_text, opening_text]:
            # Extract quoted strings (double quotes only)
            quoted = re.findall(r'"([^"]{10,})"', text)
            for q in quoted:
                if len(q) >= 10:  # Only significant quotes
                    quoted_lines.add(q.lower())

    return scenario_ids, patient_names, quoted_lines


def apply_patch(policy: dict[str, Any], patch: Patch) -> dict[str, Any] | Patch:
    """Apply a structured patch to a policy dict with validation.

    Args:
        policy: The current policy dict (section_id -> {text: ...}).
        patch: The validated patch to apply.

    Returns:
        A new policy dict with the patch applied, or the original patch unchanged
        if it's a needs_code_fix human TODO.

    Raises:
        PatchRejected: If patch violates safety constraints.
    """
    # Special case: needs_code_fix category
    if patch.failure_category == FailureCategory.NEEDS_CODE_FIX:
        if patch.changes:
            raise PatchRejected(
                PatchRejectedReason.CODE_FIX_WITH_CHANGES,
                "failure_category=needs_code_fix but changes list is non-empty. "
                "Code fixes require human review and cannot be applied via policy patches."
            )
        # Return the patch as-is for human TODO handling
        return patch

    # Validation 1: Changes must be non-empty for non-code-fix categories
    if not patch.changes:
        raise PatchRejected(
            PatchRejectedReason.EMPTY_CHANGES,
            "Patch has empty changes list. Non-code-fix patches must specify changes."
        )

    # Validation 2: Max 3 changes (Pydantic enforces this, but double-check)
    if len(patch.changes) > 3:
        raise PatchRejected(
            PatchRejectedReason.TOO_MANY_CHANGES,
            f"Patch has {len(patch.changes)} changes (max 3). "
            f"Split into smaller, targeted patches."
        )

    # Load anti-overfitting data
    scenario_ids, patient_names, quoted_lines = _load_all_scenario_data()
    known_sections = set(policy.keys())

    for change in patch.changes:
        # Validation 3: Unknown section
        if change.policy_section not in known_sections:
            raise PatchRejected(
                PatchRejectedReason.UNKNOWN_SECTION,
                f"Cannot edit unknown section: {change.policy_section}. "
                f"Known sections: {sorted(known_sections)}. "
                f"Adding new sections requires human approval."
            )

        # Validation 4: Protected section edit
        if change.policy_section in _PROTECTED_SECTIONS and change.op == "edit":
            raise PatchRejected(
                PatchRejectedReason.PROTECTED_SECTION_EDIT,
                f"Cannot edit protected section '{change.policy_section}'. "
                f"Protected sections: {sorted(_PROTECTED_SECTIONS)}. "
                f"Use op='add' to append new guidance."
            )

        # Validation 5: Change text too long
        if len(change.text) > 600:
            raise PatchRejected(
                PatchRejectedReason.TOO_LONG,
                f"Change text too long: {len(change.text)} chars (max 600). "
                f"Split into smaller changes."
            )

        # Validation 6: Anti-overfitting - scenario IDs
        change_lower = change.text.lower()
        for scenario_id in scenario_ids:
            if scenario_id in change_lower:
                raise PatchRejected(
                    PatchRejectedReason.SCENARIO_LEAK,
                    f"Change text contains scenario ID '{scenario_id}'. "
                    f"This risks overfitting to specific scenarios."
                )

        # Validation 7: Anti-overfitting - patient names
        for patient_name in patient_names:
            if patient_name in change_lower:
                raise PatchRejected(
                    PatchRejectedReason.SCENARIO_LEAK,
                    f"Change text contains patient name '{patient_name}'. "
                    f"This risks overfitting to specific test data."
                )

        # Validation 8: Anti-overfitting - quoted lines from persona/opening
        for quoted in quoted_lines:
            if quoted in change_lower:
                raise PatchRejected(
                    PatchRejectedReason.SCENARIO_LEAK,
                    f"Change text contains quoted text from scenario: '{quoted[:50]}...'. "
                    f"This risks overfitting to specific test data."
                )

    # Apply changes (do not mutate input)
    new_policy = copy.deepcopy(policy)

    for change in patch.changes:
        if change.op == "edit":
            new_policy[change.policy_section] = {"text": change.text}

        elif change.op == "add":
            # Append: keep existing text, add new text
            if change.policy_section in new_policy:
                existing_text = new_policy[change.policy_section].get("text", "")
                new_policy[change.policy_section] = {"text": f"{existing_text.rstrip()}\n{change.text}"}
            else:
                new_policy[change.policy_section] = {"text": change.text}

    return new_policy


def write_new_policy(
    policy: dict[str, Any],
    version: int,
    policy_dir: Path = Path("policy")
) -> Path:
    """Write a policy dict to policy/policy_vN.yaml without touching v1.

    Args:
        policy: The policy dict to save.
        version: The policy version number.
        policy_dir: Directory containing policy files.

    Returns:
        The path to the written file.

    Raises:
        PatchRejected: If trying to overwrite policy_v1.yaml or an existing vN file.
    """
    if version == 1:
        raise PatchRejected(
            PatchRejectedReason.V1_OVERWRITE,
            "Cannot overwrite policy_v1.yaml. It is the immutable baseline. "
            "New versions must start at v2."
        )

    v1_path = policy_dir / "policy_v1.yaml"
    if not v1_path.exists():
        raise ValueError(f"policy_v1.yaml not found at {v1_path}")

    path = policy_dir / f"policy_v{version}.yaml"
    if path.exists():
        raise PatchRejected(
            PatchRejectedReason.V_EXISTS_OVERWRITE,
            f"Cannot overwrite existing policy file {path.name}. "
            f"Use a new version number."
        )

    # Read v1 bytes to verify unchanged later
    v1_bytes_before = v1_path.read_bytes()

    save_policy(policy, path)

    # Verify v1 unchanged
    v1_bytes_after = v1_path.read_bytes()
    if v1_bytes_before != v1_bytes_after:
        raise RuntimeError(
            f"policy_v1.yaml was modified during write_new_policy! "
            f"This should never happen."
        )

    return path
