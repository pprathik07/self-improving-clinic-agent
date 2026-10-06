"""Policy loader — reads versioned YAML policy files.

The policy file is the only thing the improvement loop edits.
Sections have stable IDs so patches can target them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load a policy YAML file and return its sections as a dict.

    Each top-level key is a section ID (e.g. 'identity', 'confirmation').
    Each value has at minimum a 'text' field.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Policy file must be a YAML mapping, got {type(data).__name__}")

    return data


def policy_to_prompt(policy: dict[str, Any]) -> str:
    """Convert a policy dict into a system prompt string.

    Each section becomes a numbered block with its ID as a header.
    """
    parts = []
    for i, (section_id, section) in enumerate(policy.items(), 1):
        if isinstance(section, dict):
            text = section.get("text", "")
        else:
            text = str(section)
        parts.append(f"## Rule {i}: {section_id.replace('_', ' ').title()}\n{text}")

    return "\n\n".join(parts)


def policy_hash(policy: dict[str, Any]) -> str:
    """Return a short hash of the policy content for tracking versions."""
    raw = yaml.dump(policy, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def save_policy(policy: dict[str, Any], path: str | Path) -> None:
    """Write a policy dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(policy, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
