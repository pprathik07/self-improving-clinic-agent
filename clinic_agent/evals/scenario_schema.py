"""Scenario schema — Pydantic models for eval scenario YAML files."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StateExpectations(BaseModel):
    """Expected database/session state after the conversation."""
    appointments_created: int | None = None
    appointments_cancelled: int | None = None
    no_write_before_verified: bool = True
    no_double_booking: bool = True
    patient_verified: bool | None = None
    escalated: bool | None = None


class TraceExpectations(BaseModel):
    """Expected tool-call trace patterns."""
    must_call: list[str] = Field(default_factory=list)
    forbidden_calls: list[str] = Field(default_factory=list)
    order: list[str] = Field(default_factory=list)  # these tools must appear in this order
    confirmation_before_write: bool = True


class ExpectedOutcome(BaseModel):
    """All expected outcomes for a scenario."""
    state: StateExpectations = Field(default_factory=StateExpectations)
    trace: TraceExpectations = Field(default_factory=TraceExpectations)
    judge: list[str] = Field(default_factory=list)  # rubric items for the LLM judge


class Scenario(BaseModel):
    """A single eval scenario loaded from YAML."""
    id: str
    split: Literal["train", "heldout"]
    tags: list[str] = Field(default_factory=list)
    seed_db: str = "default"
    patient_persona: str
    opening: str
    max_turns: int = 14
    expect: ExpectedOutcome = Field(default_factory=ExpectedOutcome)
