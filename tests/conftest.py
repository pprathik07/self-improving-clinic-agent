"""Autouse fixtures for hermetic tests.

Sets MOCK_LLM=1 in every test, strips provider/debug env that would divert
chat() onto OpenRouter or leak CLINIC_DEBUG into assertions, and prevents
constructing a real google.genai Client unless a test explicitly patches it.
"""

from __future__ import annotations

import os

import pytest

# Vars that must not leak from the developer's shell into hermetic tests.
# Tests that need them set them via monkeypatch.
_PROVIDER_ENV_STRIP = (
    "OPEN_ROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "LLM_PROVIDER",
    "CLINIC_DEBUG",
)


@pytest.fixture(autouse=True)
def _force_mock_llm(monkeypatch):
    """Set MOCK_LLM=1 for every test.  Tests that need a real client must
    explicitly monkeypatch MOCK_LLM=0 and supply their own fake client."""
    monkeypatch.setenv("MOCK_LLM", "1")
    for key in _PROVIDER_ENV_STRIP:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _block_real_genai_client(monkeypatch):
    """Prevent any test from constructing a real google.genai.Client
    unless it has been explicitly patched out with a fake.

    This catches tests that accidentally call the live API."""
    try:
        from google.genai.client import Client as _Client  # noqa: F401
    except ImportError:
        # google-genai not installed — nothing to block
        return

    def _guard(self, *args, **kwargs):
        raise AssertionError(
            "Test attempted to create a real google.genai.Client. "
            "Set MOCK_LLM=1 or patch the client in your test."
        )

    monkeypatch.setattr("google.genai.client.Client.__init__", _guard)
