"""Autouse fixtures for hermetic tests.

Sets MOCK_LLM=1 in every test and prevents constructing a real google.genai
Client unless a test explicitly patches it with a fake.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _force_mock_llm(monkeypatch):
    """Set MOCK_LLM=1 for every test.  Tests that need a real client must
    explicitly monkeypatch MOCK_LLM=0 and supply their own fake client."""
    monkeypatch.setenv("MOCK_LLM", "1")


@pytest.fixture(autouse=True)
def _block_real_genai_client(monkeypatch):
    """Prevent any test from constructing a real google.genai.Client
    unless it has been explicitly patched out with a fake.

    This catches tests that accidentally call the live API."""
    _orig_init = None
    try:
        from google.genai.client import Client as _Client
        _orig_init = _Client.__init__
    except ImportError:
        # google-genai not installed — nothing to block
        return

    def _guard(self, *args, **kwargs):
        raise AssertionError(
            "Test attempted to create a real google.genai.Client. "
            "Set MOCK_LLM=1 or patch the client in your test."
        )

    monkeypatch.setattr("google.genai.client.Client.__init__", _guard)
