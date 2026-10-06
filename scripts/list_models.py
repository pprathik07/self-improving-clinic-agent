#!/usr/bin/env python
"""List available Gemini models for the configured API key.

Does NOT print the API key. For human use only — do not run from CI/tests.
Usage: uv run python scripts/list_models.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("No GEMINI_API_KEY / GOOGLE_API_KEY set. Copy .env.example to .env first.")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("google-genai not installed. Run: uv sync --all-extras")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print("Available models (key not shown):")
    try:
        for model in client.models.list():
            name = getattr(model, "name", None) or str(model)
            print(f"  {name}")
    except Exception as e:
        print(f"Failed to list models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
