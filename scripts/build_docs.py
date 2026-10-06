"""Replace README results block from before/after results.json.

Markers:
  <!-- RESULTS:START -->
  <!-- RESULTS:END -->

Usage:
  uv run python scripts/build_docs.py
  uv run python scripts/build_docs.py path/to/before/results.json path/to/after/results.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
PENDING = "**PENDING REAL RUN** — replace this block by running `uv run python scripts/build_docs.py <before/results.json> [after/results.json]` on real results (never mock)."


def replace_results_block(readme_text: str, body: str) -> str:
    if START not in readme_text or END not in readme_text:
        raise ValueError("README missing RESULTS markers")
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END),
        re.DOTALL,
    )
    replacement = f"{START}\n{body.rstrip()}\n{END}"
    return pattern.sub(replacement, readme_text, count=1)


def build_body(before_path: Path | None, after_path: Path | None) -> str:
    if before_path is None:
        return PENDING
    sys.path.insert(0, str(ROOT))
    from clinic_agent.evals.report import load_results, render_report

    before = load_results(before_path)
    after = load_results(after_path) if after_path else None
    # refuse mock via render_report default
    return render_report(before, after, allow_mock=False)


def update_readme(
    readme_path: Path,
    before_path: Path | None = None,
    after_path: Path | None = None,
) -> str:
    text = readme_path.read_text(encoding="utf-8")
    body = build_body(before_path, after_path)
    new = replace_results_block(text, body)
    readme_path.write_text(new, encoding="utf-8")
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", nargs="?", type=Path, default=None)
    parser.add_argument("after", nargs="?", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args(argv)
    try:
        update_readme(args.readme, args.before, args.after)
    except Exception as e:
        print(f"FAIL: {e}")
        return 1
    print(f"Updated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
