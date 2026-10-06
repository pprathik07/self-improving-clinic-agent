"""Freeze sha256 hashes of scenarios, scorers, and policy_v1.

Writes data/frozen_hashes.json. --check compares and prints which changed.

Usage:
  uv run python scripts/freeze_manifest.py
  uv run python scripts/freeze_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "frozen_hashes.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_paths(root: Path | None = None) -> list[Path]:
    root = root or ROOT
    paths: list[Path] = []
    scenarios = root / "clinic_agent" / "evals" / "scenarios"
    if scenarios.is_dir():
        paths.extend(sorted(scenarios.glob("*.yaml")))
    scorers = root / "clinic_agent" / "evals" / "scorers.py"
    if scorers.exists():
        paths.append(scorers)
    policy = root / "policy" / "policy_v1.yaml"
    if policy.exists():
        paths.append(policy)
    return paths


def build_manifest(root: Path | None = None) -> dict[str, str]:
    root = root or ROOT
    out: dict[str, str] = {}
    for p in collect_paths(root):
        rel = p.relative_to(root).as_posix()
        out[rel] = file_sha256(p)
    return out


def write_manifest(path: Path | None = None, root: Path | None = None) -> Path:
    path = path or DEFAULT_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_manifest(root)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_manifest(path: Path | None = None, root: Path | None = None) -> tuple[bool, list[str]]:
    path = path or DEFAULT_OUT
    root = root or ROOT
    if not path.exists():
        return False, [f"missing manifest: {path}"]
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = build_manifest(root)
    problems: list[str] = []
    for k in sorted(set(expected) | set(actual)):
        if k not in expected:
            problems.append(f"NEW: {k}")
        elif k not in actual:
            problems.append(f"MISSING: {k}")
        elif expected[k] != actual[k]:
            problems.append(f"CHANGED: {k}")
    return len(problems) == 0, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.check:
        ok, problems = check_manifest(args.out)
        if ok:
            print("PASS: frozen hashes match")
            return 0
        print("FAIL: frozen hash mismatches:")
        for p in problems:
            print(f"  {p}")
        return 1
    out = write_manifest(args.out)
    print(f"Wrote {out} ({len(build_manifest())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
