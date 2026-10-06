"""Mutation check for loop guards.

Replaces each `raise PatchRejected(...)` in the target file with `pass`, ONE AT A TIME,
runs the tests, and reports which test caught it. A row with no failing test means
the guard is untested. The original file is always restored (verified by hash).

Run from the repo root:  uv run python scripts/mutation_check.py
Optional args:           [target_file] [test_file]
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "clinic_agent/loop/patch.py")
TESTS = sys.argv[2] if len(sys.argv) > 2 else "tests/test_patch.py"


def is_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and getattr(node.exc.func, "id", "") == "PatchRejected"
    )


class DisableNth(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.seen = -1
        self.line = 0
        self.label = "?"

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        if is_guard(node):
            self.seen += 1
            if self.seen == self.target_index:
                self.line = node.lineno
                call = node.exc
                self.label = ast.unparse(call.args[0]) if call.args else "?"
                return ast.copy_location(ast.Pass(), node)
        return node


def run_tests() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )
    failed = next(
        (ln.split(" - ")[0].replace("FAILED ", "").strip()
         for ln in proc.stdout.splitlines() if ln.startswith("FAILED")),
        "",
    )
    return proc.returncode, failed


def main() -> int:
    original = TARGET.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    tree = ast.parse(original.decode("utf-8"))
    guard_count = sum(1 for n in ast.walk(tree) if is_guard(n))
    print(f"Target: {TARGET}  guards found: {guard_count}")

    code, _ = run_tests()
    if code != 0:
        print("ABORT: tests are not green before mutating (is a guard already disabled?).")
        return 2

    rows: list[tuple[int, str, str]] = []
    try:
        for i in range(guard_count):
            mutator = DisableNth(i)
            mutated = mutator.visit(ast.parse(original.decode("utf-8")))
            ast.fix_missing_locations(mutated)
            TARGET.write_text(ast.unparse(mutated), encoding="utf-8")
            code, failed = run_tests()
            rows.append((mutator.line, mutator.label, failed if code != 0 else "NONE  <-- UNTESTED GUARD"))
    finally:
        TARGET.write_bytes(original)

    restored = hashlib.sha256(TARGET.read_bytes()).hexdigest() == digest
    print(f"\n{'line':>5}  {'guard':<42} caught by")
    for line, label, caught in rows:
        print(f"{line:>5}  {label:<42} {caught}")
    print(f"\nFile restored: {'OK' if restored else 'FAILED - restore from git/backup!'}")
    untested = [r for r in rows if r[2].startswith("NONE")]
    print(f"Untested guards: {len(untested)}")
    return 1 if (untested or not restored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
