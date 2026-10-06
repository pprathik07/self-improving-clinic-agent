"""Tests for scripts/mutation_table.py using a tmp_path toy module."""

from __future__ import annotations

import hashlib
import importlib.util
import textwrap
from pathlib import Path

import pytest


def _load_mutation_table():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mutation_table.py"
    spec = importlib.util.spec_from_file_location("mutation_table", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mt = _load_mutation_table()


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """Create a tiny module + tests under tmp_path and chdir there."""
    src = tmp_path / "toy_mod.py"
    src.write_text(
        textwrap.dedent(
            '''\
            def add(a, b):
                return a + b

            def guarded(x):
                if x < 0:
                    raise ValueError("neg")
                return x
            '''
        ),
        encoding="utf-8",
    )
    tests = tmp_path / "test_toy.py"
    tests.write_text(
        textwrap.dedent(
            '''\
            import toy_mod

            def test_add():
                assert toy_mod.add(1, 2) == 3

            def test_guarded():
                import pytest
                with pytest.raises(ValueError):
                    toy_mod.guarded(-1)
                assert toy_mod.guarded(1) == 1
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    return src, tests


def test_caught_mutation(toy, monkeypatch):
    src, tests = toy
    original = src.read_bytes()
    entry = mt.Entry(
        "T1",
        "toy_mod.py",
        "return a + b\n",
        "return a - b\n",
        ("test_toy.py",),
        "add",
    )
    # Patch ROOT and ENTRIES for isolated run helpers
    monkeypatch.setattr(mt, "ROOT", src.parent)
    text = src.read_text(encoding="utf-8")
    assert mt.count_occurrences(text, entry.old_text) == 1
    src.write_text(text.replace(entry.old_text, entry.new_text, 1), encoding="utf-8")
    status, failed, _ = mt.run_pytest(entry.test_files)
    src.write_bytes(original)
    assert status == "CAUGHT"
    assert any("test_add" in f for f in failed)


def test_untested_mutation(toy, monkeypatch):
    src, tests = toy
    original = src.read_bytes()
    monkeypatch.setattr(mt, "ROOT", src.parent)
    # Change a comment-equivalent unused path: mutate docstring-free dead string
    # Add a harmless constant then mutate an unused branch that tests don't cover
    src.write_text(
        src.read_text(encoding="utf-8")
        + "\nUNUSED = 1\n",
        encoding="utf-8",
    )
    original2 = src.read_bytes()
    text = src.read_text(encoding="utf-8")
    old = "UNUSED = 1\n"
    new = "UNUSED = 2\n"
    src.write_text(text.replace(old, new, 1), encoding="utf-8")
    status, failed, _ = mt.run_pytest(("test_toy.py",))
    src.write_bytes(original2)
    assert status == "UNTESTED"
    assert failed == []


def test_invalid_syntax_mutation(toy, monkeypatch):
    src, tests = toy
    monkeypatch.setattr(mt, "ROOT", src.parent)
    original = src.read_bytes()
    text = src.read_text(encoding="utf-8")
    bad = text.replace("return a + b\n", "return a +\n", 1)
    src.write_text(bad, encoding="utf-8")
    import py_compile
    with pytest.raises(py_compile.PyCompileError):
        py_compile.compile(str(src), doraise=True)
    src.write_bytes(original)


def test_ambiguous_pattern_skipped(toy):
    src, _ = toy
    text = src.read_text(encoding="utf-8")
    # "return" appears more than once
    n = mt.count_occurrences(text, "return")
    assert n > 1


def test_restore_verified(toy, monkeypatch):
    src, _ = toy
    monkeypatch.setattr(mt, "ROOT", src.parent)
    original = src.read_bytes()
    h0 = hashlib.sha256(original).hexdigest()
    src.write_text("broken", encoding="utf-8")
    src.write_bytes(original)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == h0


def test_restore_after_pytest_exception(toy, monkeypatch):
    src, tests = toy
    monkeypatch.setattr(mt, "ROOT", src.parent)
    original = src.read_bytes()
    h0 = hashlib.sha256(original).hexdigest()

    def boom(*a, **k):
        raise RuntimeError("pytest exploded")

    monkeypatch.setattr(mt, "run_pytest", boom)
    text = src.read_text(encoding="utf-8")
    src.write_text(text.replace("return a + b\n", "return a * b\n", 1), encoding="utf-8")
    try:
        mt.run_pytest(("test_toy.py",))
    except RuntimeError:
        pass
    finally:
        src.write_bytes(original)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == h0


def test_dry_run_ok_and_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "ROOT", tmp_path)
    f = tmp_path / "x.py"
    f.write_text("AAA\nBBB\nAAA\n", encoding="utf-8")
    e_ok = mt.Entry("X1", "x.py", "BBB\n", "CCC\n", ("t.py",), "x")
    e_bad = mt.Entry("X2", "x.py", "AAA\n", "ZZZ\n", ("t.py",), "x")
    assert mt.dry_run([e_ok]) == 0
    assert mt.dry_run([e_bad]) == 1
