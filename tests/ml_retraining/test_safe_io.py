"""Task 1.2: failing tests for fraud_detect.ml.retraining.safe_io (written before the module)."""
from __future__ import annotations

import builtins
import hashlib
import os
import subprocess

import pytest

from fraud_detect.ml.retraining import safe_io
from fraud_detect.ml.retraining.safe_io import (
    ReadOnlySource, SourceChanged, WriteGuardError, atomic_write, ensure_ignored, sha256_file)


def test_ensure_ignored_uses_injected_predicate(tmp_path, is_ignored):
    target = tmp_path / "out.bin"
    ensure_ignored(target, is_ignored=is_ignored)
    assert is_ignored.calls == [target]


def test_ensure_ignored_names_path_when_not_ignored(tmp_path, is_ignored):
    is_ignored.ignored = False
    target = tmp_path / "tracked.txt"
    with pytest.raises(WriteGuardError) as exc:
        ensure_ignored(target, is_ignored=is_ignored)
    assert str(target) in str(exc.value)


def test_ensure_ignored_calls_git_check_ignore(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, *a, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"x\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ensure_ignored(tmp_path / "f.txt")
    assert seen and seen[0][0] == "git" and "check-ignore" in seen[0]


def test_ensure_ignored_exit_1_means_not_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b""))
    with pytest.raises(WriteGuardError):
        ensure_ignored(tmp_path / "f.txt")


@pytest.mark.parametrize("failure", [FileNotFoundError("git"), OSError("boom")])
def test_ensure_ignored_fails_closed_when_git_unavailable(tmp_path, monkeypatch, failure):
    def fake_run(*a, **kw):
        raise failure

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = tmp_path / "f.txt"
    with pytest.raises(WriteGuardError) as exc:
        ensure_ignored(target)
    assert str(target) in str(exc.value)


def test_ensure_ignored_fails_closed_on_git_error_status(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 128, stdout=b"", stderr=b"fatal"))
    with pytest.raises(WriteGuardError):
        ensure_ignored(tmp_path / "f.txt")


def test_atomic_write_writes_bytes_and_creates_parents(tmp_path, is_ignored):
    target = tmp_path / "a" / "b" / "file.bin"
    atomic_write(target, b"hello", is_ignored=is_ignored)
    assert target.read_bytes() == b"hello"
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]


def test_atomic_write_accepts_text(tmp_path, is_ignored):
    target = tmp_path / "t.json"
    atomic_write(target, "{}", is_ignored=is_ignored)
    assert target.read_text() == "{}"


def test_atomic_write_failure_leaves_no_partial_file(tmp_path, is_ignored, monkeypatch):
    target = tmp_path / "f.bin"

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(target, b"data", is_ignored=is_ignored)
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_failure_keeps_previous_content(tmp_path, is_ignored, monkeypatch):
    target = tmp_path / "f.bin"
    target.write_bytes(b"old")
    monkeypatch.setattr(os, "replace", lambda *a, **kw: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        atomic_write(target, b"new", is_ignored=is_ignored)
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]


def test_write_to_non_ignored_path_writes_nothing_and_raises(tmp_path, is_ignored):
    is_ignored.ignored = False
    target = tmp_path / "sub" / "f.bin"
    with pytest.raises(WriteGuardError):
        atomic_write(target, b"data", is_ignored=is_ignored)
    assert list(tmp_path.rglob("*")) == []


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    data = os.urandom(200_000) + b"tail"
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_readonly_source_reads_hashes_and_reports_stat(tmp_path):
    p = tmp_path / "src" / "a.pdf"
    p.parent.mkdir()
    p.write_bytes(b"%PDF fake")
    with ReadOnlySource(p) as src:
        assert src.read_bytes() == b"%PDF fake"
        assert src.sha256 == hashlib.sha256(b"%PDF fake").hexdigest()
        assert src.path == p
    assert [x.name for x in p.parent.iterdir()] == ["a.pdf"]


def test_readonly_source_opens_rb_only(tmp_path, monkeypatch):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"abc")
    modes = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **kw):
        if os.fspath(file) == str(p):
            modes.append(mode)
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    with ReadOnlySource(p) as src:
        src.read_bytes()
        _ = src.sha256
    assert modes and all(m == "rb" for m in modes)


def test_readonly_source_fails_when_content_or_mtime_changes(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"abc")
    with pytest.raises(SourceChanged) as exc:
        with ReadOnlySource(p) as src:
            src.read_bytes()
            p.write_bytes(b"abcdef")
    assert str(p) in str(exc.value)

    q = tmp_path / "b.pdf"
    q.write_bytes(b"abc")
    with pytest.raises(SourceChanged):
        with ReadOnlySource(q) as src:
            src.read_bytes()
            st = q.stat()
            os.utime(q, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_readonly_source_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        with ReadOnlySource(tmp_path / "nope.pdf") as src:
            src.read_bytes()


def test_readonly_source_has_no_write_api():
    assert not any(hasattr(ReadOnlySource, n) for n in ("write", "write_bytes", "write_text", "open_w"))


def test_module_imports_only_stdlib_at_import_time():
    code = ("import sys, fraud_detect.ml.retraining.safe_io;"
            "bad=[m for m in ('torch','sklearn','pdfplumber') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    import sys
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    assert safe_io is not None
