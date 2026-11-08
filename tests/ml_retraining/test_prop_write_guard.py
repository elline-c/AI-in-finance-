"""Task 1.4: Property 39, the write guard.

For any output path and any ignore predicate, a write happens if and only if the
predicate says the path is ignored; when it is not ignored, nothing is written to
that path and the error names the path.

**Validates: Requirements 17.5, 17.7**
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.ml.retraining.safe_io import WriteGuardError, atomic_write, ensure_ignored

_SEGMENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=8)
_REL_PATHS = st.lists(_SEGMENT, min_size=1, max_size=4).map(lambda parts: "/".join(parts) + ".bin")
_DATA = st.one_of(st.binary(max_size=200), st.text(max_size=50))


def _snapshot(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@settings(max_examples=100, deadline=None)
@given(rel=_REL_PATHS, data=_DATA, verdict=st.booleans(), old=st.one_of(st.none(), st.binary(max_size=50)))
def test_write_happens_iff_predicate_says_ignored(rel, data, verdict, old):
    """**Validates: Requirements 17.5, 17.7**"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / rel
        if old is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old)
        before = _snapshot(root)
        calls = []

        def predicate(path):
            calls.append(Path(path))
            return verdict

        if verdict:
            atomic_write(target, data, is_ignored=predicate)
            expected = data if isinstance(data, bytes) else data.encode()
            assert target.read_bytes() == expected
            assert not [p for p in root.rglob("*.tmp")]
        else:
            with pytest.raises(WriteGuardError) as exc:
                atomic_write(target, data, is_ignored=predicate)
            assert str(target) in str(exc.value)
            # zero bytes written: no new files or folders, old content untouched
            assert _snapshot(root) == before
            if old is None:
                assert not target.exists()
            else:
                assert target.read_bytes() == old
        assert calls == [target]


@settings(max_examples=100, deadline=None)
@given(rel=_REL_PATHS, verdict=st.booleans())
def test_ensure_ignored_raises_iff_not_ignored_and_names_path(rel, verdict):
    """**Validates: Requirements 17.5, 17.7**"""
    target = Path("/nonexistent-root") / rel
    if verdict:
        ensure_ignored(target, is_ignored=lambda p: True)
    else:
        with pytest.raises(WriteGuardError) as exc:
            ensure_ignored(target, is_ignored=lambda p: False)
        assert str(target) in str(exc.value)
    assert not target.exists()


@settings(max_examples=50, deadline=None)
@given(paths=st.lists(_REL_PATHS, min_size=1, max_size=6, unique=True), data=st.data())
def test_guard_follows_arbitrary_predicate_per_path(paths, data):
    """Each path is judged on its own: only paths the predicate accepts get written."""
    ignored = {p for p in paths if data.draw(st.booleans())}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pred = lambda path: str(Path(path).relative_to(root)) in ignored  # noqa: E731
        for rel in paths:
            target = root / rel
            if rel in ignored:
                atomic_write(target, b"x", is_ignored=pred)
            else:
                with pytest.raises(WriteGuardError):
                    atomic_write(target, b"x", is_ignored=pred)
        written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        assert written == ignored
