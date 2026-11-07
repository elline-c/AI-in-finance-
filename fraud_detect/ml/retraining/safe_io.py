"""Write guard and read-only source helpers (Task 1.3).

Standard library only. Every write the pipeline performs must go through
``atomic_write``, which refuses to write anywhere ``git check-ignore`` (or an
injected predicate) does not confirm is ignored.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

PathLike = Union[str, os.PathLike]


class WriteGuardError(Exception):
    """Raised when a write target is not confirmed to be git-ignored."""


class SourceChanged(Exception):
    """Raised when a read-only source file changed while it was open."""


def _git_check_ignore(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, OSError) as exc:
        raise WriteGuardError(f"could not confirm {path} is git-ignored: {exc}") from exc
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise WriteGuardError(
        f"could not confirm {path} is git-ignored: git check-ignore exited {result.returncode}")


def ensure_ignored(path: PathLike, is_ignored: Optional[Callable[[Path], bool]] = None) -> None:
    """Fail closed unless ``path`` is confirmed to be git-ignored.

    ``is_ignored`` lets tests inject a predicate instead of invoking git.
    """
    target = Path(path)
    checker = is_ignored if is_ignored is not None else _git_check_ignore
    ignored = checker(target)
    if not ignored:
        raise WriteGuardError(f"refusing to write to a path that is not git-ignored: {target}")


def atomic_write(path: PathLike, data: Union[bytes, str],
                  is_ignored: Optional[Callable[[Path], bool]] = None) -> None:
    """Write ``data`` to ``path`` atomically, after the write guard passes.

    On any failure, no partial file is left behind and any previous content
    at ``path`` is preserved.
    """
    target = Path(path)
    ensure_ignored(target, is_ignored=is_ignored)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, mode) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def sha256_file(path: PathLike) -> str:
    """SHA-256 hex digest of the file at ``path``, matching ``hashlib``."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ReadOnlySource:
    """Opens a source file ``rb`` only, hashes it, and verifies it did not change.

    ``os.stat`` is recorded on entry and compared again on exit; any change
    to size, mtime or content raises :class:`SourceChanged`. No file is ever
    created, modified or removed in the source folder.
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self._bytes: Optional[bytes] = None
        self._sha256: Optional[str] = None
        self._stat_before = None

    def __enter__(self) -> "ReadOnlySource":
        self._stat_before = self.path.stat()
        return self

    def read_bytes(self) -> bytes:
        if self._bytes is None:
            with open(self.path, "rb") as handle:
                self._bytes = handle.read()
        return self._bytes

    @property
    def sha256(self) -> str:
        if self._sha256 is None:
            self._sha256 = hashlib.sha256(self.read_bytes()).hexdigest()
        return self._sha256

    def __exit__(self, exc_type, exc, tb) -> None:
        stat_after = self.path.stat()
        before, after = self._stat_before, stat_after
        if (before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns):
            raise SourceChanged(f"source file changed while open: {self.path}")
        if exc_type is None and self._bytes is not None:
            current_hash = hashlib.sha256(self.read_bytes()).hexdigest()
            if current_hash != self._sha256:
                raise SourceChanged(f"source file changed while open: {self.path}")
