"""Build-time UEN collision check for the corpus generator (Requirement 20).

An invented company number must never be a real one. ``UenChecker.is_taken`` answers
"is this UEN already a real or reserved number?" from up to three sources:

1. a snapshot file of real UENs the user downloaded (``--uen-snapshot``),
2. the in-repo registry (``fraud_detect.reference_data.ACRA_REGISTRY``), reached only
   through the public ``registry.lookup_entity`` with ``live=False``,
3. the live data.gov.sg lookup (``--check-live-uen``), through the same public function
   with ``live=True`` passed explicitly, so ``ACRA_LIVE_LOOKUP`` is never read.

Only the ``state`` of a lookup result is read. No company name is kept or stored.
A live lookup that cannot answer raises ``UenCheckError``: "could not ask" is never
treated as "free". This module makes no network call itself and imports no network
library; the registry module does that, and only when ``live`` is true.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from fraud_detect.registry import FOUND, UNAVAILABLE, lookup_entity

__all__ = ["UenCheckError", "UenChecker", "load_snapshot"]

SOURCE_LOCAL = "in-repo registry only"
SOURCE_SNAPSHOT = "snapshot"
SOURCE_LIVE = "live"
SOURCE_BOTH = "snapshot and live"


class UenCheckError(Exception):
    """The UEN check could not be completed; the message is plain language."""


def _norm(uen: str) -> str:
    return str(uen).strip().upper()


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise UenCheckError(f"UEN snapshot {path.name!r} was not found ({path})") from None
    except OSError as exc:
        raise UenCheckError(f"UEN snapshot {path.name!r} could not be read: "
                            f"{type(exc).__name__}") from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UenCheckError(f"UEN snapshot {path.name!r} is not a UTF-8 text file") from None


def _from_json(text: str, name: str) -> set[str]:
    try:
        data = json.loads(text) if text.strip() else []
    except ValueError:
        raise UenCheckError(f"UEN snapshot {name!r} is not valid JSON") from None
    if not isinstance(data, list):
        raise UenCheckError(f"UEN snapshot {name!r}: expected a JSON list of UENs "
                            "(strings, or objects with a 'uen' key)")
    found: set[str] = set()
    for item in data:
        if isinstance(item, str):
            found.add(_norm(item))
        elif isinstance(item, dict) and isinstance(item.get("uen"), str):
            found.add(_norm(item["uen"]))
        else:
            raise UenCheckError(f"UEN snapshot {name!r}: every JSON item must be a string "
                                "or an object with a 'uen' key")
    return found


def _from_csv(text: str, name: str) -> set[str]:
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        return set()
    names = [h.strip().lower() for h in header]
    if "uen" not in names:
        raise UenCheckError(f"UEN snapshot {name!r}: the CSV header has no column named 'uen'")
    col = names.index("uen")
    return {_norm(row[col]) for row in reader if len(row) > col and row[col].strip()}


def _from_lines(text: str) -> set[str]:
    lines = (ln.strip() for ln in text.splitlines())
    return {_norm(ln) for ln in lines if ln and not ln.startswith("#")}


def load_snapshot(path: Union[str, Path]) -> frozenset[str]:
    """Read a snapshot of real UENs: ``.json``, ``.csv`` (column ``uen``) or text lines.

    UENs are stripped and upper-cased. A missing, unreadable, empty or unrecognised
    file raises ``UenCheckError`` naming the file.
    """
    p = Path(path)
    text = _read_text(p)
    suffix = p.suffix.lower()
    if suffix == ".json":
        found = _from_json(text, p.name)
    elif suffix == ".csv":
        found = _from_csv(text, p.name)
    else:
        found = _from_lines(text)
    if not found:
        raise UenCheckError(f"UEN snapshot {p.name!r} is empty: it holds no UEN")
    return frozenset(found)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UenChecker:
    """Answers ``is_taken(uen)`` and keeps counters for the manifest."""

    def __init__(self, snapshot: Iterable[str] = frozenset(), *, live: bool = False,
                 fetch: Optional[Callable[[str], dict]] = None,
                 snapshot_file: Union[str, Path, None] = None) -> None:
        self.snapshot = frozenset(_norm(u) for u in snapshot)
        self.live = bool(live)
        self.fetch = fetch
        self.snapshot_file = Path(snapshot_file) if snapshot_file is not None else None
        self.checked = 0
        self.redraws = 0

    @classmethod
    def from_options(cls, snapshot_path: Union[str, Path, None] = None, *, live: bool = False,
                     fetch: Optional[Callable[[str], dict]] = None) -> "UenChecker":
        """Build a checker from the build options; loads the snapshot if one is named."""
        if snapshot_path is None:
            return cls(live=live, fetch=fetch)
        return cls(load_snapshot(snapshot_path), live=live, fetch=fetch,
                   snapshot_file=snapshot_path)

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot_file is not None or bool(self.snapshot)

    @property
    def real_registry_checked(self) -> bool:
        return self.has_snapshot or self.live

    def is_taken(self, uen: str) -> bool:
        key = _norm(uen)
        self.checked += 1
        if key in self.snapshot:
            return True
        if lookup_entity(key, live=False).state == FOUND:
            return True
        if not self.live:
            return False
        result = lookup_entity(key, live=True, fetch=self.fetch)
        if result.state == UNAVAILABLE:
            reason = result.detail or "no answer"
            raise UenCheckError(f"live UEN check could not be completed (reason: {reason}); "
                                "nothing was written")
        return result.state == FOUND

    def describe(self) -> dict:
        """Counts and labels for ``corpus_manifest.json``; never a company name."""
        if self.has_snapshot and self.live:
            source = SOURCE_BOTH
        elif self.has_snapshot:
            source = SOURCE_SNAPSHOT
        elif self.live:
            source = SOURCE_LIVE
        else:
            source = SOURCE_LOCAL
        return {
            "source": source,
            "snapshot_file": self.snapshot_file.name if self.snapshot_file else None,
            "snapshot_sha256": _sha256(self.snapshot_file) if self.snapshot_file else None,
            "snapshot_uens": len(self.snapshot),
            "uens_checked": self.checked,
            "redraws": self.redraws,
            "real_registry_checked": self.real_registry_checked,
        }
