"""Append-only decision log and post-test change log (Task 1.3). Standard library only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .safe_io import atomic_write

#: Valid decisions per entry kind.
DECISION_KINDS: Dict[str, Tuple[str, ...]] = {
    "leak-feature": ("exclude", "accept"),
    "investigation": ("continue", "abort"),
    "waive-gate": ("waive",),
    "awkward-review": ("reviewed",),
}

POST_TEST_KINDS = ("cutoff", "feature", "threshold", "hyperparameter", "other",
                   "generator", "detector")


class DecisionError(Exception):
    """Raised for an unknown kind or a decision that does not fit the kind."""


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _load(path: Path, key: str) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data[key] if isinstance(data, dict) else data)


class DecisionLog:
    """Append-only log stored as ``decisions.json``. No edit or remove API."""

    def __init__(self, path, is_ignored: Optional[Callable] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.path = Path(path)
        self._is_ignored = is_ignored
        self._clock = clock or _default_clock
        self._entries: List[dict] = _load(self.path, "decisions")

    def entries(self) -> Tuple[Mapping, ...]:
        return tuple(MappingProxyType(dict(e)) for e in self._entries)

    def append(self, kind: str, target: str, decision: str, reason: str, author: str) -> dict:
        if kind not in DECISION_KINDS:
            raise DecisionError(f"unknown decision kind: {kind!r}")
        if decision not in DECISION_KINDS[kind]:
            raise DecisionError(f"decision {decision!r} is not valid for kind {kind!r}")
        entry = {
            "id": f"D-{len(self._entries) + 1:04d}",
            "kind": kind,
            "target": target,
            "decision": decision,
            "reason": reason,
            "author": author,
            "date": self._clock().isoformat(),
        }
        new_entries = self._entries + [entry]
        atomic_write(self.path, json.dumps({"decisions": new_entries}, indent=2, sort_keys=True),
                     is_ignored=self._is_ignored)
        self._entries = new_entries
        return dict(entry)


class PostTestChangeLog:
    """Append-only log of changes made after test data was seen (``post_test_changes.json``)."""

    def __init__(self, path, is_ignored: Optional[Callable] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.path = Path(path)
        self._is_ignored = is_ignored
        self._clock = clock or _default_clock
        self._entries: List[dict] = _load(self.path, "changes")

    def entries(self) -> Tuple[Mapping, ...]:
        return tuple(MappingProxyType(dict(e)) for e in self._entries)

    def append(self, kind: str, description: str, affected: Sequence[str],
               trigger: Optional[str] = None) -> dict:
        if kind not in POST_TEST_KINDS:
            raise DecisionError(f"unknown post-test change kind: {kind!r}")
        entry = {
            "date": self._clock().isoformat(),
            "kind": kind,
            "description": description,
            "trigger": trigger,
            "affected": list(affected),
        }
        new_entries = self._entries + [entry]
        atomic_write(self.path, json.dumps({"changes": new_entries}, indent=2, sort_keys=True),
                     is_ignored=self._is_ignored)
        self._entries = new_entries
        return dict(entry)
