"""Freeze record, held-out vault and post-test change recording (Task 13.2).

Standard library plus the package's own ``safe_io``, ``config`` and ``decisions`` modules.
Held-out datasets are reachable only through ``HeldOutVault(...).open().load(...)``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

from .decisions import PostTestChangeLog
from .safe_io import atomic_write

RUN_FILE = "run.json"
FROZEN_FILE = "frozen.json"
RESULTS_FILE = "heldout_results.json"
CHANGES_FILE = "post_test_changes.json"

HELD_OUT = "held out"
NO_LONGER_HELD_OUT = "no longer held out"

#: Repository-relative files and folders covered by the code fingerprint.
CODE_FINGERPRINT_PATHS = (
    "fraud_detect/corpus",
    "fraud_detect/nodes",
    "fraud_detect/ml/statement_features.py",
    "fraud_detect/ml/transaction_features.py",
    "fraud_detect/registry.py",
    "fraud_detect/reference_data.py",
)


class VaultError(Exception):
    """Base class for vault failures."""


class VaultClosed(VaultError):
    """The vault cannot be opened: no valid freeze record for the current run.json."""


class MissingClassError(VaultError, ValueError):
    """A split has no genuine or no tampered document."""


class RuleHashMismatch(VaultError):
    """The Switch_Rule hash in a report differs from the one recorded in run.json."""


@dataclass(frozen=True)
class FreezeRecord:
    run_hash: str
    cutoffs: Mapping
    candidate_sha256: Mapping
    switch_rule_sha256: str


@dataclass(frozen=True)
class FrozenCutoffs:
    medium: float
    high: float


# ---------------------------------------------------------------- helpers


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- code fingerprint


def _covered_files(root: Path, paths: Sequence[str]):
    for rel in paths:
        base = root / rel
        if base.is_file():
            yield rel, base
        elif base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    yield p.relative_to(root).as_posix(), p


def code_fingerprint(root, paths: Sequence[str] = CODE_FINGERPRINT_PATHS) -> str:
    """SHA-256 over the relative name and content of every covered file."""
    root = Path(root)
    entries = sorted((name, _sha256_bytes(p.read_bytes())) for name, p in _covered_files(root, paths))
    digest = hashlib.sha256()
    for name, file_hash in entries:
        digest.update(f"{name}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def unchanged_since_run_start(run_dir, root) -> bool:
    data = _read_json(Path(run_dir) / RUN_FILE)
    return code_fingerprint(root) == data["code_fingerprint"]


def unchanged_statement(run_dir, root) -> str:
    answer = "yes" if unchanged_since_run_start(run_dir, root) else "no"
    return f"generator and detector code unchanged since run start: {answer}"


# ---------------------------------------------------------------- run.json and freeze


def write_run_json(run_dir, *, config, switch_rule_sha256: str, cutoff_rule: Mapping,
                   fingerprint: str, is_ignored: Optional[Callable] = None,
                   extra: Optional[Mapping] = None) -> str:
    """Write run.json (before any held-out score). Returns the SHA-256 of the file.

    ``extra`` adds further top-level keys (the CLI stores ``switch_rule_text`` there); the
    four standard keys always win.
    """
    payload = dict(extra or {})
    payload.update({
        "config": config.canonical_json(),
        "switch_rule_sha256": switch_rule_sha256,
        "cutoff_rule": dict(cutoff_rule),
        "code_fingerprint": fingerprint,
    })
    text = _dump(payload)
    atomic_write(Path(run_dir) / RUN_FILE, text, is_ignored=is_ignored)
    return _sha256_bytes(text.encode("utf-8"))


def write_freeze(run_dir, *, cutoffs: Mapping, candidate_sha256: Mapping,
                 switch_rule_sha256: str, is_ignored: Optional[Callable] = None) -> FreezeRecord:
    run_path = Path(run_dir) / RUN_FILE
    if not run_path.exists():
        raise VaultError("run.json must be written before the freeze")
    run_bytes = run_path.read_bytes()
    recorded = json.loads(run_bytes.decode("utf-8")).get("switch_rule_sha256")
    if recorded != switch_rule_sha256:
        raise VaultError("Switch_Rule hash differs from the one recorded in run.json")
    record = FreezeRecord(
        run_hash=_sha256_bytes(run_bytes),
        cutoffs=json.loads(json.dumps(cutoffs)),
        candidate_sha256=json.loads(json.dumps(candidate_sha256)),
        switch_rule_sha256=switch_rule_sha256,
    )
    atomic_write(Path(run_dir) / FROZEN_FILE,
                 _dump({"run_hash": record.run_hash, "cutoffs": record.cutoffs,
                        "candidate_sha256": record.candidate_sha256,
                        "switch_rule_sha256": record.switch_rule_sha256}),
                 is_ignored=is_ignored)
    return record


def check_classes(splits: Mapping[str, Sequence[int]]) -> None:
    """Fail before fitting when ``train`` or ``validation`` lacks a class (1 tampered, 0 genuine)."""
    for name in ("train", "validation"):
        labels = list(splits.get(name, ()))
        if not any(int(v) == 0 for v in labels):
            raise MissingClassError(f"split {name!r} has no genuine documents")
        if not any(int(v) == 1 for v in labels):
            raise MissingClassError(f"split {name!r} has no tampered documents")


# ---------------------------------------------------------------- the vault


class HeldOutVault:
    """Closed until a valid freeze record exists; ``open()`` returns the only access object."""

    def __init__(self, run_dir, loaders: Optional[Mapping[str, Callable]] = None,
                 is_ignored: Optional[Callable] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self._run_dir = Path(run_dir)
        self._loaders = dict(loaders or {})
        self._is_ignored = is_ignored
        self._clock = clock

    def open(self) -> "OpenVault":
        run_path = self._run_dir / RUN_FILE
        frozen_path = self._run_dir / FROZEN_FILE
        if not frozen_path.exists() or not run_path.exists():
            raise VaultClosed("no freeze record: frozen.json and run.json are both required")
        try:
            frozen = _read_json(frozen_path)
            run_bytes = run_path.read_bytes()
            run = json.loads(run_bytes.decode("utf-8"))
            record = FreezeRecord(
                run_hash=frozen["run_hash"], cutoffs=frozen["cutoffs"],
                candidate_sha256=frozen["candidate_sha256"],
                switch_rule_sha256=frozen["switch_rule_sha256"])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise VaultClosed(f"freeze record unreadable: {exc}") from exc
        if record.run_hash != _sha256_bytes(run_bytes):
            raise VaultClosed("run.json differs from the version recorded at the freeze")
        return OpenVault(self._run_dir, record, run.get("switch_rule_sha256"), self._loaders,
                         self._is_ignored, self._clock)


class OpenVault:
    def __init__(self, run_dir: Path, freeze: FreezeRecord, run_rule_hash, loaders: Dict,
                 is_ignored, clock):
        self.freeze = freeze
        self._run_dir = run_dir
        self._run_rule_hash = run_rule_hash
        self._loaders = loaders
        self._is_ignored = is_ignored
        self._clock = clock
        results_path = run_dir / RESULTS_FILE
        self._results: Dict[str, dict] = (
            _read_json(results_path)["results"] if results_path.exists() else {})

    def cutoffs(self, node: str) -> FrozenCutoffs:
        try:
            cut = self.freeze.cutoffs[node]
            return FrozenCutoffs(medium=cut["medium"], high=cut["high"])
        except (KeyError, TypeError) as exc:
            raise VaultError(f"no frozen cutoffs for node {node!r}") from exc

    def check_report_rule_hash(self, rule_hash: str) -> None:
        if rule_hash != self._run_rule_hash:
            raise RuleHashMismatch("report Switch_Rule hash differs from run.json")

    def load(self, dataset_id: str):
        if dataset_id not in self._loaders:
            raise VaultError(f"dataset {dataset_id!r} is not registered with the vault")
        return self._loaders[dataset_id]()

    def _save_results(self, results: Dict[str, dict]) -> None:
        atomic_write(self._run_dir / RESULTS_FILE, _dump({"results": results}),
                     is_ignored=self._is_ignored)
        self._results = results

    def record_score(self, dataset_id: str, values: Mapping) -> None:
        results = {k: dict(v) for k, v in self._results.items()}
        results[dataset_id] = {"values": dict(values), "held_out": HELD_OUT}
        self._save_results(results)

    def results(self) -> Dict[str, dict]:
        return {k: {"values": dict(v["values"]), "held_out": v["held_out"]}
                for k, v in self._results.items()}

    def record_change(self, kind: str, description: str, trigger: Optional[str] = None):
        """Log a change made after the first held-out score and mark existing results.

        Returns the log entry, or ``None`` when no held-out score exists yet.
        """
        log = PostTestChangeLog(self._run_dir / CHANGES_FILE, is_ignored=self._is_ignored,
                                clock=self._clock)
        if not self._results:
            # Still validate the kind, but write nothing.
            from .decisions import POST_TEST_KINDS, DecisionError
            if kind not in POST_TEST_KINDS:
                raise DecisionError(f"unknown post-test change kind: {kind!r}")
            return None
        affected = [k for k, v in self._results.items() if v["held_out"] == HELD_OUT]
        entry = log.append(kind, description, affected, trigger)
        marked = {k: dict(v) for k, v in self._results.items()}
        for k in affected:
            marked[k]["held_out"] = NO_LONGER_HELD_OUT
        self._save_results(marked)
        return entry
