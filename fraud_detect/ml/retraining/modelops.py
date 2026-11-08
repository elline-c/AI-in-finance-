"""Model operations: switch and restore (Task 21.3).

``switch`` backs up the Current model (SHA-256 verified), records the backup in
``index.json``, verifies the Candidate hash against the report, then replaces
the Current model. ``restore`` copies the recorded backup back and verifies it.
Only the installed copy is ever edited (Node 6b ``size_guard``); a file in the
work folder is never modified. Standard library at module level; ``joblib`` is
imported inside functions.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from fraud_detect.ml.retraining.safe_io import atomic_write, ensure_ignored, sha256_file

DEFAULT_BACKUP_DIR = Path("models/backup")
DEFAULT_MODELS_DIR = Path("models")
MODEL_FILES = {"6b": "statement_forgery_clf.joblib", "6a": "txn_fraud_clf.joblib"}
SIZE_GUARD_CHOICES = ("candidate", "off", "current")


class ModelOpsError(Exception):
    """Base class for model operation failures."""


class BackupError(ModelOpsError):
    """The backup could not be written or verified."""


class NoBackupError(ModelOpsError):
    """No recorded backup exists for the node."""


class CandidateMismatchError(ModelOpsError):
    """The Candidate file hash differs from the hash in the report."""


class SizeGuardRequired(ModelOpsError, ValueError):
    """Node 6b switch needs an explicit size-guard choice."""


@dataclass(frozen=True)
class SwitchRecord:
    node: str
    report_id: str
    current_sha256: str
    candidate_sha256: str
    backup_sha256: str
    backup_path: Path
    installed_sha256: str


@dataclass(frozen=True)
class RestoreRecord:
    node: str
    backup_sha256: str
    restored_sha256: str


def _check_node(node: str) -> str:
    if node not in MODEL_FILES:
        raise ValueError(f"unknown node {node!r}; expected one of {sorted(MODEL_FILES)}")
    return MODEL_FILES[node]


def _read_index(backup_dir: Path) -> dict:
    index_path = backup_dir / "index.json"
    if not index_path.is_file():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupError(f"backup index unreadable: {index_path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _install(source_bytes_path: Path, target: Path, edit: Optional[Callable] = None) -> None:
    """Atomically place a file at ``target`` (same folder temp file, then replace)."""
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if edit is None:
            shutil.copyfile(source_bytes_path, tmp)
        else:
            edit(tmp)
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def switch(node: str, report_id: str, *, size_guard: Optional[str] = None,
           override_reason: Optional[str] = None,
           models_dir=DEFAULT_MODELS_DIR, backup_dir=DEFAULT_BACKUP_DIR,
           candidate_path, report_candidate_sha256: str,
           is_ignored: Optional[Callable[[Path], bool]] = None) -> SwitchRecord:
    name = _check_node(node)
    if node == "6b" and size_guard not in SIZE_GUARD_CHOICES:
        raise SizeGuardRequired(
            f"Node 6b switch requires size_guard in {SIZE_GUARD_CHOICES}; got {size_guard!r}")
    models_dir, backup_dir, candidate_path = Path(models_dir), Path(backup_dir), Path(candidate_path)
    current = models_dir / name

    candidate_sha = sha256_file(candidate_path)
    if candidate_sha != report_candidate_sha256:
        raise CandidateMismatchError(
            f"candidate hash {candidate_sha} does not match the report hash {report_candidate_sha256}")
    if not current.is_file():
        raise ModelOpsError(f"current model not found: {current}")

    ensure_ignored(backup_dir, is_ignored=is_ignored)   # WriteGuardError propagates
    current_sha = sha256_file(current)
    backup_file = backup_dir / f"{current.stem}.{current_sha[:8]}.joblib"

    try:
        atomic_write(backup_file, current.read_bytes(), is_ignored=is_ignored)
    except OSError as exc:
        raise BackupError(f"could not write backup {backup_file}: {exc}") from exc
    backup_sha = sha256_file(backup_file)
    if backup_sha != current_sha:
        try:
            backup_file.unlink()
        except OSError:
            pass
        raise BackupError(
            f"backup hash {backup_sha} differs from current hash {current_sha}; current model left unchanged")

    try:
        index = _read_index(backup_dir)
        index[node] = {
            "report_id": report_id,
            "current_sha256": current_sha,
            "candidate_sha256": candidate_sha,
            "backup_sha256": backup_sha,
            "backup_file": backup_file.name,
        }
        if override_reason:
            index[node]["override_reason"] = override_reason
        atomic_write(backup_dir / "index.json",
                     json.dumps(index, indent=2, sort_keys=True) + "\n", is_ignored=is_ignored)
    except OSError as exc:
        raise BackupError(f"could not write backup index: {exc}") from exc

    edit = None
    if node == "6b" and size_guard != "candidate":
        import joblib
        if size_guard == "off":
            new_ranges = None
        else:
            new_ranges = joblib.load(current).get("size_ranges")

        def edit(tmp: Path, _ranges=new_ranges) -> None:
            bundle = joblib.load(candidate_path)
            bundle["size_ranges"] = _ranges
            joblib.dump(bundle, tmp)

    _install(candidate_path, current, edit)
    return SwitchRecord(node=node, report_id=report_id, current_sha256=current_sha,
                        candidate_sha256=candidate_sha, backup_sha256=backup_sha,
                        backup_path=backup_file, installed_sha256=sha256_file(current))


def restore(node: str, *, models_dir=DEFAULT_MODELS_DIR, backup_dir=DEFAULT_BACKUP_DIR,
            is_ignored: Optional[Callable[[Path], bool]] = None) -> RestoreRecord:
    name = _check_node(node)
    models_dir, backup_dir = Path(models_dir), Path(backup_dir)
    entry = _read_index(backup_dir).get(node)
    if not entry:
        raise NoBackupError(f"no recorded backup for node {node} in {backup_dir}")
    backup_file = backup_dir / entry["backup_file"]
    if not backup_file.is_file():
        raise NoBackupError(f"backup file missing for node {node}: {backup_file}")
    recorded = entry["backup_sha256"]
    if sha256_file(backup_file) != recorded:
        raise BackupError(f"backup file {backup_file} does not match its recorded hash; current model left unchanged")

    target = models_dir / name
    _install(backup_file, target)
    restored = sha256_file(target)
    if restored != recorded:
        raise ModelOpsError(f"restored file hash {restored} differs from recorded backup hash {recorded}")
    return RestoreRecord(node=node, backup_sha256=recorded, restored_sha256=restored)
