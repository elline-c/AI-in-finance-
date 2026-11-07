"""Task 21.5: Property 36, switch and restore round trip.

Only temporary folders and fake model bytes are used; the real ``models/`` folder is never touched.

Feature: ml-retraining, Property 36: Switch and restore round trip
**Validates: Requirements 14.9, 14.10, 14.11, 14.12**
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import modelops
from fraud_detect.ml.retraining.modelops import (
    BackupError, NoBackupError, restore, switch)
from fraud_detect.ml.retraining.safe_io import sha256_file

NAME_6A = "txn_fraud_clf.joblib"
NAME_6B = "statement_forgery_clf.joblib"

model_bytes = st.binary(min_size=1, max_size=300)
SETTINGS = settings(max_examples=60, deadline=None)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _always_ignored(path) -> bool:
    return True


class Sandbox:
    def __init__(self, root: Path, current: bytes, other: bytes, candidate: bytes):
        self.models = root / "models"
        self.models.mkdir()
        (self.models / NAME_6A).write_bytes(current)
        (self.models / NAME_6B).write_bytes(other)
        self.backup = root / "backup"
        self.cand = root / "candidate.joblib"
        self.cand.write_bytes(candidate)

    def switch(self, report_id="R-1", report_hash=None):
        return switch("6a", report_id, models_dir=self.models, backup_dir=self.backup,
                      candidate_path=self.cand,
                      report_candidate_sha256=report_hash or _sha(self.cand.read_bytes()),
                      is_ignored=_always_ignored)

    def restore(self):
        return restore("6a", models_dir=self.models, backup_dir=self.backup,
                       is_ignored=_always_ignored)

    def current(self) -> bytes:
        return (self.models / NAME_6A).read_bytes()


@given(current=model_bytes, other=model_bytes, candidate=model_bytes,
       report_id=st.text(min_size=1, max_size=20))
@SETTINGS
def test_switch_then_restore_round_trip(current, other, candidate, report_id):
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(Path(tmp), current, other, candidate)
        rec = sb.switch(report_id=report_id)

        # backup equals the Current_Model; index records report id and both hashes
        assert rec.current_sha256 == rec.backup_sha256 == _sha(current)
        assert rec.candidate_sha256 == _sha(candidate)
        assert Path(rec.backup_path).read_bytes() == current
        entry = json.loads((sb.backup / "index.json").read_text())["6a"]
        assert entry["report_id"] == report_id
        assert entry["current_sha256"] == _sha(current)
        assert entry["candidate_sha256"] == _sha(candidate)
        assert entry["backup_sha256"] == _sha(current)

        # installed file equals the Candidate; the other node is untouched
        assert sb.current() == candidate
        assert (sb.models / NAME_6B).read_bytes() == other

        # restore returns exactly the recorded backup bytes
        rrec = sb.restore()
        assert rrec.restored_sha256 == rrec.backup_sha256 == entry["backup_sha256"]
        assert sb.current() == current
        assert _sha(sb.current()) == entry["backup_sha256"]
        assert (sb.models / NAME_6B).read_bytes() == other


@given(current=model_bytes, other=model_bytes, candidate=model_bytes)
@SETTINGS
def test_backup_write_failure_leaves_current_unchanged(current, other, candidate):
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(Path(tmp), current, other, candidate)
        sb.backup.write_text("a file where the backup folder should be")
        with pytest.raises(BackupError):
            sb.switch()
        assert sb.current() == current


@given(current=model_bytes, other=model_bytes, candidate=model_bytes)
@SETTINGS
def test_backup_hash_mismatch_leaves_current_unchanged(current, other, candidate):
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(Path(tmp), current, other, candidate)

        def lying(path):
            return "f" * 64 if sb.backup in Path(path).parents else sha256_file(path)

        with mock.patch.object(modelops, "sha256_file", lying):
            with pytest.raises(BackupError):
                sb.switch()
        assert sb.current() == current


@given(current=model_bytes, other=model_bytes, candidate=model_bytes)
@SETTINGS
def test_restore_without_backup_fails_and_leaves_current_unchanged(current, other, candidate):
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(Path(tmp), current, other, candidate)
        with pytest.raises(NoBackupError):
            sb.restore()
        assert sb.current() == current
        assert (sb.models / NAME_6B).read_bytes() == other
