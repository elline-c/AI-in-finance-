"""Task 21.1: failing tests for fraud_detect.ml.retraining.modelops (written before the module).

Hard rule: only temporary models and backup folders (``fake_current_models`` and tmp_path); the
real ``models/`` folder is never read or written. The only touch of the repo is a read-only
``git check-ignore`` on the *default* backup path.

Pinned interface:

    DEFAULT_BACKUP_DIR == Path("models/backup")
    MODEL_FILES == {"6b": "statement_forgery_clf.joblib", "6a": "txn_fraud_clf.joblib"}
    switch(node, report_id, *, size_guard=None, override_reason=None,
           models_dir, backup_dir, candidate_path, report_candidate_sha256,
           is_ignored=None) -> SwitchRecord(node, report_id, current_sha256, candidate_sha256,
                                            backup_sha256, backup_path, installed_sha256)
    restore(node, *, models_dir, backup_dir, is_ignored=None) -> RestoreRecord(node, backup_sha256, restored_sha256)
    Errors (all subclass ModelOpsError): BackupError, NoBackupError, CandidateMismatchError,
           SizeGuardRequired (also a ValueError)
    <backup_dir>/index.json : {node: {report_id, current_sha256, candidate_sha256, backup_sha256, backup_file}}
    backup file name        : "<model stem>.<current sha256[:8]>.joblib"
    size_guard in {"candidate","off","current"} required for "6b"; only the installed copy is edited
    (bundle key "size_ranges": candidate -> keep the candidate's, off -> None, current -> the Current's).
    ``modelops`` must call ``sha256_file`` imported by name from ``safe_io`` (tests patch it).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest

from fraud_detect.ml.retraining import modelops
from fraud_detect.ml.retraining.modelops import (
    BackupError, CandidateMismatchError, ModelOpsError, NoBackupError, SizeGuardRequired,
    restore, switch)
from fraud_detect.ml.retraining.safe_io import WriteGuardError, ensure_ignored, sha256_file

REPO = Path(__file__).resolve().parents[2]
NAME_6B = "statement_forgery_clf.joblib"
NAME_6A = "txn_fraud_clf.joblib"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, fake_current_models, is_ignored):
    cand = tmp_path / "cand" / "candidate.joblib"
    cand.parent.mkdir()
    cand.write_bytes(b"candidate model bytes\n" * 9)
    snapshot = {p.name: p.read_bytes() for p in fake_current_models.iterdir()}
    return dict(models=fake_current_models, backup=tmp_path / "backup", cand=cand,
                is_ignored=is_ignored, snapshot=snapshot)


def do_switch(env, node="6a", report_id="R-1", report_hash=None, **kw):
    return switch(node, report_id, models_dir=env["models"], backup_dir=env["backup"],
                  candidate_path=env["cand"],
                  report_candidate_sha256=report_hash or _sha(env["cand"]),
                  is_ignored=env["is_ignored"], **kw)


def current_bytes(env, name=NAME_6A):
    return (env["models"] / name).read_bytes()


# ---------- switch ----------

def test_switch_backs_up_verifies_records_then_replaces(env):
    cur_hash = _sha(env["models"] / NAME_6A)
    rec = do_switch(env)
    backup = env["backup"] / f"txn_fraud_clf.{cur_hash[:8]}.joblib"
    assert backup.is_file() and _sha(backup) == cur_hash
    assert rec.current_sha256 == cur_hash and rec.backup_sha256 == cur_hash
    assert rec.candidate_sha256 == _sha(env["cand"]) and rec.report_id == "R-1"
    assert current_bytes(env) == env["cand"].read_bytes()          # replaced
    index = json.loads((env["backup"] / "index.json").read_text())
    entry = index["6a"]
    assert entry["report_id"] == "R-1"
    assert entry["current_sha256"] == cur_hash and entry["backup_sha256"] == cur_hash
    assert entry["candidate_sha256"] == _sha(env["cand"])
    assert entry["backup_file"] == backup.name


def test_other_node_model_untouched(env):
    do_switch(env)
    assert current_bytes(env, NAME_6B) == env["snapshot"][NAME_6B]


def test_candidate_hash_must_match_report_and_nothing_changes(env):
    with pytest.raises(CandidateMismatchError):
        do_switch(env, report_hash="0" * 64)
    assert current_bytes(env) == env["snapshot"][NAME_6A]
    assert not (env["backup"] / "index.json").exists()


def test_failed_backup_write_stops_and_leaves_current_unchanged(env, tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file where the backup folder should be")
    env["backup"] = blocker
    with pytest.raises(BackupError):
        do_switch(env)
    assert current_bytes(env) == env["snapshot"][NAME_6A]


def test_backup_hash_mismatch_stops_and_leaves_current_unchanged(env, monkeypatch):
    real = sha256_file
    backup_dir = env["backup"]

    def lying(path):
        return "f" * 64 if backup_dir in Path(path).parents else real(path)

    monkeypatch.setattr(modelops, "sha256_file", lying)
    with pytest.raises(BackupError):
        do_switch(env)
    assert current_bytes(env) == env["snapshot"][NAME_6A]


def test_switch_refuses_backup_folder_that_is_not_ignored(env):
    env["is_ignored"].ignored = False
    with pytest.raises(WriteGuardError):
        do_switch(env)
    assert current_bytes(env) == env["snapshot"][NAME_6A]
    assert not env["backup"].exists()


def test_errors_share_a_base_class():
    for exc in (BackupError, NoBackupError, CandidateMismatchError, SizeGuardRequired):
        assert issubclass(exc, ModelOpsError)
    assert issubclass(SizeGuardRequired, ValueError)


def test_unknown_node_is_rejected(env):
    with pytest.raises(ValueError):
        do_switch(env, node="7z")


# ---------- restore ----------

def test_restore_copies_backup_back_and_verifies_hash(env):
    cur_hash = _sha(env["models"] / NAME_6A)
    do_switch(env)
    rec = restore("6a", models_dir=env["models"], backup_dir=env["backup"], is_ignored=env["is_ignored"])
    assert current_bytes(env) == env["snapshot"][NAME_6A]
    assert rec.restored_sha256 == rec.backup_sha256 == cur_hash
    assert _sha(env["models"] / NAME_6A) == cur_hash


def test_restore_with_no_backup_stops_and_leaves_current_unchanged(env):
    with pytest.raises(NoBackupError):
        restore("6a", models_dir=env["models"], backup_dir=env["backup"], is_ignored=env["is_ignored"])
    assert current_bytes(env) == env["snapshot"][NAME_6A]


def test_restore_for_node_without_backup_ignores_other_nodes_backups(env):
    do_switch(env, node="6a")
    with pytest.raises(NoBackupError):
        restore("6b", models_dir=env["models"], backup_dir=env["backup"], is_ignored=env["is_ignored"])
    assert current_bytes(env, NAME_6B) == env["snapshot"][NAME_6B]


# ---------- backup folder ----------

def test_default_backup_folder_is_models_backup_and_is_git_ignored():
    assert Path(modelops.DEFAULT_BACKUP_DIR) == Path("models/backup")
    ensure_ignored(REPO / modelops.DEFAULT_BACKUP_DIR)      # real git check-ignore, read-only


def test_model_file_names():
    assert modelops.MODEL_FILES == {"6b": NAME_6B, "6a": NAME_6A}


# ---------- Node 6b --size-guard ----------

def _bundle(path: Path, ranges):
    joblib.dump({"model": "stub", "size_ranges": ranges, "thresholds": {"medium": 0.5}}, path)


@pytest.fixture
def env6b(env):
    _bundle(env["models"] / NAME_6B, {"n_pages": [1.0, 4.0]})
    _bundle(env["cand"], {"n_pages": [1.0, 12.0]})
    env["snapshot"][NAME_6B] = (env["models"] / NAME_6B).read_bytes()
    return env


def test_6b_switch_requires_size_guard(env6b):
    with pytest.raises(SizeGuardRequired):
        do_switch(env6b, node="6b")
    with pytest.raises(SizeGuardRequired):
        do_switch(env6b, node="6b", size_guard="bogus")
    assert current_bytes(env6b, NAME_6B) == env6b["snapshot"][NAME_6B]
    assert not env6b["backup"].exists()


@pytest.mark.parametrize("choice,expected", [
    ("candidate", {"n_pages": [1.0, 12.0]}),
    ("off", None),
    ("current", {"n_pages": [1.0, 4.0]}),
])
def test_6b_size_guard_edits_only_the_installed_copy(env6b, choice, expected):
    cand_before = env6b["cand"].read_bytes()
    rec = do_switch(env6b, node="6b", size_guard=choice)
    installed = joblib.load(env6b["models"] / NAME_6B)
    assert installed["size_ranges"] == expected
    assert env6b["cand"].read_bytes() == cand_before             # the file in the work folder is never edited
    assert rec.installed_sha256 == _sha(env6b["models"] / NAME_6B)
    assert rec.candidate_sha256 == _sha(env6b["cand"])


def test_6a_switch_needs_no_size_guard(env):
    assert do_switch(env, node="6a").node == "6a"
