"""Task 13.1: failing tests for the freeze, the held-out vault and the post-test change log.

``fraud_detect/ml/retraining/vault.py`` does not exist yet (task 13.2). These tests pin its interface:

    CODE_FINGERPRINT_PATHS: tuple[str, ...]          # repo-relative files and folders
    code_fingerprint(root, paths=CODE_FINGERPRINT_PATHS) -> str          # SHA-256 hex
    unchanged_since_run_start(run_dir, root) -> bool
    unchanged_statement(run_dir, root) -> str
    write_run_json(run_dir, *, config, switch_rule_sha256, cutoff_rule, fingerprint,
                   is_ignored=None) -> str                                # returns SHA-256 of run.json
    write_freeze(run_dir, *, cutoffs, candidate_sha256, switch_rule_sha256, is_ignored=None) -> FreezeRecord
    FreezeRecord(run_hash, cutoffs, candidate_sha256, switch_rule_sha256)  # frozen dataclass
    check_classes(splits: Mapping[str, Sequence[int]])                     # 1 = tampered, 0 = genuine
    HeldOutVault(run_dir, loaders=None, is_ignored=None, clock=None).open() -> OpenVault
    OpenVault.freeze / .cutoffs(node) / .load(dataset_id) / .check_report_rule_hash(h)
    OpenVault.record_score(dataset_id, values) / .results() / .record_change(kind, description, trigger=None)
    exceptions: VaultError, VaultClosed(VaultError), MissingClassError(VaultError, ValueError),
                RuleHashMismatch(VaultError)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil

import pytest

from fraud_detect.ml.retraining import vault as vault_mod
from fraud_detect.ml.retraining.config import RunConfig
from fraud_detect.ml.retraining.vault import (
    CODE_FINGERPRINT_PATHS,
    FreezeRecord,
    HeldOutVault,
    MissingClassError,
    RuleHashMismatch,
    VaultClosed,
    VaultError,
    check_classes,
    code_fingerprint,
    unchanged_since_run_start,
    unchanged_statement,
    write_freeze,
    write_run_json,
)

RULE_HASH = "a" * 64
CUTOFF_RULE = {"target_fa_medium": 5.0, "target_fa_high": 1.0, "level": "document"}
CUTOFFS = {"6b": {"medium": 0.31, "high": 0.77}, "6a": {"medium": 0.42, "high": 0.88}}
CANDIDATES = {"6b": "b" * 64, "6a": "c" * 64}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_tree(root):
    """A tiny stand-in for the repository files named in the code fingerprint."""
    files = {
        "fraud_detect/corpus/gen.py": "A = 1\n",
        "fraud_detect/corpus/sub/more.py": "B = 2\n",
        "fraud_detect/nodes/node1.py": "C = 3\n",
        "fraud_detect/ml/statement_features.py": "D = 4\n",
        "fraud_detect/ml/transaction_features.py": "E = 5\n",
        "fraud_detect/registry.py": "F = 6\n",
        "fraud_detect/reference_data.py": "G = 7\n",
        "fraud_detect/unrelated.py": "H = 8\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


@pytest.fixture
def repo(tmp_path):
    return _make_tree(tmp_path / "repo")


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "runs" / "r1"


@pytest.fixture
def init_run(run_dir, repo, is_ignored):
    """Phase 0: run.json written with rule hash, cutoff rule and fingerprint."""
    run_hash = write_run_json(
        run_dir, config=RunConfig(), switch_rule_sha256=RULE_HASH, cutoff_rule=CUTOFF_RULE,
        fingerprint=code_fingerprint(repo), is_ignored=is_ignored)
    return run_hash


def _freeze(run_dir, is_ignored, **over):
    kw = dict(cutoffs=CUTOFFS, candidate_sha256=CANDIDATES, switch_rule_sha256=RULE_HASH,
              is_ignored=is_ignored)
    kw.update(over)
    return write_freeze(run_dir, **kw)


# ---------------------------------------------------------------- run.json


def test_run_json_holds_rule_hash_cutoff_rule_and_fingerprint(run_dir, repo, init_run):
    path = run_dir / "run.json"
    assert init_run == _sha(path)
    data = json.loads(path.read_text())
    assert data["switch_rule_sha256"] == RULE_HASH
    assert data["cutoff_rule"] == CUTOFF_RULE
    assert data["code_fingerprint"] == code_fingerprint(repo)
    assert json.loads(data["config"]) == json.loads(RunConfig().canonical_json())


def test_run_json_write_goes_through_the_write_guard(run_dir, repo, is_ignored):
    is_ignored.ignored = False
    with pytest.raises(Exception):
        write_run_json(run_dir, config=RunConfig(), switch_rule_sha256=RULE_HASH,
                       cutoff_rule=CUTOFF_RULE, fingerprint="x", is_ignored=is_ignored)
    assert not (run_dir / "run.json").exists()


# ---------------------------------------------------------------- freeze and open


def test_open_raises_when_frozen_json_missing(run_dir, init_run):
    with pytest.raises(VaultClosed):
        HeldOutVault(run_dir).open()


def test_open_raises_when_run_dir_missing(tmp_path):
    with pytest.raises(VaultClosed):
        HeldOutVault(tmp_path / "nope").open()


def test_open_raises_when_run_json_missing(run_dir, init_run, is_ignored):
    _freeze(run_dir, is_ignored)
    (run_dir / "run.json").unlink()
    with pytest.raises(VaultClosed):
        HeldOutVault(run_dir).open()


def test_open_succeeds_after_valid_freeze(run_dir, init_run, is_ignored):
    rec = _freeze(run_dir, is_ignored)
    assert isinstance(rec, FreezeRecord) and dataclasses.is_dataclass(rec)
    assert rec.run_hash == init_run
    data = json.loads((run_dir / "frozen.json").read_text())
    assert data["run_hash"] == init_run
    assert data["cutoffs"] == CUTOFFS
    assert data["candidate_sha256"] == CANDIDATES
    assert data["switch_rule_sha256"] == RULE_HASH
    opened = HeldOutVault(run_dir).open()
    assert opened.freeze == rec


def test_freeze_record_is_frozen(run_dir, init_run, is_ignored):
    rec = _freeze(run_dir, is_ignored)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.run_hash = "x"


def test_open_raises_when_run_json_changed_after_freeze(run_dir, init_run, is_ignored):
    _freeze(run_dir, is_ignored)
    (run_dir / "run.json").write_text((run_dir / "run.json").read_text() + " ")
    with pytest.raises(VaultClosed):
        HeldOutVault(run_dir).open()


def test_open_raises_when_recorded_run_hash_is_wrong(run_dir, init_run, is_ignored):
    _freeze(run_dir, is_ignored)
    data = json.loads((run_dir / "frozen.json").read_text())
    data["run_hash"] = "0" * 64
    (run_dir / "frozen.json").write_text(json.dumps(data))
    with pytest.raises(VaultClosed):
        HeldOutVault(run_dir).open()


def test_open_raises_on_unreadable_frozen_json(run_dir, init_run):
    (run_dir / "frozen.json").write_text("{not json")
    with pytest.raises(VaultClosed):
        HeldOutVault(run_dir).open()


def test_vault_closed_is_a_vault_error():
    assert issubclass(VaultClosed, VaultError)


def test_freeze_refuses_a_rule_hash_that_differs_from_run_json(run_dir, init_run, is_ignored):
    with pytest.raises(VaultError):
        _freeze(run_dir, is_ignored, switch_rule_sha256="d" * 64)
    assert not (run_dir / "frozen.json").exists()


def test_freeze_before_run_json_is_refused(run_dir, is_ignored):
    with pytest.raises(VaultError):
        _freeze(run_dir, is_ignored)
    assert not (run_dir / "frozen.json").exists()


def test_freeze_write_goes_through_the_write_guard(run_dir, init_run, is_ignored):
    is_ignored.ignored = False
    with pytest.raises(Exception):
        _freeze(run_dir, is_ignored)
    assert not (run_dir / "frozen.json").exists()


# ---------------------------------------------------------------- cutoffs and report hash


def test_held_out_scores_use_the_frozen_cutoffs(run_dir, init_run, is_ignored):
    _freeze(run_dir, is_ignored)
    opened = HeldOutVault(run_dir).open()
    for node, cut in CUTOFFS.items():
        got = opened.cutoffs(node)
        assert got.medium == cut["medium"]
        assert got.high == cut["high"]
    with pytest.raises(VaultError):
        opened.cutoffs("6z")


def test_cutoffs_come_from_the_file_not_from_memory(run_dir, init_run, is_ignored):
    """Editing frozen.json (leaving run_hash intact) is what the vault reports: no hidden cache."""
    _freeze(run_dir, is_ignored)
    data = json.loads((run_dir / "frozen.json").read_text())
    data["cutoffs"]["6b"]["medium"] = 0.5
    (run_dir / "frozen.json").write_text(json.dumps(data))
    assert HeldOutVault(run_dir).open().cutoffs("6b").medium == 0.5


def test_report_rule_hash_must_equal_run_json(run_dir, init_run, is_ignored):
    _freeze(run_dir, is_ignored)
    opened = HeldOutVault(run_dir).open()
    opened.check_report_rule_hash(RULE_HASH)  # no error
    for bad in ("e" * 64, "", RULE_HASH.upper(), RULE_HASH[:-1]):
        with pytest.raises(RuleHashMismatch):
            opened.check_report_rule_hash(bad)
    assert issubclass(RuleHashMismatch, VaultError)


# ---------------------------------------------------------------- datasets only through the vault


def test_datasets_are_only_reachable_through_the_open_vault(run_dir, init_run, is_ignored):
    calls = []
    loaders = {
        "test": lambda: calls.append("test") or ["t"],
        "heldout_seed": lambda: calls.append("heldout_seed") or ["h"],
    }
    vault = HeldOutVault(run_dir, loaders=loaders)
    # No public way to read a dataset from the closed vault.
    public = {n for n in dir(vault) if not n.startswith("_")}
    assert public == {"open"} or public <= {"open", "run_dir"}
    assert not any(hasattr(vault, n) for n in ("load", "datasets", "loaders", "get"))
    with pytest.raises(VaultClosed):
        vault.open()
    assert calls == []

    _freeze(run_dir, is_ignored)
    opened = vault.open()
    assert calls == []  # opening alone loads nothing
    assert opened.load("test") == ["t"]
    assert calls == ["test"]
    with pytest.raises(VaultError):
        opened.load("not_registered")


def test_module_exposes_no_other_loader_path():
    """Held-out data has one entry point; guard against a convenience bypass appearing."""
    assert not hasattr(vault_mod, "load_heldout")
    assert not hasattr(vault_mod, "load_dataset")


# ---------------------------------------------------------------- missing class


def test_check_classes_accepts_both_classes():
    check_classes({"train": [0, 1, 0], "validation": [1, 0]})


@pytest.mark.parametrize("split,labels,missing", [
    ("train", [], "genuine"),
    ("validation", [0, 0, 0], "tampered"),
    ("validation", [1, 1], "genuine"),
])
def test_check_classes_names_split_and_missing_class(split, labels, missing):
    splits = {"train": [0, 1], "validation": [0, 1]}
    splits[split] = labels
    with pytest.raises(MissingClassError) as err:
        check_classes(splits)
    assert split in str(err.value)
    assert missing in str(err.value)
    assert isinstance(err.value, ValueError) and isinstance(err.value, VaultError)


def test_missing_class_leaves_the_vault_unopened(run_dir, init_run):
    opened_loaders = []
    vault = HeldOutVault(run_dir, loaders={"test": lambda: opened_loaders.append(1)})
    with pytest.raises(MissingClassError):
        check_classes({"train": [0, 0], "validation": [0, 1]})
    assert not (run_dir / "frozen.json").exists()
    with pytest.raises(VaultClosed):
        vault.open()
    assert opened_loaders == []


# ---------------------------------------------------------------- code fingerprint


def test_fingerprint_paths_are_the_design_set():
    assert set(CODE_FINGERPRINT_PATHS) == {
        "fraud_detect/corpus", "fraud_detect/nodes",
        "fraud_detect/ml/statement_features.py", "fraud_detect/ml/transaction_features.py",
        "fraud_detect/registry.py", "fraud_detect/reference_data.py"}


def test_fingerprint_is_sha256_hex_and_stable(repo):
    fp = code_fingerprint(repo)
    assert len(fp) == 64 and int(fp, 16) >= 0
    assert code_fingerprint(repo) == fp


def test_fingerprint_equal_for_identical_trees_in_different_places(tmp_path, repo):
    other = tmp_path / "copy"
    shutil.copytree(repo, other)
    assert code_fingerprint(other) == code_fingerprint(repo)


@pytest.mark.parametrize("rel", [
    "fraud_detect/corpus/gen.py", "fraud_detect/corpus/sub/more.py", "fraud_detect/nodes/node1.py",
    "fraud_detect/ml/statement_features.py", "fraud_detect/ml/transaction_features.py",
    "fraud_detect/registry.py", "fraud_detect/reference_data.py"])
def test_fingerprint_differs_when_any_covered_file_changes(repo, rel):
    before = code_fingerprint(repo)
    p = repo / rel
    p.write_text(p.read_text() + "# edit\n")
    assert code_fingerprint(repo) != before


def test_fingerprint_differs_when_covered_file_added_removed_or_renamed(repo):
    base = code_fingerprint(repo)
    added = repo / "fraud_detect/nodes/node2.py"
    added.write_text("X = 1\n")
    with_added = code_fingerprint(repo)
    assert with_added != base
    added.unlink()
    assert code_fingerprint(repo) == base
    (repo / "fraud_detect/nodes/node1.py").rename(repo / "fraud_detect/nodes/node1b.py")
    assert code_fingerprint(repo) != base


def test_fingerprint_ignores_files_outside_the_covered_set(repo):
    base = code_fingerprint(repo)
    (repo / "fraud_detect/unrelated.py").write_text("changed\n")
    (repo / "fraud_detect/ml/other.py").write_text("new\n")
    assert code_fingerprint(repo) == base


def test_unchanged_statement_yes_then_no(run_dir, repo, init_run):
    assert unchanged_since_run_start(run_dir, repo) is True
    assert unchanged_statement(run_dir, repo) == (
        "generator and detector code unchanged since run start: yes")
    (repo / "fraud_detect/corpus/gen.py").write_text("A = 99\n")
    assert unchanged_since_run_start(run_dir, repo) is False
    assert unchanged_statement(run_dir, repo) == (
        "generator and detector code unchanged since run start: no")


# ---------------------------------------------------------------- post-test change log


@pytest.fixture
def opened(run_dir, init_run, is_ignored, clock):
    _freeze(run_dir, is_ignored)
    return HeldOutVault(run_dir, is_ignored=is_ignored, clock=clock).open()


def test_change_before_first_score_creates_no_mark_and_no_entry(opened, run_dir):
    assert opened.record_change("cutoff", "move medium cutoff") is None
    assert opened.results() == {}
    assert not (run_dir / "post_test_changes.json").exists()


def test_scores_start_as_held_out(opened):
    opened.record_score("test", {"auc": 0.91})
    res = opened.results()
    assert res["test"]["values"] == {"auc": 0.91}
    assert res["test"]["held_out"] == "held out"


def test_change_after_first_score_is_logged_and_marks_all_existing_results(opened, run_dir):
    opened.record_score("test", {"auc": 0.91})
    opened.record_score("heldout_seed", {"auc": 0.88})
    entry = opened.record_change("hyperparameter", "depth 3 to 4", trigger="auc 0.99 on test")
    assert entry["kind"] == "hyperparameter"
    assert entry["date"]
    assert entry["description"] == "depth 3 to 4"
    assert entry["trigger"] == "auc 0.99 on test"
    assert sorted(entry["affected"]) == ["heldout_seed", "test"]

    log = json.loads((run_dir / "post_test_changes.json").read_text())["changes"]
    assert [e["kind"] for e in log] == ["hyperparameter"]

    res = opened.results()
    for ds, auc in (("test", 0.91), ("heldout_seed", 0.88)):
        assert res[ds]["held_out"] == "no longer held out"
        assert res[ds]["values"] == {"auc": auc}  # original values kept


def test_marks_persist_across_reopening(opened, run_dir, is_ignored):
    opened.record_score("test", {"auc": 0.91})
    opened.record_change("feature", "drop a feature")
    again = HeldOutVault(run_dir, is_ignored=is_ignored).open()
    assert again.results()["test"]["held_out"] == "no longer held out"
    assert again.results()["test"]["values"] == {"auc": 0.91}


def test_score_recorded_after_a_change_is_not_marked_by_that_change(opened):
    opened.record_score("test", {"auc": 0.91})
    opened.record_change("cutoff", "new cutoff")
    opened.record_score("real_style", {"auc": 0.7})
    res = opened.results()
    assert res["test"]["held_out"] == "no longer held out"
    assert res["real_style"]["held_out"] == "held out"


def test_a_second_change_marks_results_added_since_and_keeps_both_entries(opened, run_dir):
    opened.record_score("test", {"auc": 0.91})
    opened.record_change("cutoff", "first")
    opened.record_score("real_style", {"auc": 0.7})
    entry = opened.record_change("generator", "second")
    assert entry["affected"] == ["real_style"] or sorted(entry["affected"]) == ["real_style", "test"]
    assert opened.results()["real_style"]["held_out"] == "no longer held out"
    log = json.loads((run_dir / "post_test_changes.json").read_text())["changes"]
    assert [e["description"] for e in log] == ["first", "second"]


@pytest.mark.parametrize("kind", ["cutoff", "hyperparameter", "feature", "generator", "detector"])
def test_design_change_kinds_are_accepted(opened, kind):
    opened.record_score("test", {"auc": 0.5})
    assert opened.record_change(kind, "x")["kind"] == kind


def test_unknown_change_kind_is_rejected(opened):
    opened.record_score("test", {"auc": 0.5})
    with pytest.raises(Exception):
        opened.record_change("vibes", "x")
    assert opened.results()["test"]["held_out"] == "held out"


def test_change_log_write_goes_through_the_write_guard(opened, is_ignored, run_dir):
    opened.record_score("test", {"auc": 0.5})
    is_ignored.ignored = False
    with pytest.raises(Exception):
        opened.record_change("cutoff", "x")
    assert not (run_dir / "post_test_changes.json").exists()
