"""Task 13.3: property tests for the vault and change detection (design Properties 8, 9 and 10).

**Validates: Requirements 2.3, 2.4, 2.5, 2.7, 4.10, 5.7, 14.1**
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining.config import RunConfig
from fraud_detect.ml.retraining.vault import (
    CODE_FINGERPRINT_PATHS,
    HeldOutVault,
    MissingClassError,
    RuleHashMismatch,
    VaultClosed,
    check_classes,
    code_fingerprint,
    write_freeze,
    write_run_json,
)

SETTINGS = settings(max_examples=40, deadline=None)

CUTOFF_RULE = {"target_fa_medium": 5.0, "target_fa_high": 1.0, "level": "document"}
_hex = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
_cut = st.floats(0.0, 1.0, allow_nan=False)


def _always_ignored(_path) -> bool:
    return True


def _make_clock():
    state = {"n": 0}
    start = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)

    def now() -> datetime:
        state["n"] += 1
        return start + timedelta(seconds=state["n"])

    return now


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@st.composite
def _cutoffs(draw):
    out = {}
    for node in ("6a", "6b"):
        a, b = draw(_cut), draw(_cut)
        out[node] = {"medium": min(a, b), "high": max(a, b)}
    return out


def _init(run_dir: Path, rule_hash: str, fingerprint: str = "f" * 64) -> str:
    return write_run_json(run_dir, config=RunConfig(), switch_rule_sha256=rule_hash,
                          cutoff_rule=CUTOFF_RULE, fingerprint=fingerprint,
                          is_ignored=_always_ignored)


def _frozen_vault(run_dir: Path, rule_hash: str, cutoffs=None, loaders=None, clock=None):
    _init(run_dir, rule_hash)
    write_freeze(run_dir, cutoffs=cutoffs or {"6a": {"medium": 0.1, "high": 0.2},
                                              "6b": {"medium": 0.3, "high": 0.4}},
                 candidate_sha256={"6a": "1" * 64, "6b": "2" * 64},
                 switch_rule_sha256=rule_hash, is_ignored=_always_ignored)
    return HeldOutVault(run_dir, loaders=loaders, is_ignored=_always_ignored, clock=clock).open()


# ---------------------------------------------------------------- Property 8


_labels = st.lists(st.integers(0, 1), max_size=8)


@SETTINGS
@given(train=_labels, validation=_labels, test=_labels)
def test_property_8_missing_class_stops_before_the_vault_opens(train, validation, test):
    """Property 8: A missing class stops the run before cutoffs.

    **Validates: Requirements 2.7, 4.10**
    """
    splits = {"train": train, "validation": validation, "test": test}
    ok = all(0 in splits[n] and 1 in splits[n] for n in ("train", "validation"))
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        calls = []
        vault = HeldOutVault(run_dir, loaders={"test": lambda: calls.append(1)})
        if ok:
            check_classes(splits)
            return
        with pytest.raises(MissingClassError) as err:
            check_classes(splits)
        msg = str(err.value)
        # Names the first offending split (train is checked first) and the missing class.
        first = "train" if not (0 in train and 1 in train) else "validation"
        labels = splits[first]
        missing = "genuine" if 0 not in labels else "tampered"
        assert repr(first) in msg or first in msg
        assert missing in msg
        # The vault was never opened and nothing was loaded or written.
        with pytest.raises(VaultClosed):
            vault.open()
        assert calls == []
        assert not run_dir.exists()


@SETTINGS
@given(present=st.sets(st.sampled_from(["train", "validation"])))
def test_property_8_absent_split_counts_as_empty(present):
    """**Validates: Requirements 2.7, 4.10**"""
    splits = {n: [0, 1] for n in present}
    if present == {"train", "validation"}:
        check_classes(splits)
    else:
        with pytest.raises(MissingClassError):
            check_classes(splits)


# ---------------------------------------------------------------- Property 9


@SETTINGS
@given(
    run_present=st.booleans(),
    frozen_mode=st.sampled_from(["absent", "valid", "wrong_hash", "garbage"]),
    run_modified=st.booleans(),
    rule_hash=_hex,
    cutoffs=_cutoffs(),
)
def test_property_9_vault_opens_iff_frozen_hash_matches_run_json(
        run_present, frozen_mode, run_modified, rule_hash, cutoffs):
    """Property 9: opens iff frozen.json exists and its run hash equals the hash of run.json.

    **Validates: Requirements 2.3, 2.4, 14.1**
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        _init(run_dir, rule_hash)
        run_path = run_dir / "run.json"
        original_hash = _sha(run_path)
        frozen_path = run_dir / "frozen.json"
        if frozen_mode == "valid":
            write_freeze(run_dir, cutoffs=cutoffs, candidate_sha256={"6a": "1" * 64},
                         switch_rule_sha256=rule_hash, is_ignored=_always_ignored)
        elif frozen_mode == "wrong_hash":
            frozen_path.write_text(json.dumps({
                "run_hash": "0" * 64, "cutoffs": cutoffs, "candidate_sha256": {},
                "switch_rule_sha256": rule_hash}))
        elif frozen_mode == "garbage":
            frozen_path.write_text("{not json")
        if run_modified:
            run_path.write_text(run_path.read_text() + " ")
        if not run_present:
            run_path.unlink()

        expected_open = (
            frozen_mode != "absent" and run_present
            and frozen_mode == "valid" and not run_modified
            and original_hash == json.loads(frozen_path.read_text())["run_hash"])
        vault = HeldOutVault(run_dir)
        if expected_open:
            opened = vault.open()
            assert opened.freeze.run_hash == _sha(run_path)
        else:
            with pytest.raises(VaultClosed):
                vault.open()


@SETTINGS
@given(rule_hash=_hex, cutoffs=_cutoffs())
def test_property_9_reported_cutoffs_equal_frozen_values(rule_hash, cutoffs):
    """**Validates: Requirements 2.3, 2.4**"""
    with tempfile.TemporaryDirectory() as tmp:
        opened = _frozen_vault(Path(tmp) / "run", rule_hash, cutoffs=cutoffs)
        for node, cut in cutoffs.items():
            got = opened.cutoffs(node)
            assert (got.medium, got.high) == (cut["medium"], cut["high"])


@SETTINGS
@given(rule_hash=_hex, other=_hex)
def test_property_9_only_the_run_json_rule_hash_is_accepted(rule_hash, other):
    """**Validates: Requirements 2.5, 14.1**"""
    with tempfile.TemporaryDirectory() as tmp:
        opened = _frozen_vault(Path(tmp) / "run", rule_hash)
        opened.check_report_rule_hash(rule_hash)
        if other != rule_hash:
            with pytest.raises(RuleHashMismatch):
                opened.check_report_rule_hash(other)


@SETTINGS
@given(rule_hash=_hex, other=_hex)
def test_property_9_freeze_requires_the_recorded_rule_hash(rule_hash, other):
    """**Validates: Requirements 2.5, 14.1**"""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        _init(run_dir, rule_hash)
        if other == rule_hash:
            return
        with pytest.raises(Exception):
            write_freeze(run_dir, cutoffs={}, candidate_sha256={}, switch_rule_sha256=other,
                         is_ignored=_always_ignored)
        assert not (run_dir / "frozen.json").exists()


# ---------------------------------------------------------------- Property 10: change log

_kind = st.sampled_from(["cutoff", "hyperparameter", "feature", "generator", "detector"])
_dataset = st.sampled_from(["test", "heldout_seed", "real_style", "awkward_set", "forgery"])
_event = st.one_of(
    st.tuples(st.just("score"), _dataset, st.integers(0, 100)),
    st.tuples(st.just("change"), _kind, st.text(alphabet="abc xyz", min_size=1, max_size=8)),
)


@SETTINGS
@given(events=st.lists(_event, max_size=14))
def test_property_10_post_test_changes_are_recorded_and_marked(events):
    """Property 10 (change part): compare the vault with a small model of the rules.

    **Validates: Requirements 2.5, 5.7**
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        opened = _frozen_vault(run_dir, "a" * 64, clock=_make_clock())
        model = {}          # dataset -> (value, held_out flag)
        log = []            # expected log: (kind, description, affected)
        for ev in events:
            if ev[0] == "score":
                _, ds, val = ev
                opened.record_score(ds, {"v": val})
                model[ds] = (val, True)
            else:
                _, kind, desc = ev
                entry = opened.record_change(kind, desc)
                if not model:
                    assert entry is None
                    continue
                affected = sorted(d for d, (_, held) in model.items() if held)
                assert entry["kind"] == kind and entry["description"] == desc
                assert entry["date"]
                assert sorted(entry["affected"]) == affected
                log.append((kind, desc))
                model = {d: (v, False) for d, (v, _) in model.items()}
        res = opened.results()
        assert set(res) == set(model)
        for ds, (val, held) in model.items():
            assert res[ds]["values"] == {"v": val}  # original values kept
            assert res[ds]["held_out"] == ("held out" if held else "no longer held out")
        log_path = run_dir / "post_test_changes.json"
        if log:
            written = json.loads(log_path.read_text())["changes"]
            assert [(e["kind"], e["description"]) for e in written] == log
            assert all(e["date"] for e in written)
        else:
            assert not log_path.exists()


@SETTINGS
@given(kinds=st.lists(_kind, min_size=1, max_size=5))
def test_property_10_changes_before_the_first_score_leave_no_trace(kinds):
    """**Validates: Requirements 2.5**"""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        opened = _frozen_vault(run_dir, "a" * 64)
        for kind in kinds:
            assert opened.record_change(kind, "early") is None
        assert opened.results() == {}
        assert not (run_dir / "post_test_changes.json").exists()


# ---------------------------------------------------------------- Property 10: fingerprint

_COVERED = [
    "fraud_detect/corpus/a.py", "fraud_detect/corpus/sub/b.py", "fraud_detect/nodes/n1.py",
    "fraud_detect/nodes/n2.py", "fraud_detect/ml/statement_features.py",
    "fraud_detect/ml/transaction_features.py", "fraud_detect/registry.py",
    "fraud_detect/reference_data.py",
]
_OUTSIDE = ["fraud_detect/unrelated.py", "fraud_detect/ml/other.py", "README.md"]
_content = st.text(alphabet="abcxyz=1\n ", max_size=12)
_tree = st.dictionaries(st.sampled_from(_COVERED), _content, max_size=len(_COVERED))


def _write_tree(root: Path, tree: dict) -> None:
    for rel, text in tree.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@SETTINGS
@given(a=_tree, b=_tree)
def test_property_10_fingerprint_differs_iff_some_file_differs(a, b):
    """Property 10 (fingerprint part): equal iff the covered file sets and contents are equal.

    **Validates: Requirements 5.7, 14.1**
    """
    with tempfile.TemporaryDirectory() as tmp:
        ra, rb = Path(tmp) / "a", Path(tmp) / "b"
        ra.mkdir()
        rb.mkdir()
        _write_tree(ra, a)
        _write_tree(rb, b)
        assert (code_fingerprint(ra) == code_fingerprint(rb)) == (a == b)


@SETTINGS
@given(tree=_tree, outside=st.dictionaries(st.sampled_from(_OUTSIDE), _content))
def test_property_10_fingerprint_ignores_uncovered_files(tree, outside):
    """**Validates: Requirements 5.7**"""
    with tempfile.TemporaryDirectory() as tmp:
        ra, rb = Path(tmp) / "a", Path(tmp) / "b"
        ra.mkdir()
        rb.mkdir()
        _write_tree(ra, tree)
        _write_tree(rb, tree)
        _write_tree(rb, outside)
        assert code_fingerprint(ra) == code_fingerprint(rb)


def test_fingerprint_paths_cover_the_design_set():
    assert len(CODE_FINGERPRINT_PATHS) == 6
