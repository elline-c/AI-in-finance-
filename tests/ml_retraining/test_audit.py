"""Tests for the Feature_Audit (task 8.1; written before the module exists).

Expected API of ``fraud_detect.ml.retraining.audit`` (task 8.3):

* ``AuditEntry`` / ``FeatureAudit`` frozen dataclasses (fields as in the design)
* ``AUDIT_V1``: the v1 ``FeatureAudit`` literal (``version == "audit-v1"``)
* ``AuditError`` (an ``Exception``; its message names the affected feature)
* ``validate(audit)``: raises ``AuditError`` on any defect, returns ``None`` otherwise
* ``FeatureAudit.digest()``: SHA-256 hex of the canonical JSON
* ``model_input_columns(audit, node)``: ordered non-dropped names, ``node`` is "6a" or "6b"
* ``creator_only_rule(forged_values, genuine_values)``: returns "keep" or "drop"

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 3.9**
"""
from __future__ import annotations

import dataclasses
import importlib
import re

import pytest

from fraud_detect.ml.statement_features import FEATURE_NAMES
from fraud_detect.ml.transaction_features import PDF_FEATURE_COLUMNS

DECISIONS = {"keep", "neutralise", "drop"}

# Requirement 3.3: the 12 candidate Generator_Tells, by entry name.
CANDIDATE_TELLS = (
    "producer_string", "creator_string", "font_names", "font_count",
    "page_width_height", "page_count", "round_amount_share", "row_order",
    "row_spacing", "creation_time_stamp", "modification_time_stamp", "footer_wording",
)

# Pinned digest per audit version. A change to the literal without a version bump
# (and a new pin) must fail ``test_digest_is_pinned_per_version``.
# Task 8.3 pins the real value when it writes the literal.
PINNED_DIGESTS = {"audit-v1": "7a34d1ac3a8da1283e3dc546eb02fabcc58a7512df74a6f8d7a1da0520e5b5eb"}

NODE_6B_KEPT = {"has_editing_software", "moddate_after_creation", "balance_reconcile_breaks"}
NODE_6B_NEUTRALISED = {"minor_font_share"}
NODE_6B_DROPPED = {
    "n_pages", "n_text_objects", "n_distinct_fonts", "n_content_streams",
    "n_money_tokens", "balance_reconcile_rate", "n_currency_styles",
}
NODE_6A_DROPPED = {"balance_after_sgd", "day_of_week"}


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module("fraud_detect.ml.retraining.audit")


@pytest.fixture(scope="module")
def v1(mod):
    return mod.AUDIT_V1


def _model_entries(audit):
    return [e for e in audit.entries if e.section == "model_feature"]


def _by_name(audit, name, section="model_feature"):
    found = [e for e in audit.entries if e.name == name and e.section == section]
    assert len(found) == 1, f"expected exactly one {section} entry for {name}"
    return found[0]


def _replace_entry(audit, name, **changes):
    entries = tuple(
        dataclasses.replace(e, **changes) if e.name == name and e.section == "model_feature" else e
        for e in audit.entries)
    return dataclasses.replace(audit, entries=entries)


# ---------------------------------------------------------------- structure (3.1, 3.3)

def test_version_label(v1):
    assert v1.version == "audit-v1"


def test_one_model_feature_entry_per_feature(v1):
    names_6b = [e.name for e in _model_entries(v1) if e.node == "6b"]
    names_6a = [e.name for e in _model_entries(v1) if e.node == "6a"]
    assert sorted(names_6b) == sorted(FEATURE_NAMES) and len(names_6b) == 11
    assert sorted(names_6a) == sorted(PDF_FEATURE_COLUMNS) and len(names_6a) == 13
    assert len(_model_entries(v1)) == 24


def test_twelve_candidate_tells(v1):
    tells = [e for e in v1.entries if e.section == "candidate_tell"]
    assert sorted(e.name for e in tells) == sorted(CANDIDATE_TELLS)
    assert len(tells) == 12
    assert all(e.decision == "drop" for e in tells)


def test_no_duplicate_entries(v1):
    keys = [(e.section, e.node, e.name) for e in v1.entries]
    assert len(keys) == len(set(keys))
    assert len(v1.entries) == 36


# ---------------------------------------------------------------- fields (3.2)

def test_every_field_present_and_valid(v1):
    for e in v1.entries:
        assert isinstance(e.measures, str) and e.measures.strip(), e.name
        assert e.label_independent in {"yes", "no"}, e.name
        assert e.real_file_could_differ in {"yes", "no"}, e.name
        assert e.decision in DECISIONS, e.name
        assert isinstance(e.note, str), e.name
        if e.decision == "neutralise":
            assert e.neutralised_form and e.neutralised_form.strip(), e.name


def test_neutralised_form_recorded(v1):
    entry = _by_name(v1, "minor_font_share")
    assert entry.decision == "neutralise"
    assert entry.neutralised_form == "clip(0, 0.10)"


# ---------------------------------------------------------------- v1 contents

def test_v1_node_6b_decisions(v1):
    got = {e.name: e.decision for e in _model_entries(v1) if e.node == "6b"}
    assert {n for n, d in got.items() if d == "keep"} == NODE_6B_KEPT
    assert {n for n, d in got.items() if d == "neutralise"} == NODE_6B_NEUTRALISED
    assert {n for n, d in got.items() if d == "drop"} == NODE_6B_DROPPED
    assert sum(d == "keep" for d in got.values()) == 3
    assert sum(d == "neutralise" for d in got.values()) == 1
    assert sum(d == "drop" for d in got.values()) == 7


def test_v1_node_6a_dropped(v1):
    got = {e.name: e.decision for e in _model_entries(v1) if e.node == "6a"}
    assert {n for n, d in got.items() if d == "drop"} == NODE_6A_DROPPED
    assert all(d == "keep" for n, d in got.items() if n not in NODE_6A_DROPPED)


def test_has_editing_software_uses_forgery_gate(v1):
    assert _by_name(v1, "has_editing_software").decision_rule == "forgery_gate"


# ---------------------------------------------------------------- model input (3.4)

def test_model_input_columns_exclude_dropped(mod, v1):
    cols_6b = mod.model_input_columns(v1, "6b")
    assert cols_6b == [n for n in FEATURE_NAMES if n in NODE_6B_KEPT | NODE_6B_NEUTRALISED]
    assert len(cols_6b) == 4
    cols_6a = mod.model_input_columns(v1, "6a")
    assert cols_6a == [n for n in PDF_FEATURE_COLUMNS if n not in NODE_6A_DROPPED]
    assert not set(cols_6a) & NODE_6A_DROPPED


def test_model_input_follows_a_changed_decision(mod, v1):
    changed = _replace_entry(v1, "n_pages", decision="keep")
    assert "n_pages" in mod.model_input_columns(changed, "6b")


# ---------------------------------------------------------------- validation (3.9)

def test_v1_validates(mod, v1):
    assert mod.validate(v1) is None


def test_missing_feature_is_named(mod, v1):
    entries = tuple(e for e in v1.entries if not (e.name == "n_pages" and e.section == "model_feature"))
    with pytest.raises(mod.AuditError, match="n_pages"):
        mod.validate(dataclasses.replace(v1, entries=entries))


def test_duplicate_feature_is_named(mod, v1):
    dup = _by_name(v1, "n_text_objects")
    with pytest.raises(mod.AuditError, match="n_text_objects"):
        mod.validate(dataclasses.replace(v1, entries=v1.entries + (dup,)))


def test_unknown_decision_is_named(mod, v1):
    bad = _replace_entry(v1, "log_amount", decision="maybe")
    with pytest.raises(mod.AuditError, match="log_amount"):
        mod.validate(bad)


@pytest.mark.parametrize("field", ["measures", "label_independent", "real_file_could_differ", "decision"])
def test_missing_field_is_named(mod, v1, field):
    bad = _replace_entry(v1, "n_txn_7d", **{field: None})
    with pytest.raises(mod.AuditError, match="n_txn_7d"):
        mod.validate(bad)


def test_neutralise_without_form_is_named(mod, v1):
    bad = _replace_entry(v1, "minor_font_share", neutralised_form=None)
    with pytest.raises(mod.AuditError, match="minor_font_share"):
        mod.validate(bad)


def test_missing_candidate_tell_is_named(mod, v1):
    entries = tuple(e for e in v1.entries if e.name != "footer_wording")
    with pytest.raises(mod.AuditError, match="footer_wording"):
        mod.validate(dataclasses.replace(v1, entries=entries))


# ---------------------------------------------------------------- digest (versioning)

def test_digest_is_stable_sha256(v1):
    d = v1.digest()
    assert re.fullmatch(r"[0-9a-f]{64}", d)
    assert d == v1.digest()


def test_digest_changes_with_content(v1):
    changed = _replace_entry(v1, "n_pages", decision="keep")
    assert changed.digest() != v1.digest()


def test_digest_is_pinned_per_version(v1):
    pinned = PINNED_DIGESTS.get(v1.version)
    assert pinned is not None, "bump the version and pin a new digest"
    assert re.fullmatch(r"[0-9a-f]{64}", pinned), "pin the real SHA-256 digest of audit-v1"
    assert v1.digest() == pinned, "audit literal changed without a version bump"


# ---------------------------------------------------------------- Requirement 3.8 rule

@pytest.mark.parametrize("forged, genuine, expected", [
    ([1], [0], "keep"),
    ([0, 1], [0], "keep"),
    ([0, 0], [0, 0], "drop"),
    ([], [0], "drop"),
    ([1], [], "drop"),
    ([], [], "drop"),
    (["a", "b"], ["a"], "keep"),
])
def test_creator_only_rule(mod, forged, genuine, expected):
    assert mod.creator_only_rule(forged, genuine) == expected
