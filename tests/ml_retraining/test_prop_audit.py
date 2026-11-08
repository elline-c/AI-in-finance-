"""Property tests for the Feature_Audit (task 8.4).

Property 11: Audit validation.
Property 13: The Creator-only rule.

**Validates: Requirements 3.1, 3.2, 3.8, 3.9**
"""
from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining.audit import (
    AUDIT_V1, AuditEntry, AuditError, FeatureAudit, creator_only_rule, validate,
)

_MODEL = [i for i, e in enumerate(AUDIT_V1.entries) if e.section == "model_feature"]
_DECISIONS = ("keep", "neutralise", "drop")


def _with(entries):
    return FeatureAudit(version="audit-test", entries=tuple(entries))


def _replace(idx, **changes):
    entries = list(AUDIT_V1.entries)
    entries[idx] = dataclasses.replace(entries[idx], **changes)
    return entries


@st.composite
def valid_audits(draw):
    """AUDIT_V1 with every decision redrawn; neutralise gets a recorded form."""
    entries = []
    for e in AUDIT_V1.entries:
        d = draw(st.sampled_from(_DECISIONS))
        form = "clip(0, 1)" if d == "neutralise" else None
        entries.append(dataclasses.replace(e, decision=d, neutralised_form=form))
    return _with(entries)


@settings(max_examples=50, deadline=None)
@given(valid_audits())
def test_valid_audit_passes(audit):
    assert validate(audit) is None


@settings(max_examples=50, deadline=None)
@given(st.sampled_from(_MODEL))
def test_missing_model_feature_fails_and_names_it(idx):
    entries = list(AUDIT_V1.entries)
    name = entries.pop(idx).name
    with pytest.raises(AuditError, match=name):
        validate(_with(entries))


@settings(max_examples=50, deadline=None)
@given(st.integers(0, len(AUDIT_V1.entries) - 1))
def test_duplicate_entry_fails_and_names_it(idx):
    entries = list(AUDIT_V1.entries)
    entries.append(entries[idx])
    with pytest.raises(AuditError, match=entries[idx].name):
        validate(_with(entries))


@settings(max_examples=50, deadline=None)
@given(st.integers(0, len(AUDIT_V1.entries) - 1),
       st.text(max_size=12).filter(lambda s: s not in _DECISIONS))
def test_unknown_decision_fails_and_names_it(idx, bad):
    with pytest.raises(AuditError, match=AUDIT_V1.entries[idx].name):
        validate(_with(_replace(idx, decision=bad)))


@settings(max_examples=50, deadline=None)
@given(st.integers(0, len(AUDIT_V1.entries) - 1),
       st.sampled_from(["name", "section", "node", "measures", "label_independent",
                        "real_file_could_differ", "decision", "note"]))
def test_missing_field_fails_and_names_it(idx, field):
    entries = _replace(idx, **{field: None})
    with pytest.raises(AuditError):
        validate(_with(entries))


@settings(max_examples=50, deadline=None)
@given(st.sampled_from(_MODEL), st.sampled_from([None, "", "   "]))
def test_neutralise_without_form_fails_and_names_it(idx, form):
    entries = _replace(idx, decision="neutralise", neutralised_form=form)
    with pytest.raises(AuditError, match=AUDIT_V1.entries[idx].name):
        validate(_with(entries))


_vals = st.lists(st.sampled_from(["a", "b", "c", "d", 0, 1]), max_size=6)


@settings(max_examples=200, deadline=None)
@given(_vals, _vals)
def test_creator_only_rule_matches_spec(forged, genuine):
    """keep iff both non-empty and the forged and genuine value sets differ (Req 3.8)."""
    expected = "keep" if (forged and genuine and
                          set(forged) != set(genuine)) else "drop"
    assert creator_only_rule(forged, genuine) == expected


@settings(max_examples=100, deadline=None)
@given(_vals)
def test_creator_only_rule_empty_side_drops(values):
    assert creator_only_rule([], values) == "drop"
    assert creator_only_rule(values, []) == "drop"
