"""Task 10.1: failing tests for ``fraud_detect.ml.retraining.leak`` (Leak_Test, Requirement 4).

Pinned interface (see the task summary):

    leak_test(train, valid, audit, thresholds, decisions, forgery=None, node="6b") -> LeakResult

``train``, ``valid`` and ``forgery`` are pandas DataFrames with one column per model feature of the
node (all 11 Node 6b features, dropped ones included), a 0/1 column ``y`` (1 = tampered) and, for
``train`` and ``valid``, a ``layout`` column. ``decisions`` is a ``DecisionLog`` (or None). Tables are
in memory only: no file is read or written.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fraud_detect.ml.retraining import leak as lk
from fraud_detect.ml.retraining.audit import AUDIT_V1
from fraud_detect.ml.retraining.config import Thresholds
from fraud_detect.ml.retraining.decisions import DecisionLog

FEATURES_6B = (
    "n_pages", "n_text_objects", "n_distinct_fonts", "minor_font_share", "n_content_streams",
    "has_editing_software", "moddate_after_creation", "n_money_tokens",
    "balance_reconcile_breaks", "balance_reconcile_rate", "n_currency_styles",
)
F = "balance_reconcile_breaks"  # a kept feature used as the signal carrier


# ----------------------------------------------------------------------------- helpers
def _table(rows, **extra):
    """rows: list of (y, layout, value of F). Every other feature is constant 0."""
    data = {name: [0.0] * len(rows) for name in FEATURES_6B}
    data[F] = [float(r[2]) for r in rows]
    data["y"] = [int(r[0]) for r in rows]
    data["layout"] = [r[1] for r in rows]
    data.update(extra)
    return pd.DataFrame(data)


def _separable(n=25, layouts=("A", "B")):
    """F equals the label in every layout: a perfect single-feature leak."""
    rows = []
    for layout in layouts:
        rows += [(0, layout, 0)] * n + [(1, layout, 1)] * n
    return _table(rows)


def _clean(n=25, layouts=("A", "B")):
    """No feature carries the label (all constant)."""
    rows = []
    for layout in layouts:
        rows += [(0, layout, 0)] * n + [(1, layout, 0)] * n
    return _table(rows)


def _binary_leak(p_tampered_one, n=50):
    """F=1 for p_tampered_one of n tampered and for (n - p_tampered_one) of n genuine docs.

    Single-feature AUC = 0.5 + (p1 - p0) / 2 with p1 = p_tampered_one / n, p0 = 1 - p1.
    """
    rows = ([(1, "A", 1)] * p_tampered_one + [(1, "A", 0)] * (n - p_tampered_one)
            + [(0, "A", 1)] * (n - p_tampered_one) + [(0, "A", 0)] * p_tampered_one)
    return _table(rows)


@pytest.fixture
def log(tmp_path, is_ignored, clock):
    return DecisionLog(tmp_path / "decisions.json", is_ignored=is_ignored, clock=clock)


def _run(train, valid, log=None, thresholds=None, **kw):
    return lk.leak_test(train, valid, AUDIT_V1, thresholds or Thresholds(), log, **kw)


def _row(result, name):
    return next(r for r in result.rows if r.feature == name)


# ------------------------------------------------------------------- shape of the result
def test_every_node_feature_is_reported_in_audit_order_joined_to_the_audit():
    res = _run(_separable(), _separable())
    assert [r.feature for r in res.rows] == list(FEATURES_6B)  # dropped ones included (4.2)
    assert _row(res, "n_pages").audit_decision == "drop"
    assert _row(res, F).audit_decision == "keep"
    assert _row(res, "minor_font_share").audit_decision == "neutralise"  # 3.6: audit row joined


def test_single_feature_auc_is_depth2_tree_fitted_on_train_scored_on_validation():
    res = _run(_separable(), _separable())
    assert _row(res, F).auc == pytest.approx(1.0)
    assert _row(res, "n_pages").auc == pytest.approx(0.5)  # constant feature: ties only
    assert _row(res, F).mean_genuine == 0.0
    assert _row(res, F).mean_tampered == 1.0


def test_auc_direction_is_learned_on_train_and_low_auc_is_reported_not_flagged():
    inverted = _table([(0, "A", 1)] * 25 + [(1, "A", 0)] * 25)  # valid: tampered has lower F
    res = _run(_separable(layouts=("A",)), inverted)
    assert _row(res, F).auc == pytest.approx(0.0)  # reported as is (design item I7)
    assert F not in [s.name for s in res.suspects]


def test_two_runs_give_equal_aucs_to_four_decimal_places():
    train, valid = _separable(), _binary_leak(40)
    a, b = _run(train, valid), _run(train, valid)
    assert [round(r.auc, 4) for r in a.rows] == [round(r.auc, 4) for r in b.rows]
    assert {k: round(v, 4) for k, v in a.simple.items()} == {k: round(v, 4) for k, v in b.simple.items()}


def test_leak_test_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(_separable(), _separable())
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------------ simple models
def test_logistic_and_depth3_tree_are_reported():
    res = _run(_separable(), _separable())
    assert set(res.simple) == {"logistic", "tree3"}
    assert res.simple["logistic"] == pytest.approx(1.0)
    assert res.simple["tree3"] == pytest.approx(1.0)


def test_simple_model_alone_can_be_the_suspect():
    # y = a xor b: no single feature separates, a depth-3 tree does.
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            rows += [(a ^ b, "A", a, b)] * 25
    def build():
        t = _table([(y, l, a) for y, l, a, _ in rows])
        t["moddate_after_creation"] = [float(r[3]) for r in rows]
        return t
    res = _run(build(), build())
    assert all(r.auc < 0.98 for r in res.rows)
    assert [(s.name, s.kind) for s in res.suspects] == [("tree3", "model")]
    assert res.exit_code == 3


# ------------------------------------------------------------------ threshold and blocking
def test_feature_exactly_at_threshold_is_a_suspect_and_just_below_is_not():
    at = _binary_leak(49)       # AUC 0.98
    below = _binary_leak(48)    # AUC 0.96
    res = _run(at, at)
    assert round(_row(res, F).auc, 4) == 0.98
    assert F in [s.name for s in res.suspects]  # >= threshold, compared on the 4 d.p. value
    res = _run(below, below)
    assert round(_row(res, F).auc, 4) == 0.96
    assert F not in [s.name for s in res.suspects]
    assert res.exit_code == 0


def test_suspect_line_has_name_auc_means_and_furthest_layout():
    # Validation: layout A follows F, layout B has F constant, so A is furthest from 0.5.
    train = _separable()
    valid = _table([(0, "A", 0)] * 20 + [(1, "A", 1)] * 20 + [(0, "B", 0)] * 20 + [(1, "B", 0)] * 20)
    res = _run(train, valid, thresholds=Thresholds(leak_auc=0.7))
    s = next(s for s in res.suspects if s.name == F)
    assert s.kind == "feature"
    assert s.auc == pytest.approx(0.75)
    assert s.mean_genuine == pytest.approx(0.0)
    assert s.mean_tampered == pytest.approx(0.5)
    assert s.worst_layout == "A"
    row = _row(res, F)
    assert row.per_layout["A"] == pytest.approx(1.0)
    assert row.per_layout["B"] == pytest.approx(0.5)
    assert row.worst_layout == "A"


def test_undecided_suspect_blocks_with_exit_3_and_no_model_file(log, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _run(_separable(), _separable(), log)
    assert res.blocked is True
    assert res.exit_code == 3
    assert F in [s.name for s in res.suspects]
    assert list(p for p in tmp_path.rglob("*") if p.suffix in (".joblib", ".pkl")) == []


@pytest.mark.parametrize("decision", ["exclude", "accept"])
def test_recorded_decision_for_every_suspect_unblocks(log, decision):
    first = _run(_separable(), _separable(), log)
    for s in first.suspects:
        log.append("leak-feature", target=s.name, decision=decision, reason="r", author="a")
    res = _run(_separable(), _separable(), log)
    assert res.suspects == ()
    assert res.blocked is False and res.exit_code == 0
    assert {d.name for d in res.decided} == {s.name for s in first.suspects}


def test_one_undecided_suspect_among_decided_still_blocks(log):
    first = _run(_separable(), _separable(), log)
    assert len(first.suspects) >= 2  # the feature and at least one simple model
    log.append("leak-feature", target=first.suspects[0].name, decision="accept", reason="r", author="a")
    res = _run(_separable(), _separable(), log)
    assert res.blocked and res.exit_code == 3
    assert first.suspects[0].name not in [s.name for s in res.suspects]


def test_decision_of_another_kind_or_target_does_not_unblock(log):
    log.append("waive-gate", target=F, decision="waive", reason="r", author="a")
    log.append("leak-feature", target="n_pages", decision="accept", reason="r", author="a")
    assert _run(_separable(), _separable(), log).blocked


# ------------------------------------------------------------------------------- per layout
def test_per_layout_auc_is_not_computable_with_fewer_than_two_of_a_class():
    valid = _table(
        [(0, "A", 0)] * 10 + [(1, "A", 1)] * 10
        + [(0, "C", 0)] * 1 + [(1, "C", 1)] * 5     # one genuine
        + [(0, "D", 0)] * 5 + [(1, "D", 1)] * 1     # one tampered
        + [(0, "E", 0)] * 2 + [(1, "E", 1)] * 2)    # exactly two of each: computable
    res = _run(_separable(layouts=("A", "C", "D", "E")), valid)
    pl = _row(res, F).per_layout
    assert pl["C"] == lk.NOT_COMPUTABLE
    assert pl["D"] == lk.NOT_COMPUTABLE
    assert pl["A"] == pytest.approx(1.0)
    assert pl["E"] == pytest.approx(1.0)
    assert set(pl) == {"A", "C", "D", "E"}  # one entry for each layout in validation


# --------------------------------------------------------------------------- bad splits
@pytest.mark.parametrize("which,split,missing", [
    ("train_empty", "train", None),
    ("valid_empty", "validation", None),
    ("train_one_class", "train", "tampered"),
    ("valid_one_class", "validation", "genuine"),
])
def test_empty_or_single_class_split_stops_with_exit_2_naming_split_and_class(which, split, missing):
    good = _separable()
    empty = good.iloc[0:0]
    genuine_only = good[good.y == 0]
    tampered_only = good[good.y == 1]
    train, valid = good, good
    if which == "train_empty":
        train = empty
    elif which == "valid_empty":
        valid = empty
    elif which == "train_one_class":
        train = genuine_only
    else:
        valid = tampered_only
    with pytest.raises(lk.LeakTestError) as exc:
        _run(train, valid)
    assert exc.value.exit_code == 2
    assert exc.value.split == split
    assert split in str(exc.value)
    if missing:
        assert exc.value.missing_class == missing
        assert missing in str(exc.value)


# -------------------------------------------------------------------------- Forgery_Dataset
def test_forgery_comparison_lists_features_with_gap_above_020_with_both_values():
    forgery = _table([(0, "X", 0)] * 20 + [(1, "X", 0)] * 20)  # F is constant there: AUC 0.5
    res = _run(_separable(), _separable(), forgery=forgery, thresholds=Thresholds(leak_auc=2.0))
    assert [g.feature for g in res.forgery_flagged] == [F]
    gap = res.forgery_flagged[0]
    assert gap.generator_auc == pytest.approx(1.0)
    assert gap.forgery_auc == pytest.approx(0.5)


def test_forgery_gap_not_above_020_is_not_listed():
    # Forgery AUC for F is 0.9 (gap 0.1 against the generator's 1.0).
    forgery = _table([(1, "X", 1)] * 18 + [(1, "X", 0)] * 2 + [(0, "X", 0)] * 18 + [(0, "X", 1)] * 2)
    res = _run(_separable(), _separable(), forgery=forgery, thresholds=Thresholds(leak_auc=2.0))
    assert _row(res, F).forgery_auc == pytest.approx(0.9)
    assert res.forgery_flagged == ()


def test_missing_forgery_dataset_prints_a_notice_and_the_test_continues():
    res = _run(_separable(), _separable(), forgery=None)
    assert res.forgery == "skipped"
    assert "skipped" in res.notice.lower() and "forgery" in res.notice.lower()
    assert res.forgery_flagged == ()
    assert len(res.rows) == len(FEATURES_6B)  # the rest of the test ran
