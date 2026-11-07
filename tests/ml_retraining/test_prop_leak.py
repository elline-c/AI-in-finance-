"""Task 10.3: property tests for the Leak_Test (design Properties 14 and 15).

Tables are in memory only. A kept feature carries the signal; every other feature is constant.
"""
from __future__ import annotations

import pandas as pd
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import leak as lk
from fraud_detect.ml.retraining.audit import AUDIT_V1
from fraud_detect.ml.retraining.config import Thresholds

FEATURES_6B = (
    "n_pages", "n_text_objects", "n_distinct_fonts", "minor_font_share", "n_content_streams",
    "has_editing_software", "moddate_after_creation", "n_money_tokens",
    "balance_reconcile_breaks", "balance_reconcile_rate", "n_currency_styles",
)
F = "balance_reconcile_breaks"
N = 50
LIMIT = Thresholds().leak_auc
GAP = Thresholds().forgery_auc_gap

SETTINGS = settings(max_examples=40, deadline=None)


class _Log:
    """Minimal decision log: the Leak_Test only calls ``entries()``."""

    def __init__(self, entries=()):
        self._e = list(entries)

    def entries(self):
        return list(self._e)


def _table(rows):
    """rows: iterable of (y, layout, value of F)."""
    rows = list(rows)
    data = {name: [0.0] * len(rows) for name in FEATURES_6B}
    data[F] = [float(r[2]) for r in rows]
    data["y"] = [int(r[0]) for r in rows]
    data["layout"] = [r[1] for r in rows]
    return pd.DataFrame(data)


def _binary_leak(k, n=N):
    """F=1 for k of n tampered and for n-k of n genuine documents: single-feature AUC = k / n."""
    return _table([(1, "A", 1)] * k + [(1, "A", 0)] * (n - k)
                  + [(0, "A", 1)] * (n - k) + [(0, "A", 0)] * k)


def _run(train, valid, log=None, **kw):
    return lk.leak_test(train, valid, AUDIT_V1, Thresholds(), log, **kw)


def _row(res, name):
    return next(r for r in res.rows if r.feature == name)


# ---------------------------------------------------------------- Property 14
@SETTINGS
@given(k=st.integers(0, N), k_valid=st.integers(0, N))
def test_p14_deterministic_and_flags_at_threshold(k, k_valid):
    """**Validates: Requirements 4.2, 4.4, 4.5**"""
    train, valid = _binary_leak(k), _binary_leak(k_valid)
    a, b = _run(train, valid), _run(train, valid)
    assert [r.auc for r in a.rows] == [r.auc for r in b.rows]
    assert a.simple == b.simple

    names = {s.name: s for s in a.suspects}
    for r in a.rows:
        assert (r.feature in names) == (r.auc >= LIMIT)
    for model, auc in a.simple.items():
        assert (model in names) == (auc >= LIMIT)
    assert a.blocked == bool(a.suspects)
    assert a.exit_code == (3 if a.suspects else 0)


@SETTINGS
@given(k=st.integers(0, N))
def test_p14_suspect_line_carries_means_and_worst_layout(k):
    """**Validates: Requirements 4.4, 4.5**"""
    valid = _binary_leak(k)
    res = _run(valid, valid)
    for s in res.suspects:
        if s.kind != "feature":
            continue
        row = _row(res, s.name)
        assert s.auc == row.auc
        assert s.mean_genuine == valid.loc[valid["y"] == 0, s.name].mean()
        assert s.mean_tampered == valid.loc[valid["y"] == 1, s.name].mean()
        assert s.worst_layout == row.worst_layout
        computable = {l: a for l, a in row.per_layout.items() if a != lk.NOT_COMPUTABLE}
        if computable:
            far = max(abs(a - 0.5) for a in computable.values())
            assert abs(computable[s.worst_layout] - 0.5) == far


def test_p14_exactly_at_threshold_is_a_suspect():
    """**Validates: Requirements 4.4**  k=49 of 50 gives AUC exactly 0.98."""
    res = _run(_binary_leak(49), _binary_leak(49))
    assert _row(res, F).auc == 0.98
    assert F in {s.name for s in res.suspects}
    below = _run(_binary_leak(48), _binary_leak(48))
    assert F not in {s.name for s in below.suspects}


@SETTINGS
@given(decision=st.sampled_from(["exclude", "accept", "other", None]),
       target=st.sampled_from([F, "n_pages"]))
def test_p14_blocked_until_suspect_has_exclude_or_accept(decision, target):
    """**Validates: Requirements 4.5**"""
    tbl = _binary_leak(N)  # AUC 1.0 for F
    entries = [] if decision is None else [
        {"kind": "leak-feature", "target": target, "decision": decision}]
    res = _run(tbl, tbl, _Log(entries))
    resolved = decision in ("exclude", "accept") and target == F
    assert (F in {s.name for s in res.suspects}) == (not resolved)
    assert (F in {s.name for s in res.decided}) == resolved
    # the simple models see the same leak and carry no decision of their own
    assert res.blocked is True


# ---------------------------------------------------------------- Property 15
_counts = st.tuples(st.integers(0, 4), st.integers(0, 4))  # (genuine, tampered)


@SETTINGS
@given(layouts=st.dictionaries(st.sampled_from(["L1", "L2", "L3", "L4"]), _counts,
                               min_size=1, max_size=4),
       data=st.data())
def test_p15_not_computable_iff_fewer_than_two_per_class(layouts, data):
    """**Validates: Requirements 4.6**"""
    rows = []
    for layout, (g, t) in layouts.items():
        for y, count in ((0, g), (1, t)):
            for _ in range(count):
                rows.append((y, layout, data.draw(st.integers(0, 1))))
    valid = _table(rows + [(0, "Z", 0)] * 2 + [(1, "Z", 1)] * 2)  # keeps both classes present
    train = _binary_leak(40)
    res = _run(train, valid)
    for r in res.rows:
        for layout in valid["layout"].unique():
            sub = valid[valid["layout"] == layout]
            n1 = int(sub["y"].sum())
            n0 = len(sub) - n1
            assert (r.per_layout[layout] == lk.NOT_COMPUTABLE) == (n0 < 2 or n1 < 2)


@SETTINGS
@given(k_train=st.integers(0, N), k_valid=st.integers(0, N), k_forgery=st.integers(0, N))
def test_p15_forgery_flag_iff_gap_over_limit(k_train, k_valid, k_forgery):
    """**Validates: Requirements 4.8**"""
    res = _run(_binary_leak(k_train), _binary_leak(k_valid), forgery=_binary_leak(k_forgery))
    assert res.forgery == "compared"
    flagged = {g.feature: g for g in res.forgery_flagged}
    for r in res.rows:
        over = abs(r.auc - r.forgery_auc) > GAP
        assert (r.feature in flagged) == over
        if over:
            assert flagged[r.feature].generator_auc == r.auc
            assert flagged[r.feature].forgery_auc == r.forgery_auc


def test_p15_forgery_missing_is_skipped_with_notice():
    """**Validates: Requirements 4.8**"""
    res = _run(_binary_leak(10), _binary_leak(10))
    assert res.forgery == "skipped" and res.forgery_flagged == () and res.notice
