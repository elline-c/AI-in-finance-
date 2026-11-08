"""Task 19.3: property tests for the Size_Guard (Properties 31 and 32)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import size_guard as sg

FEATS = list(sg.SIZE_FEATURES)
_val = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
_pct = st.floats(0, 100, allow_nan=False)


def _frame(rows):
    return pd.DataFrame({f: [r[i] for r in rows] for i, f in enumerate(FEATS)}, columns=FEATS)


_rows = st.lists(st.tuples(*[_val for _ in FEATS]), min_size=0, max_size=40)
_rows1 = st.lists(st.tuples(*[_val for _ in FEATS]), min_size=1, max_size=40)
_ranges = st.fixed_dictionaries({
    f: st.tuples(_val, _val).map(lambda t: [min(t), max(t)]) for f in FEATS})


# Feature: ml-retraining, Property 31: Size_Guard arithmetic
@settings(max_examples=100, deadline=None)
@given(_rows, _rows, _ranges)
def test_outside_counts_equal_direct_counts(a, b, ranges):
    """**Validates: Requirements 12.1, 12.3**"""
    splits = {"train": _frame(a), "test": _frame(b)}
    out = sg.outside_summary(splits, ranges)
    for name, rows in (("train", a), ("test", b)):
        direct = sum(any(v < ranges[f][0] or v > ranges[f][1] for f, v in zip(FEATS, r))
                     for r in rows)
        assert out[name].count == direct
        assert out[name].total == len(rows)
        if rows:
            assert abs(out[name].percent - 100.0 * direct / len(rows)) < 1e-9
        else:
            assert out[name].percent is None


# Feature: ml-retraining, Property 31: Size_Guard arithmetic
@settings(max_examples=100, deadline=None)
@given(_rows1)
def test_candidate_ranges_are_training_quantiles(rows):
    """**Validates: Requirements 12.2**"""
    df = _frame(rows)
    r = sg.candidate_ranges(df)
    assert set(r) == set(FEATS)
    for f in FEATS:
        assert r[f] == [float(df[f].quantile(0.01)), float(df[f].quantile(0.99))]
        assert r[f][0] <= r[f][1]


# Feature: ml-retraining, Property 31: Size_Guard arithmetic
@settings(max_examples=100, deadline=None)
@given(_rows1, _ranges, st.data())
def test_guard_on_flagged_subset_and_abstained_share(rows, ranges, data):
    """**Validates: Requirements 12.3**"""
    outside = sg.outside_mask(_frame(rows), ranges)
    flagged = np.array(data.draw(st.lists(st.booleans(), min_size=len(rows), max_size=len(rows))))
    on = sg.apply_guard(flagged, outside)
    assert not (on & ~flagged).any()          # subset of guard-off flags
    assert not (on & outside).any()           # abstained documents never flagged
    assert (on == (flagged & ~outside)).all()
    assert abs(sg.abstained_share(outside) - 100.0 * outside.sum() / len(rows)) < 1e-9


_src = st.lists(st.sampled_from(["a", "b", "c", "d"]), unique=True, min_size=0, max_size=4)


@st.composite
def _evidence(draw):
    sources = draw(_src)
    return sg.GuardEvidence(
        cand_fa_12_month=draw(_pct), cand_fa_6_month=draw(_pct),
        fa_on={s: draw(_pct) for s in sources}, fa_off={s: draw(_pct) for s in sources},
        sources_with_30_genuine=sources,
        cand_outside_share=draw(_pct), cur_outside_share=draw(_pct))


# Feature: ml-retraining, Property 32: Size_Guard recommendation
@settings(max_examples=200, deadline=None)
@given(_evidence())
def test_recommendation_rules(ev):
    """**Validates: Requirements 12.4, 12.5, 12.6**"""
    rec = sg.recommend_guard(ev)
    assert rec.option in set(sg.GuardOption)
    gap = ev.cand_fa_12_month - ev.cand_fa_6_month
    off_ok = all(ev.fa_off[s] <= ev.fa_on[s] + 5.0 for s in ev.sources_with_30_genuine)
    if rec.option is sg.GuardOption.REMOVE_GUARD:
        assert gap <= 5.0 and off_ok
        assert rec.condition_failure is None
    else:
        assert gap > 5.0 or not off_ok
        assert rec.condition_failure is not None
        if ev.cand_outside_share < ev.cur_outside_share:
            assert rec.option is sg.GuardOption.CANDIDATE_GUARD
        else:
            assert rec.option is sg.GuardOption.KEEP_CURRENT_GUARD
    if gap <= 5.0 and off_ok:
        assert rec.option is sg.GuardOption.REMOVE_GUARD
    if gap > 5.0:
        assert f"{ev.cand_fa_12_month:.1f}" in rec.condition_failure
        assert f"{ev.cand_fa_6_month:.1f}" in rec.condition_failure
    # figures carry the evidence
    for k in ("cand_fa_12_month", "cand_fa_6_month", "fa_on", "fa_off",
              "cand_outside_share", "cur_outside_share"):
        assert k in rec.figures
    assert rec.figures["cand_fa_12_month"] == ev.cand_fa_12_month
    assert rec.figures["cand_fa_6_month"] == ev.cand_fa_6_month
