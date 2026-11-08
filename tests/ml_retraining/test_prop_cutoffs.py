"""Task 11.4: property tests for cutoffs (design Properties 6 and 7).

Deviation from the design wording: cutoffs.py allows at least one genuine
document, so the effective limit is ``max(target, 100 / n_genuine)`` rather
than the bare target. The properties below model that limit; the "target not
met" failure therefore occurs only when even the highest genuine score flags
more genuine documents than the effective limit (ties at the top).
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import cutoffs as c

# Coarse score grid so ties are common.
_score = st.integers(0, 40).map(lambda i: i / 40.0)
_target = st.floats(0.0, 100.0, allow_nan=False)


@st.composite
def _labelled(draw, max_n=60):
    n = draw(st.integers(2, max_n))
    scores = draw(st.lists(_score, min_size=n, max_size=n))
    flags = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    if all(flags):
        flags[0] = False
    if not any(flags):
        flags[-1] = True
    return np.array(scores), np.array(flags, dtype=bool)


def _fa(genuine, cut):
    return 100.0 * float(np.sum(genuine >= cut)) / len(genuine)


def _limit(genuine, target):
    return max(target, 100.0 / len(genuine))


@settings(max_examples=200, deadline=None)
@given(_labelled(), _target, _target)
def test_property_6_cutoffs_meet_target_and_are_lowest(data, t_med, t_high):
    """**Validates: Requirements 2.1, 2.2, 2.8**"""
    scores, is_g = data
    genuine = scores[is_g]
    rule = c.CutoffRule(target_fa_medium=t_med, target_fa_high=t_high)
    top_fa = _fa(genuine, genuine.max())
    expected_fail = None
    if top_fa > _limit(genuine, t_med):
        expected_fail = "medium"
    elif top_fa > _limit(genuine, t_high):
        expected_fail = "high"

    if expected_fail:
        with pytest.raises(c.CutoffError) as ei:
            c.choose_cutoffs(scores, is_g, rule)
        assert ei.value.kind == "target not met"
        assert ei.value.cutoff == expected_fail
        assert ei.value.closest_fa == pytest.approx(top_fa)
        assert expected_fail in str(ei.value)
        return

    cuts = c.choose_cutoffs(scores, is_g, rule)
    for cut, target in ((cuts.medium, t_med), (cuts.high, t_high)):
        assert cut in genuine
    assert _fa(genuine, cuts.medium) <= _limit(genuine, t_med)
    assert cuts.achieved_fa_medium == pytest.approx(_fa(genuine, cuts.medium))
    assert cuts.achieved_fa_high == pytest.approx(_fa(genuine, cuts.high))
    # lowest observed genuine score that meets the effective limit
    for cut, target in ((cuts.medium, t_med),):
        for s in np.unique(genuine):
            if s < cut:
                assert _fa(genuine, s) > _limit(genuine, target)
    if cuts.high != cuts.medium:
        # high was selected on its own target
        assert _fa(genuine, cuts.high) <= _limit(genuine, t_high)
        for s in np.unique(genuine):
            if s < cuts.high:
                assert _fa(genuine, s) > _limit(genuine, t_high)
    assert cuts.high >= cuts.medium


@settings(max_examples=100, deadline=None)
@given(_labelled(), st.data())
def test_property_6_depends_on_validation_only(data, draw):
    """**Validates: Requirements 2.1, 2.8**"""
    scores, is_g = data
    rule = c.CutoffRule(target_fa_medium=100.0, target_fa_high=100.0)
    before = c.choose_cutoffs(scores, is_g, rule)
    # Other datasets (anything not passed in) cannot matter; also tampered
    # scores never influence the result.
    new_t = draw.draw(st.lists(_score, min_size=int((~is_g).sum()),
                               max_size=int((~is_g).sum())))
    changed = scores.copy()
    changed[~is_g] = new_t
    assert c.choose_cutoffs(changed, is_g, rule) == before
    assert c.choose_cutoffs(scores.copy(), is_g.copy(), rule) == before


@settings(max_examples=100, deadline=None)
@given(_labelled())
def test_property_6_missing_class_raises(data):
    """**Validates: Requirements 2.8**"""
    scores, is_g = data
    rule = c.CutoffRule()
    for flags in (np.zeros_like(is_g), np.ones_like(is_g)):
        with pytest.raises(c.CutoffError) as ei:
            c.choose_cutoffs(scores, flags, rule)
        assert ei.value.kind == "missing class"


@settings(max_examples=200, deadline=None)
@given(_labelled(), st.lists(_score, min_size=1, max_size=6))
def test_property_7_operating_curve_consistent(data, cut_values):
    """**Validates: Requirements 2.6**"""
    scores, is_g = data
    cuts = {f"c{i}": v for i, v in enumerate(cut_values)}
    rows = c.operating_curve(scores, is_g, cuts)
    assert [r.name for r in rows] == list(cuts)
    n_g, n_t = int(is_g.sum()), int((~is_g).sum())
    for r in rows:
        flagged = scores >= r.cutoff
        fg = int((flagged & is_g).sum())
        ft = int((flagged & ~is_g).sum())
        assert r.false_alarm_rate == pytest.approx(100.0 * fg / n_g)
        assert r.catch_rate == pytest.approx(100.0 * ft / n_t)
        assert r.alert_rate == pytest.approx(100.0 * (fg + ft) / len(scores))


@settings(max_examples=100, deadline=None)
@given(_labelled(), st.lists(_score, min_size=1, max_size=4))
def test_property_7_printed_percentages_have_one_decimal(data, cut_values):
    """**Validates: Requirements 2.6**"""
    scores, is_g = data
    cuts = {f"c{i}": v for i, v in enumerate(cut_values)}
    text = c.format_operating_curve(scores, is_g, cuts)
    lines = text.splitlines()
    assert len(lines) == len(cuts) + len(c.FIXED_GRID)
    for line in lines:
        pcts = re.findall(r"(\d+(?:\.\d+)?)%", line)
        assert len(pcts) == 3
        assert all(re.fullmatch(r"\d+\.\d", p) for p in pcts)
