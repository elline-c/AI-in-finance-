"""Task 11.1: failing tests for fraud_detect.ml.retraining.cutoffs (written before the module).

Pinned API (design "Cutoff selection"):
  CutoffRule(target_fa_medium, target_fa_high, level)   frozen, percent targets
  Cutoffs(medium, high, achieved_fa_medium, achieved_fa_high)
  CutoffError(kind, detail=None, closest_fa=None, cutoff=None)
  choose_cutoffs(scores, is_genuine, rule) -> Cutoffs        validation arrays only
  operating_curve(scores, is_genuine, cutoffs) -> list[CurveRow(name, cutoff, alert_rate, false_alarm_rate, catch_rate)]
  format_operating_curve(scores, is_genuine, cutoffs) -> str     medium/high rows, then FIXED_GRID rows
  statement_max_row(statement_ids, row_scores) -> {statement_id: max row score}
  FIXED_GRID == (0.5, 0.7, 0.8, 0.9, 0.95)
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from fraud_detect.ml.retraining import cutoffs as c


def _rule(medium=5.0, high=1.0, level="document"):
    return c.CutoffRule(target_fa_medium=medium, target_fa_high=high, level=level)


def _data(seed=0, n_g=200, n_t=100):
    rng = np.random.default_rng(seed)
    g = np.round(rng.random(n_g) * 0.6, 3)
    t = np.round(0.3 + rng.random(n_t) * 0.7, 3)
    scores = np.concatenate([g, t])
    is_genuine = np.concatenate([np.ones(n_g, bool), np.zeros(n_t, bool)])
    return scores, is_genuine


def _fa(scores, is_genuine, cut):
    g = scores[is_genuine]
    return 100.0 * np.sum(g >= cut) / len(g)


def test_rule_is_frozen():
    r = _rule()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.target_fa_medium = 1.0  # type: ignore[misc]


def test_choose_cutoffs_takes_validation_arrays_only():
    params = list(inspect.signature(c.choose_cutoffs).parameters)
    assert params == ["scores", "is_genuine", "rule"]


def test_cutoffs_meet_target_and_are_the_lowest_that_do():
    scores, is_genuine = _data()
    cuts = c.choose_cutoffs(scores, is_genuine, _rule())
    for cut, achieved, target in (
        (cuts.medium, cuts.achieved_fa_medium, 5.0),
        (cuts.high, cuts.achieved_fa_high, 1.0),
    ):
        assert _fa(scores, is_genuine, cut) <= target
        assert achieved == pytest.approx(_fa(scores, is_genuine, cut))
        lower = sorted(s for s in set(scores[is_genuine]) if s < cut)
        assert all(_fa(scores, is_genuine, s) > target for s in lower)
        assert cut in set(scores[is_genuine])  # an observed genuine score
    assert cuts.high >= cuts.medium


def test_equal_targets_give_equal_cutoffs():
    scores, is_genuine = _data(1)
    same = c.choose_cutoffs(scores, is_genuine, _rule(medium=2.0, high=2.0))
    assert same.high == same.medium


def test_plateau_at_top_means_no_cutoff_meets_target():
    # 20 genuine docs, 10 of them share the top score: lowest reachable false-alarm rate is 50%
    g = np.array([0.1] * 10 + [0.9] * 10)
    scores = np.concatenate([g, [0.95, 0.96]])
    is_genuine = np.array([True] * 20 + [False] * 2)
    with pytest.raises(c.CutoffError) as exc:
        c.choose_cutoffs(scores, is_genuine, _rule(medium=5.0, high=1.0))
    err = exc.value
    assert err.cutoff == "medium"
    assert err.closest_fa == pytest.approx(50.0)
    assert "medium" in str(err) and "50.0" in str(err)


def test_high_cutoff_failure_is_named_when_only_high_fails():
    g = np.array([0.1] * 90 + [0.9] * 10)  # top plateau = 10% of genuine
    scores = np.concatenate([g, [0.95]])
    is_genuine = np.array([True] * 100 + [False])
    with pytest.raises(c.CutoffError) as exc:
        c.choose_cutoffs(scores, is_genuine, _rule(medium=10.0, high=1.0))
    assert exc.value.cutoff == "high"
    assert exc.value.closest_fa == pytest.approx(10.0)


@pytest.mark.parametrize("missing", ["genuine", "tampered"])
def test_missing_class_raises_before_any_cutoff_is_chosen(missing):
    scores = np.array([0.1, 0.2, 0.3])
    is_genuine = np.ones(3, bool) if missing == "tampered" else np.zeros(3, bool)
    with pytest.raises(c.CutoffError) as exc:
        c.choose_cutoffs(scores, is_genuine, _rule())
    assert exc.value.kind == "missing class"
    assert missing in str(exc.value)


def test_results_do_not_depend_on_other_datasets():
    scores, is_genuine = _data(2)
    first = c.choose_cutoffs(scores, is_genuine, _rule())
    _other = _data(99)  # unrelated "test" data created in between changes nothing
    again = c.choose_cutoffs(scores.copy(), is_genuine.copy(), _rule())
    assert first == again


def test_input_arrays_are_not_modified():
    scores, is_genuine = _data(3)
    s0, g0 = scores.copy(), is_genuine.copy()
    c.choose_cutoffs(scores, is_genuine, _rule())
    assert np.array_equal(scores, s0) and np.array_equal(is_genuine, g0)


# ---------------------------------------------------------------- operating curve
def test_operating_curve_rows_match_direct_counts():
    scores, is_genuine = _data(4)
    cuts = c.choose_cutoffs(scores, is_genuine, _rule())
    rows = c.operating_curve(scores, is_genuine, {"medium": cuts.medium, "high": cuts.high})
    assert [r.name for r in rows] == ["medium", "high"]
    for r in rows:
        alerts = scores >= r.cutoff
        assert r.alert_rate == pytest.approx(100.0 * alerts.mean())
        assert r.false_alarm_rate == pytest.approx(100.0 * alerts[is_genuine].mean())
        assert r.catch_rate == pytest.approx(100.0 * alerts[~is_genuine].mean())


def test_formatted_curve_has_one_decimal_percentages_and_fixed_grid_below():
    scores, is_genuine = _data(5)
    cuts = c.choose_cutoffs(scores, is_genuine, _rule())
    text = c.format_operating_curve(scores, is_genuine, {"medium": cuts.medium, "high": cuts.high})
    assert c.FIXED_GRID == (0.5, 0.7, 0.8, 0.9, 0.95)
    lines = text.splitlines()
    medium_line = next(i for i, ln in enumerate(lines) if "medium" in ln)
    high_line = next(i for i, ln in enumerate(lines) if "high" in ln)
    grid_lines = [i for i, ln in enumerate(lines) for g in c.FIXED_GRID if f"{g}" in ln and i > high_line]
    assert medium_line < high_line < min(grid_lines)
    row = lines[medium_line]
    pcts = [tok for tok in row.replace(",", " ").split() if tok.endswith("%")]
    assert len(pcts) == 3  # alert rate, false-alarm rate, catch rate
    assert all(len(p[:-1].split(".")[1]) == 1 for p in pcts)


# ---------------------------------------------------------------- statement level
def test_statement_level_uses_the_maximum_row_score():
    ids = ["a", "a", "a", "b", "b", "c"]
    row_scores = np.array([0.1, 0.9, 0.3, 0.2, 0.25, 0.7])
    out = c.statement_max_row(ids, row_scores)
    assert out == {"a": 0.9, "b": 0.25, "c": 0.7}


def test_statement_level_cutoffs_use_statement_maxima_not_rows():
    # 10 genuine statements x 100 rows; one noisy row each. Row-level 5% target would be wildly different.
    rng = np.random.default_rng(7)
    ids, row_scores = [], []
    for k in range(40):
        rows = rng.random(100) * 0.2
        rows[0] = 0.3 + k * 0.01  # distinct statement maxima 0.30 .. 0.69
        ids += [f"g{k}"] * 100
        row_scores += list(rows)
    stmt = c.statement_max_row(ids, np.array(row_scores))
    scores = np.array(list(stmt.values()) + [0.99, 0.98])
    is_genuine = np.array([True] * 40 + [False] * 2)
    cuts = c.choose_cutoffs(scores, is_genuine, _rule(level="statement_max_row"))
    assert 100.0 * np.sum(np.array(list(stmt.values())) >= cuts.medium) / 40 <= 5.0
    assert cuts.medium in set(stmt.values())
