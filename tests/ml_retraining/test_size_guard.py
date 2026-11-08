"""Task 19.1: failing tests for fraud_detect.ml.retraining.size_guard (written before the module).

Pinned API (design "Size_Guard recommendation"):
  SIZE_FEATURES                                   == statement_features.SIZE_FEATURES
  candidate_ranges(train_sizes, quantiles=GUARD_QUANTILES) -> {feature: [lo, hi]}
        train_sizes: pandas DataFrame with the SIZE_FEATURES columns (training documents only);
        lo/hi are the 0.01 / 0.99 quantiles (pandas default interpolation, as train_statement.py)
  outside_mask(sizes, ranges) -> bool ndarray      True when ANY size feature is < lo or > hi
  outside_summary(sizes_by_split, ranges) -> {split: OutsideCount(count, total, percent)}
        percent = 100 * count / total, None when total == 0
  apply_guard(flagged, outside) -> bool ndarray    guard on: flagged & ~outside
  abstained_share(outside) -> float percent        100 * outside.mean()
  GuardOption (Enum): KEEP_CURRENT_GUARD="current", CANDIDATE_GUARD="candidate", REMOVE_GUARD="off"
  GuardEvidence(cand_fa_12_month, cand_fa_6_month, fa_on, fa_off, sources_with_30_genuine,
                cand_outside_share, cur_outside_share, outside_by_split=None)   (percent values)
        fa_on / fa_off: {source: percent}
  GuardRecommendation(option, figures, condition_failure)
        figures: dict that contains every evidence field used; condition_failure: None or str that
        contains both the 12-month and 6-month rates formatted with one decimal
  recommend_guard(ev) -> GuardRecommendation
  require_size_guard_choice(choice) -> "candidate" | "off" | "current"   ValueError otherwise (used by switch)
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml import statement_features as sf
from fraud_detect.ml import train_statement as ts
from fraud_detect.ml.retraining import size_guard as sg


def _sizes(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "n_pages": rng.integers(1, 6, n).astype(float),
        "n_text_objects": rng.normal(2000, 400, n),
        "n_money_tokens": rng.normal(150, 30, n),
    })


def _ev(**kw):
    base = dict(cand_fa_12_month=4.0, cand_fa_6_month=3.0,
                fa_on={"corpus": 3.0}, fa_off={"corpus": 4.0},
                sources_with_30_genuine=["corpus"],
                cand_outside_share=2.0, cur_outside_share=10.0)
    base.update(kw)
    return sg.GuardEvidence(**base)


def test_size_features_come_from_statement_features():
    assert list(sg.SIZE_FEATURES) == list(sf.SIZE_FEATURES)


def test_candidate_ranges_are_training_quantiles():
    tr = _sizes()
    lo, hi = ts.GUARD_QUANTILES
    assert (lo, hi) == (0.01, 0.99)
    r = sg.candidate_ranges(tr)
    assert set(r) == set(sf.SIZE_FEATURES)
    for f in sf.SIZE_FEATURES:
        assert r[f] == [float(tr[f].quantile(lo)), float(tr[f].quantile(hi))]


def test_outside_count_and_percent_equal_direct_counts():
    tr, va = _sizes(0), _sizes(1, 120)
    ranges = sg.candidate_ranges(tr)
    cur = {"n_pages": [2.0, 3.0], "n_text_objects": [1500.0, 2500.0], "n_money_tokens": [100.0, 200.0]}
    for rg in (ranges, cur):
        out = sg.outside_summary({"train": tr, "validation": va}, rg)
        for name, df in (("train", tr), ("validation", va)):
            direct = sum(
                any(row[f] < rg[f][0] or row[f] > rg[f][1] for f in sf.SIZE_FEATURES)
                for _, row in df.iterrows())
            assert out[name].count == direct
            assert out[name].total == len(df)
            assert out[name].percent == pytest.approx(100.0 * direct / len(df))


def test_outside_summary_empty_split_has_no_percent():
    out = sg.outside_summary({"test": _sizes(0, 0)}, sg.candidate_ranges(_sizes()))
    assert out["test"].count == 0 and out["test"].total == 0 and out["test"].percent is None


def test_bounds_are_inclusive():
    rg = {f: [1.0, 2.0] for f in sf.SIZE_FEATURES}
    df = pd.DataFrame({f: [1.0, 2.0, 2.5] for f in sf.SIZE_FEATURES})
    assert sg.outside_mask(df, rg).tolist() == [False, False, True]


@settings(max_examples=50, deadline=None)
@given(st.lists(st.booleans(), min_size=1, max_size=60), st.data())
def test_guard_on_flags_are_subset_of_guard_off(flagged, data):
    flagged = np.array(flagged)
    outside = np.array(data.draw(st.lists(st.booleans(), min_size=len(flagged), max_size=len(flagged))))
    on = sg.apply_guard(flagged, outside)
    assert not (on & ~flagged).any()
    assert not (on & outside).any()
    assert sg.abstained_share(outside) == pytest.approx(100.0 * outside.mean())


def test_remove_when_12_month_within_5_points_and_guard_off_not_worse():
    rec = sg.recommend_guard(_ev(cand_fa_12_month=8.0, cand_fa_6_month=3.0))  # gap exactly 5.0
    assert rec.option is sg.GuardOption.REMOVE_GUARD
    assert rec.condition_failure is None
    for k in ("cand_fa_12_month", "cand_fa_6_month", "fa_on", "fa_off",
              "cand_outside_share", "cur_outside_share"):
        assert k in rec.figures


def test_no_remove_when_12_month_gap_over_5_points_states_both_rates():
    rec = sg.recommend_guard(_ev(cand_fa_12_month=8.1, cand_fa_6_month=3.0))
    assert rec.option is not sg.GuardOption.REMOVE_GUARD
    assert "8.1" in rec.condition_failure and "3.0" in rec.condition_failure


def test_no_remove_when_guard_off_more_than_5_points_worse_on_a_source():
    rec = sg.recommend_guard(_ev(fa_on={"corpus": 3.0}, fa_off={"corpus": 8.5}))
    assert rec.option is not sg.GuardOption.REMOVE_GUARD


def test_guard_off_gap_ignored_for_sources_under_30_genuine():
    rec = sg.recommend_guard(_ev(fa_on={"corpus": 3.0, "real_style": 1.0},
                                 fa_off={"corpus": 4.0, "real_style": 50.0},
                                 sources_with_30_genuine=["corpus"]))
    assert rec.option is sg.GuardOption.REMOVE_GUARD


def test_candidate_guard_when_remove_fails_and_candidate_abstains_less():
    rec = sg.recommend_guard(_ev(cand_fa_12_month=20.0, cand_outside_share=2.0, cur_outside_share=10.0))
    assert rec.option is sg.GuardOption.CANDIDATE_GUARD


def test_keep_current_when_remove_fails_and_candidate_abstains_no_less():
    rec = sg.recommend_guard(_ev(cand_fa_12_month=20.0, cand_outside_share=10.0, cur_outside_share=10.0))
    assert rec.option is sg.GuardOption.KEEP_CURRENT_GUARD


@settings(max_examples=100, deadline=None)
@given(*[st.floats(0, 100, allow_nan=False) for _ in range(6)])
def test_recommendation_is_exactly_one_option_and_remove_needs_condition(a, b, c, d, e, f):
    ev = _ev(cand_fa_12_month=a, cand_fa_6_month=b, fa_on={"s": c}, fa_off={"s": d},
             sources_with_30_genuine=["s"], cand_outside_share=e, cur_outside_share=f)
    rec = sg.recommend_guard(ev)
    assert rec.option in set(sg.GuardOption)
    if rec.option is sg.GuardOption.REMOVE_GUARD:
        assert a - b <= 5.0
    elif a - b > 5.0:
        assert rec.condition_failure is not None


def test_option_values_match_switch_flag_choices():
    assert {o.value for o in sg.GuardOption} == {"current", "candidate", "off"}


@pytest.mark.parametrize("choice", ["candidate", "off", "current"])
def test_switch_accepts_explicit_choice(choice):
    assert sg.require_size_guard_choice(choice) == choice


@pytest.mark.parametrize("choice", [None, "", "auto", "CANDIDATE", "remove"])
def test_switch_refuses_missing_or_unknown_choice(choice):
    with pytest.raises(ValueError):
        sg.require_size_guard_choice(choice)


def test_module_never_writes_model_files():
    src = inspect.getsource(sg)
    for banned in ("joblib.dump", "pickle.dump", "open(", "atomic_write", "shutil", ".write("):
        assert banned not in src
