"""Task 11.1: failing tests for fraud_detect.ml.retraining.metrics (written before the module).

Pinned API (design "metrics.py"):
  top_k_count(n, fraction=0.01) -> int
  roc_auc(y, s), pr_auc(y, s) -> float
  precision_recall_at_top_k(y, s, k) -> (precision, recall)
  rate(numer, denom) -> Rate(value, numer, denom, text)   value is a fraction or None
  metric_flag(n_pos, n_neg, min_docs=30) -> "ok" | "indicative only" | "not computable"
  catch_rate_by(scores, is_tampered, keys, cutoff) -> {key: Rate}   keys: str or tuple of str per doc
  empty_pairs(rows, types, splits) -> [(type, split), ...] with zero tampered documents
  split_negatives(is_genuine, genuine_resaved) -> (hard_negative_mask, ordinary_genuine_mask)
  summarise_seeds(values) -> SeedStats(n, mean, min, max, std)
  seed_completeness(attempted, completed, min_seeds=5) -> RunCompleteness(status, failed)
  seed_difference(candidate, current) -> SeedDifference(per_seed, min, max)
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detect.ml.report import topk_precision_recall
from fraud_detect.ml.retraining import metrics as m


# ---------------------------------------------------------------- ranking metrics
def test_roc_and_pr_auc_equal_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 2)  # ties included
    assert m.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert m.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))


@pytest.mark.parametrize(
    "n,expected", [(1, 1), (50, 1), (100, 1), (101, 2), (250, 3), (1000, 10), (1001, 11)]
)
def test_top_k_count_rounds_up_and_is_at_least_one(n, expected):
    assert m.top_k_count(n) == expected


def test_top_k_count_differs_from_old_round_helper():
    # old helper: max(1, round(n * 0.01)); n=250 -> round(2.5)=2 (banker's), ceil gives 3
    assert round(250 * 0.01) == 2
    assert m.top_k_count(250) == 3
    # the old helper in report.py stays as it is (untouched), only its behaviour is pinned here
    # positives only at ranks 1 and 2: old slice (k=2) precision 1.0, new slice (k=3) 2/3
    y = np.array([1] * 2 + [0] * 248)
    s = np.linspace(1, 0, 250)
    old_p, _, old_k = topk_precision_recall(y, s)
    new_p, _ = m.precision_recall_at_top_k(y, s, m.top_k_count(250))
    assert old_k == 2
    assert old_p == pytest.approx(1.0)
    assert new_p == pytest.approx(2 / 3)
    assert old_p != new_p  # 2 documents vs 3 documents in the top slice


def test_precision_recall_at_top_k_equal_direct_counts():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    p, r = m.precision_recall_at_top_k(y, s, 3)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)


# ---------------------------------------------------------------- rates, flags
def test_rate_value_and_counts():
    r = m.rate(1, 4)
    assert (r.value, r.numer, r.denom) == (0.25, 1, 4)
    assert r.text.startswith("25.0%")


def test_rate_with_zero_denominator_has_no_value_but_keeps_counts():
    r = m.rate(0, 0)
    assert r.value is None and (r.numer, r.denom) == (0, 0)


@pytest.mark.parametrize(
    "numer,denom,text",
    [(1, 3, "33.3%"), (2, 3, "66.7%"), (1, 8, "12.5%"), (0, 5, "0.0%"), (5, 5, "100.0%")],
)
def test_percentages_print_with_one_decimal(numer, denom, text):
    assert m.rate(numer, denom).text.startswith(text)


@pytest.mark.parametrize(
    "n_pos,n_neg,flag",
    [
        (30, 30, "ok"),
        (29, 30, "indicative only"),
        (30, 29, "indicative only"),
        (1, 500, "indicative only"),
        (0, 500, "not computable"),
        (500, 0, "not computable"),
        (0, 0, "not computable"),
    ],
)
def test_metric_flag(n_pos, n_neg, flag):
    assert m.metric_flag(n_pos, n_neg) == flag


def test_metric_flag_threshold_is_configurable():
    assert m.metric_flag(10, 10, min_docs=10) == "ok"


def test_not_computable_metric_has_no_numeric_value():
    r = m.rate(0, 0)
    assert "not computable" in r.text
    assert "%" not in r.text


# ---------------------------------------------------------------- per type / method
def test_catch_rate_per_type_and_method_with_counts():
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.95, 0.3])
    tampered = np.array([1, 1, 1, 1, 0, 0])
    types = [("edited_amount",), ("edited_amount",), ("font_mismatch", "edited_amount"), ("structuring",), (), ()]
    out = m.catch_rate_by(scores, tampered, types, cutoff=0.5)
    assert set(out) == {"edited_amount", "font_mismatch", "structuring"}
    assert (out["edited_amount"].numer, out["edited_amount"].denom) == (2, 3)
    assert (out["font_mismatch"].numer, out["font_mismatch"].denom) == (1, 1)
    assert (out["structuring"].numer, out["structuring"].denom) == (0, 1)
    methods = ["pdf_edit", "pdf_edit", "regen", "regen", "", ""]
    by_method = m.catch_rate_by(scores, tampered, methods, cutoff=0.5)
    assert (by_method["pdf_edit"].numer, by_method["pdf_edit"].denom) == (1, 2)
    assert (by_method["regen"].numer, by_method["regen"].denom) == (1, 2)


def test_cutoff_comparison_is_greater_or_equal():
    out = m.catch_rate_by(np.array([0.5]), np.array([1]), ["x"], cutoff=0.5)
    assert out["x"].numer == 1


def test_empty_type_split_pair_is_listed_and_has_no_rate():
    rows = [
        {"split": "test", "tamper_types": ("edited_amount",), "is_tampered": 1},
        {"split": "validation", "tamper_types": ("edited_amount",), "is_tampered": 1},
        {"split": "test", "tamper_types": ("structuring",), "is_tampered": 1},
        {"split": "test", "tamper_types": (), "is_tampered": 0},
    ]
    missing = m.empty_pairs(rows, types=["edited_amount", "structuring"], splits=["validation", "test"])
    assert missing == [("structuring", "validation")]
    assert m.rate(0, 0).value is None


# ---------------------------------------------------------------- hard negatives
def test_hard_negative_and_ordinary_genuine_are_disjoint_and_cover_genuine():
    is_genuine = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    resaved = np.array([1, 0, 0, 0, 1, 1], dtype=bool)  # resaved flag on a tampered doc is ignored
    hard, ordinary = m.split_negatives(is_genuine, resaved)
    assert not np.any(hard & ordinary)
    assert np.array_equal(hard | ordinary, is_genuine)
    assert np.array_equal(hard, np.array([1, 0, 0, 0, 0, 1], dtype=bool))


# ---------------------------------------------------------------- seeds
def test_summarise_seeds_matches_numpy_with_sample_std():
    vals = [0.91, 0.88, 0.95, 0.90, 0.93]
    s = m.summarise_seeds(vals)
    assert s.mean == pytest.approx(np.mean(vals))
    assert s.min == min(vals) and s.max == max(vals)
    assert s.std == pytest.approx(np.std(vals, ddof=1))


def test_summarise_seeds_single_value_has_no_std():
    s = m.summarise_seeds([0.9])
    assert s.mean == 0.9 and s.min == s.max == 0.9
    assert s.std is None


def test_run_with_fewer_than_five_seeds_is_incomplete_and_lists_failed():
    attempted = [1, 2, 3, 4, 5]
    r = m.seed_completeness(attempted, completed=[1, 2, 4, 5])
    assert r.status == "incomplete" and tuple(r.failed) == (3,)
    ok = m.seed_completeness(attempted, completed=attempted)
    assert ok.status == "complete" and tuple(ok.failed) == ()


def test_duplicate_seeds_count_once():
    r = m.seed_completeness([1, 2, 3, 4, 5], completed=[1, 1, 2, 2, 3])
    assert r.status == "incomplete"
    assert tuple(sorted(r.failed)) == (4, 5)


def test_seed_difference_is_candidate_minus_current_with_spread():
    d = m.seed_difference([0.9, 0.8, 0.7], [0.8, 0.8, 0.9])
    assert d.per_seed == pytest.approx([0.1, 0.0, -0.2])
    assert d.min == pytest.approx(-0.2) and d.max == pytest.approx(0.1)
    assert not math.isnan(d.min)
