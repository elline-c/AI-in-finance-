"""Task 11.5: property tests for metrics and statistics (Properties 20, 21, 22).

All data are synthetic; nothing touches the disk.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detect.ml.retraining import metrics as m

from .strategies import score_label_arrays

SETTINGS = settings(max_examples=100, deadline=None)

TYPES = ("edited_amount", "font_mismatch", "structuring", "fabricated_receipts")
SPLITS = ("train", "validation", "test", "test_unseen_layout")


# ------------------------------------------------------------ Property 20
@SETTINGS
@given(st.integers(0, 500).flatmap(lambda d: st.tuples(st.integers(0, d), st.just(d))))
def test_rate_equals_count_ratio_and_formats_one_decimal(nd):
    """**Validates: Requirements 6.4, 6.7, 6.8, 9.8, 9.10, 9.11, 9.12**"""
    numer, denom = nd
    r = m.rate(numer, denom)
    assert (r.numer, r.denom) == (numer, denom)
    if denom == 0:
        assert r.value is None
        assert "not computable" in r.text
    else:
        assert r.value == numer / denom
        assert r.text.startswith(f"{r.value * 100:.1f}%")
        assert f"({numer}/{denom})" in r.text


@SETTINGS
@given(st.integers(0, 80), st.integers(0, 80), st.integers(1, 60))
def test_metric_flag_iff_rules(n_pos, n_neg, min_docs):
    """**Validates: Requirements 6.4, 6.7, 6.8, 9.8, 9.10, 9.11, 9.12**"""
    flag = m.metric_flag(n_pos, n_neg, min_docs)
    if n_pos == 0 or n_neg == 0:
        assert flag == "not computable"
    elif n_pos < min_docs or n_neg < min_docs:
        assert flag == "indicative only"
    else:
        assert flag == "ok"


@SETTINGS
@given(
    st.lists(
        st.tuples(st.booleans(), st.integers(0, 10), st.lists(st.sampled_from(TYPES), max_size=3, unique=True),
                  st.booleans()),
        min_size=0, max_size=60,
    ),
    st.integers(0, 10),
)
def test_catch_rate_by_matches_direct_counts(docs, cutoff):
    """**Validates: Requirements 6.4, 6.7, 6.8, 9.8, 9.10, 9.11, 9.12**"""
    scores = [d[1] / 10 for d in docs]
    tampered = [d[0] for d in docs]
    keys = [tuple(d[2]) for d in docs]
    out = m.catch_rate_by(scores, tampered, keys, cutoff / 10)
    for t in TYPES:
        idx = [i for i, d in enumerate(docs) if d[0] and t in d[2]]
        if not idx:
            assert t not in out
            continue
        hit = sum(1 for i in idx if scores[i] >= cutoff / 10)
        assert (out[t].numer, out[t].denom) == (hit, len(idx))
        assert out[t].value == hit / len(idx)


@SETTINGS
@given(
    st.lists(
        st.tuples(st.booleans(), st.lists(st.sampled_from(TYPES), max_size=2, unique=True), st.sampled_from(SPLITS)),
        max_size=40,
    )
)
def test_empty_pairs_lists_exactly_pairs_without_documents(docs):
    """**Validates: Requirements 6.4, 6.7, 6.8, 9.8, 9.10, 9.11, 9.12**"""
    rows = [{"is_tampered": t, "tamper_types": ty, "split": sp} for t, ty, sp in docs]
    got = m.empty_pairs(rows, TYPES, SPLITS)
    present = {(ty_, sp) for t, ty, sp in docs if t for ty_ in ty}
    assert set(got) == {(t, s) for t in TYPES for s in SPLITS} - present
    assert len(got) == len(set(got))


@SETTINGS
@given(st.lists(st.tuples(st.booleans(), st.booleans()), max_size=80))
def test_hard_negative_and_ordinary_are_disjoint_and_cover_genuine(pairs):
    """**Validates: Requirements 6.4, 6.7, 6.8, 9.8, 9.10, 9.11, 9.12**"""
    genuine = np.array([p[0] for p in pairs], dtype=bool)
    resaved = np.array([p[1] for p in pairs], dtype=bool)
    hard, ordinary = m.split_negatives(genuine, resaved)
    assert not np.any(hard & ordinary)
    assert np.array_equal(hard | ordinary, genuine)
    assert np.array_equal(hard, genuine & resaved)


# ------------------------------------------------------------ Property 21
@SETTINGS
@given(score_label_arrays(min_size=2, max_size=80))
def test_auc_equals_sklearn(arrays):
    """**Validates: Requirements 9.8, 9.9**"""
    s, y = arrays
    assert m.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert m.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))


@SETTINGS
@given(st.integers(1, 100000))
def test_top_k_count_is_ceil_one_percent_at_least_one(n):
    """**Validates: Requirements 9.8, 9.9**"""
    k = m.top_k_count(n)
    assert k >= 1
    # exact integer arithmetic: smallest k with 100*k >= n
    assert k == max(1, -(-n // 100))


@SETTINGS
@given(score_label_arrays(min_size=2, max_size=80), st.data())
def test_precision_recall_at_top_k_equal_direct_counts(arrays, data):
    """**Validates: Requirements 9.8, 9.9**"""
    s, y = arrays
    k = data.draw(st.integers(1, len(s)))
    p, r = m.precision_recall_at_top_k(y, s, k)
    # Ties: the implementation takes a stable descending order. Recompute that
    # selection independently and count true positives directly.
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    tp = sum(y[i] for i in order[:k])
    assert p == pytest.approx(tp / k)
    assert r == pytest.approx(tp / sum(y))
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0


# ------------------------------------------------------------ Property 22
@SETTINGS
@given(st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=20))
def test_summarise_seeds_equals_numpy(values):
    """**Validates: Requirements 9.3, 9.4, 9.5, 15.5**"""
    st_ = m.summarise_seeds(values)
    arr = np.array(values)
    assert st_.n == len(values)
    assert st_.mean == pytest.approx(float(arr.mean()))
    assert st_.min == float(arr.min())
    assert st_.max == float(arr.max())
    if len(values) == 1:
        assert st_.std is None
    else:
        assert st_.std == pytest.approx(float(np.std(arr, ddof=1)), abs=1e-12)


@SETTINGS
@given(
    st.lists(st.integers(20260800, 20260820), min_size=0, max_size=10, unique=True),
    st.data(),
    st.integers(1, 6),
)
def test_seed_completeness_incomplete_iff_fewer_than_min_and_lists_failed(attempted, data, min_seeds):
    """**Validates: Requirements 9.3, 9.4, 9.5, 15.5**"""
    completed = data.draw(st.lists(st.sampled_from(attempted), unique=True)) if attempted else []
    res = m.seed_completeness(attempted, completed, min_seeds)
    assert (res.status == "incomplete") == (len(set(completed)) < min_seeds)
    assert set(res.failed) == set(attempted) - set(completed)
    # every attempted seed is accounted for: completed or failed, none lost
    assert set(completed) | set(res.failed) == set(attempted)


@SETTINGS
@given(st.data())
def test_seed_difference_is_candidate_minus_current_with_min_max(data):
    """**Validates: Requirements 9.3, 9.4, 9.5, 15.5**"""
    n = data.draw(st.integers(1, 10))
    f = st.floats(0, 1, allow_nan=False)
    cand = data.draw(st.lists(f, min_size=n, max_size=n))
    cur = data.draw(st.lists(f, min_size=n, max_size=n))
    d = m.seed_difference(cand, cur)
    expected = [a - b for a, b in zip(cand, cur)]
    assert d.per_seed == expected
    assert d.min == min(expected) and d.max == max(expected)
    assert not any(math.isnan(x) for x in d.per_seed)
