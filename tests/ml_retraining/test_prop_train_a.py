"""Task 16.3: property tests for Node 6a statement-level flags and GNN lookup accounting.

Property 29: Statement-level flags and model combination.
Property 30: GNN lookup accounting.
All data here is synthetic and generated in memory; no model file, no torch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import gnn_eval
from fraud_detect.ml.retraining.cutoffs import operating_curve
from fraud_detect.ml.retraining.train_a import statement_scores

SETTINGS = settings(max_examples=100, deadline=None)

score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


class _FixedPipeline:
    """Returns the ``p`` column as the fraud probability (real predict_proba shape)."""

    def predict_proba(self, X):
        p = X["p"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------ Property 29
@st.composite
def statements_with_rows(draw):
    n = draw(st.integers(1, 12))
    rows = []
    for i in range(n):
        rows += [(f"c{i % 3}", f"d{i}", draw(score_st)) for _ in range(draw(st.integers(1, 5)))]
    return rows


@SETTINGS
@given(rows=statements_with_rows(), cutoff=score_st)
def test_statement_flag_iff_max_row_score_at_least_cutoff(rows, cutoff):
    """**Validates: Requirements 11.9**"""
    df = pd.DataFrame(rows, columns=["corpus_key", "doc_id", "p"])
    got = statement_scores(_FixedPipeline(), df, ["p"])
    expected = df.groupby(["corpus_key", "doc_id"])["p"].max().to_dict()
    assert got == expected
    for key, mx in expected.items():
        assert (got[key] >= cutoff) == (mx >= cutoff)
        assert (got[key] >= cutoff) == bool((df[(df.corpus_key == key[0]) & (df.doc_id == key[1])]["p"] >= cutoff).any())


@SETTINGS
@given(data=st.lists(st.tuples(score_st, st.booleans()), min_size=1, max_size=30), cutoff=score_st)
def test_false_alarm_rate_is_flagged_genuine_over_genuine(data, cutoff):
    """**Validates: Requirements 11.9**"""
    scores = [d[0] for d in data]
    genuine = [d[1] for d in data]
    n_gen = sum(genuine)
    flags = [s >= cutoff for s in scores]
    res = gnn_eval.compare_models(flags, [None] * len(data), genuine, [not g for g in genuine])
    fa = res["gbm"].false_alarm
    flagged_gen = sum(1 for f, g in zip(flags, genuine) if f and g)
    assert (fa.numer, fa.denom) == (flagged_gen, n_gen)
    if n_gen:
        assert fa.value == flagged_gen / n_gen
        if any(genuine):
            curve = operating_curve(scores, genuine, {"c": cutoff})[0]
            assert abs(curve.false_alarm_rate - 100.0 * flagged_gen / n_gen) < 1e-9
    else:
        assert fa.value is None


@SETTINGS
@given(pairs=st.lists(st.tuples(st.booleans(), st.one_of(st.none(), st.booleans())), max_size=30))
def test_combined_flag_is_gbm_or_gnn_and_gbm_when_gnn_missing(pairs):
    """**Validates: Requirements 11.10**"""
    gbm = [p[0] for p in pairs]
    gnn = [p[1] for p in pairs]
    combined = gnn_eval.combine_flags(gbm, gnn)
    for g, n, c in zip(gbm, gnn, combined):
        if n is None:
            assert c == g
        else:
            assert c == (g or n)


@SETTINGS
@given(scores=st.lists(st.one_of(st.none(), score_st), max_size=30), cutoff=score_st)
def test_missing_gnn_score_never_replaced_by_number(scores, cutoff):
    """**Validates: Requirements 11.10**"""
    flags = gnn_eval.gnn_flags(scores, cutoff)
    assert len(flags) == len(scores)
    for s, f in zip(scores, flags):
        if s is None:
            assert f is None
        else:
            assert f == (s >= cutoff)


@SETTINGS
@given(rows=st.lists(st.tuples(st.booleans(), st.one_of(st.none(), st.booleans()),
                               st.booleans(), st.booleans()), min_size=1, max_size=30))
def test_compare_models_combined_uses_or_and_gnn_uses_scored_only(rows):
    """**Validates: Requirements 11.9, 11.10**"""
    gbm = [r[0] for r in rows]
    gnn = [r[1] for r in rows]
    # a statement is genuine or positive, never both
    genuine = [r[2] for r in rows]
    positive = [(not r[2]) and r[3] for r in rows]
    res = gnn_eval.compare_models(gbm, gnn, genuine, positive)
    comb = gnn_eval.combine_flags(gbm, gnn)

    def counts(flags, mask):
        fa = [(f, g) for f, g, m in zip(flags, genuine, mask) if m and g]
        de = [(f, p) for f, p, m in zip(flags, positive, mask) if m and p]
        return (sum(bool(f) for f, _ in fa), len(fa)), (sum(bool(f) for f, _ in de), len(de))

    all_mask = [True] * len(rows)
    scored = [f is not None for f in gnn]
    for name, flags, mask in (("gbm", gbm, all_mask), ("combined", comb, all_mask), ("gnn", gnn, scored)):
        (fn, fd), (dn, dd) = counts(flags, mask)
        r = res[name]
        assert (r.false_alarm.numer, r.false_alarm.denom) == (fn, fd)
        assert (r.detection.numer, r.detection.denom) == (dn, dd)
        ok = fd > 0 and dd > 0
        assert r.status == ("ok" if ok else "not computable")
    if not any(scored):
        assert res["gnn"].status == "not computable"
        assert res["gnn"].false_alarm.value is None and res["gnn"].detection.value is None


# ------------------------------------------------------------------ Property 30
uen_st = st.text(alphabet="0123456789ABCDEFGHIJ", min_size=3, max_size=9)


@SETTINGS
@given(heldout=st.lists(uen_st, max_size=25), stored=st.lists(uen_st, max_size=25),
       none_store=st.booleans(), data=st.data())
def test_gnn_lookup_accounting(heldout, stored, none_store, data):
    """**Validates: Requirements 11.7, 11.8**"""
    table = {u: data.draw(score_st) for u in stored}
    store = None if none_store else {"account_scores": table, "edge_scores": {}, "meta": {}}
    cov = gnn_eval.lookup_coverage(heldout, store)

    held = set(heldout)
    found = held & set(table) if store is not None else set()
    absent = held - found
    assert cov.total == len(held)
    assert cov.found == len(found)
    assert cov.no_score == len(absent) == cov.total - cov.found
    assert cov.no_score_uens == tuple(sorted(absent))
    for u in absent:
        assert cov.scores[u] is None  # never a substitute number
    for u in found:
        assert cov.scores[u] == table[u]
    if held:
        expected = 100.0 * len(found) / len(held)
        assert cov.share == expected
        assert cov.text.startswith(f"{expected:.1f}%")
        assert cov.text == f"{expected:.1f}% ({len(found)}/{len(held)})"
    else:
        assert cov.share is None
        assert cov.text == "not computable (0/0)"
