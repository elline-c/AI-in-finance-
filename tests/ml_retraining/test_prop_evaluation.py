"""Task 18.4: property tests for source separation, comparison arithmetic and baseline states.

Property 23: Sources and datasets never mix.
Property 33: Comparison arithmetic and baseline states.

Only in-memory datasets and tiny fake models are used; nothing under ``models/`` is touched.

**Validates: Requirements 9.1, 9.2, 9.6, 13.1, 13.3, 13.5, 13.6**
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from sklearn.linear_model import LogisticRegression

from fraud_detect.ml import statement_features as sf
from fraud_detect.ml.retraining import baseline, sources
from fraud_detect.ml.retraining import evaluation as ev
from fraud_detect.ml.retraining.vault import FrozenCutoffs

PBT = settings(max_examples=12, deadline=None)
CUT = FrozenCutoffs(medium=0.5, high=0.8)
DATASET_IDS = list(sources.DATASETS)


# ----------------------------------------------------------------------------- fakes
def _feat(breaks, pages=2):
    f = {name: 0.0 for name in sf.FEATURE_NAMES}
    f.update(n_pages=pages, n_text_objects=60, n_money_tokens=70, balance_reconcile_breaks=breaks)
    return f


def _bundle(seed, coef):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(60, len(sf.FEATURE_NAMES))), columns=sf.FEATURE_NAMES)
    y = (X["balance_reconcile_breaks"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    model.coef_ = np.zeros_like(model.coef_)
    model.coef_[0, list(sf.FEATURE_NAMES).index("balance_reconcile_breaks")] = coef
    model.intercept_ = np.zeros_like(model.intercept_)
    return {"model": model, "feature_names": list(sf.FEATURE_NAMES),
            "thresholds": {"medium": 0.5, "high": 0.8},
            "size_ranges": {"n_pages": [1, 3], "n_text_objects": [10, 100],
                            "n_money_tokens": [10, 120]}}


CANDIDATE = _bundle(0, 0.8)
CURRENT = baseline.wrap("6b", _bundle(1, 1.0))


@st.composite
def _docs(draw):
    """Documents with both classes present (so AUC is defined)."""
    n_gen = draw(st.integers(2, 5))
    n_tam = draw(st.integers(2, 5))
    breaks = st.floats(-3, 3, allow_nan=False, allow_infinity=False)
    docs = []
    for i in range(n_gen + n_tam):
        tam = i >= n_gen
        docs.append(ev.EvalDoc(
            doc_id=f"d{i}", is_tampered=tam, features=_feat(draw(breaks), draw(st.sampled_from([2, 2, 20]))),
            genuine_resaved=(not tam) and draw(st.booleans()),
            n_months=draw(st.sampled_from([6, 12])),
            tamper_types=("edited_amount",) if tam else (), tamper_method="overlay" if tam else None))
    return tuple(docs)


def _dataset(ds_id, docs):
    return ev.EvalDataset(dataset_id=ds_id, source=sources.DATASETS[ds_id], docs=docs)


def _eval(dataset, current=CURRENT, seed=1):
    return ev.evaluate_dataset("6b", dataset, candidate=CANDIDATE, current=current,
                               cutoffs=CUT, matched=CUT, seed=seed)


# ----------------------------------------------------------------------------- Property 23
@PBT
@given(ds_id=st.sampled_from(DATASET_IDS), docs=_docs())
def test_p23_each_row_has_exactly_one_allowed_source_and_dataset(ds_id, docs):
    r = _eval(_dataset(ds_id, docs))
    assert r["source"] in ev.SOURCES and r["dataset"] in ev.DATASET_IDS
    assert (r["dataset"], r["source"]) == (ds_id, sources.DATASETS[ds_id])


@PBT
@given(a=st.sampled_from(DATASET_IDS), b=st.sampled_from(DATASET_IDS), docs=_docs())
def test_p23_aggregation_only_within_one_source_and_dataset(a, b, docs):
    da, db = _dataset(a, docs), _dataset(b, docs)
    if a == b:
        agg = ev.aggregate([da, db])
        assert agg.dataset_id == a and len(agg.docs) == 2 * len(docs)
    else:
        with pytest.raises(ev.MixedSourcesError):
            ev.aggregate([da, db])


@PBT
@given(ds_id=st.sampled_from(DATASET_IDS), wrong=st.sampled_from(ev.SOURCES))
def test_p23_dataset_with_a_foreign_source_is_rejected(ds_id, wrong):
    if wrong == sources.DATASETS[ds_id]:
        assert ev.EvalDataset(ds_id, wrong, ()).source == wrong
    else:
        with pytest.raises(ev.MixedSourcesError):
            ev.EvalDataset(ds_id, wrong, ())


@PBT
@given(docs=_docs(), guard=st.booleans())
def test_p23_candidate_and_current_are_scored_on_identical_document_lists(docs, guard):
    seen = []
    real = baseline.score_b

    def spy(handle, feats, **kw):
        seen.append((handle.bundle is CANDIDATE, [id(f) for f in feats]))
        return real(handle, feats, **kw)

    with mock.patch.object(baseline, "score_b", spy):
        r = ev.evaluate_dataset("6b", _dataset("test", docs), candidate=CANDIDATE, current=CURRENT,
                                cutoffs=CUT, matched=CUT, seed=3, size_guard=guard)
    expected = [id(d.features) for d in docs]
    assert len(seen) >= 2 and any(c for c, _ in seen) and any(not c for c, _ in seen)
    assert all(ids == expected for _, ids in seen)
    assert r["doc_ids"] == [d.doc_id for d in docs]


# ----------------------------------------------------------------------------- Property 33
def _approx_diff(cand, cur):
    return None if cand is None or cur is None else cand - cur


@PBT
@given(docs=_docs(), seeds=st.lists(st.integers(1, 10**6), min_size=1, max_size=3, unique=True))
def test_p33_difference_is_candidate_minus_current_and_spread_is_min_max(docs, seeds):
    diffs = []
    for seed in seeds:
        r = _eval(_dataset("test", docs), seed=seed)
        assert r["current_state"] == baseline.STATUS_OK
        c, k, d = r["candidate"]["matched"], r["current"]["matched"], r["difference"]["matched"]
        for key, cv, kv in (("roc_auc", c["roc_auc"], k["roc_auc"]),
                            ("catch_rate", c["catch_rate"]["value"], k["catch_rate"]["value"]),
                            ("false_alarm_rate", c["false_alarm_rate"]["value"],
                             k["false_alarm_rate"]["value"])):
            want = _approx_diff(cv, kv)
            if want is None:
                assert d[key] is None
            else:
                assert d[key] == pytest.approx(want, abs=1e-12)
        diffs.append(d["catch_rate"])
    spread = (min(diffs), max(diffs))
    assert spread[0] <= spread[1] and all(spread[0] <= x <= spread[1] for x in diffs)


def _failing_handle():
    # A loaded-looking model that cannot score: it lacks the feature names the live node needs.
    return baseline.wrap("6b", {"model": object(), "feature_names": ["nope"]})


@PBT
@given(docs=_docs(), ds_id=st.sampled_from(["test", "heldout_layout", "heldout_seed", "forgery_test",
                                            "forgery_ood", "real_style", "awkward"]),
       kind=st.sampled_from(["missing", "unloadable", "cannot_score"]))
def test_p33_missing_or_failing_current_keeps_candidate_and_empties_current_cells(
        docs, ds_id, kind, tmp_path_factory):
    if kind == "missing":
        handle = baseline.load_current("6b", tmp_path_factory.getbasetemp() / "absent.joblib")
        expect = baseline.NO_BASELINE
    elif kind == "unloadable":
        p = tmp_path_factory.mktemp("bad") / "broken.joblib"
        p.write_bytes(b"not a joblib file")
        handle = baseline.load_current("6b", p)
        expect = baseline.BASELINE_FAILED
    else:
        handle = _failing_handle()
        expect = baseline.BASELINE_FAILED
    r = _eval(_dataset(ds_id, docs), current=handle)
    assert r["current_state"] == expect
    assert r["current"] is None and r["difference"] is None
    assert r["candidate"]["matched"]["cutoffs"] == {"medium": 0.5, "high": 0.8}
    assert r["candidate"]["matched"]["n_pos"] + r["candidate"]["matched"]["n_neg"] > 0
    if expect == baseline.BASELINE_FAILED:
        assert "6b" in r["current_detail"] and ds_id in r["current_detail"]


@PBT
@given(ds_id=st.sampled_from(DATASET_IDS), guard=st.booleans(), n=st.integers(1, 4))
def test_p33_baseline_states_from_the_scoring_helpers(ds_id, guard, n, tmp_path_factory):
    feats = [_feat(0.5)] * n
    missing = baseline.load_current("6b", tmp_path_factory.getbasetemp() / "absent.joblib")
    s = baseline.score_b(missing, feats, dataset=ds_id, guard=guard)
    assert s.status == baseline.NO_BASELINE and s.probs is None and s.abstain is None
    s = baseline.score_b(_failing_handle(), feats, dataset=ds_id, guard=guard)
    assert s.status == baseline.BASELINE_FAILED and s.probs is None
    assert "6b" in s.detail and ds_id in s.detail
