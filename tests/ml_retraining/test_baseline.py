"""Task 18.1: failing tests for ``fraud_detect.ml.retraining.baseline`` (Current-model scoring).

Written before the module exists. Only pytest temporary folders and tiny fake models fitted here
are used; nothing under the real ``models/`` folder is read or written (Requirement 17.9).

PINNED INTERFACE (module ``fraud_detect.ml.retraining.baseline``)::

    STATUS_OK == "ok"
    NO_BASELINE == "no baseline"            # model file missing
    BASELINE_FAILED == "baseline failed"    # model cannot load or cannot score a dataset
    UNLOADABLE == "unloadable"              # ModelHandle.status for an existing file that will not load

    @dataclass(frozen=True) ModelHandle:
        node: str                  # "6a" | "6b"
        path: Path | None
        status: str                # STATUS_OK | NO_BASELINE | UNLOADABLE
        bundle: object | None      # the loaded object (dict for both nodes), else None
        sha256: str | None         # SHA-256 of the file, None when missing

    load_current(node, path) -> ModelHandle
        Never raises. Missing file -> NO_BASELINE (bundle None, sha256 None). Existing file that
        the live loader (ml_statement.load_model for "6b", ml_transaction.load_model for "6a")
        returns None for -> UNLOADABLE. Otherwise STATUS_OK with the bundle and the file SHA-256.
    wrap(node, bundle, path=None) -> ModelHandle        # an in-memory bundle (e.g. a Candidate), STATUS_OK

    with_guard(bundle, guard: bool) -> dict
        guard True: the bundle itself. guard False: a NEW dict (shallow copy) without "size_ranges";
        the input is never modified.

    @dataclass(frozen=True) BaselineScores:
        node, dataset: str
        status: str                # STATUS_OK | NO_BASELINE | BASELINE_FAILED
        detail: str | None         # for BASELINE_FAILED names the node and the dataset id
        probs: np.ndarray | None   # float, one per input; NaN where Node 6b abstained; None unless ok
        abstain: np.ndarray | None # bool, same length (all False for 6a); None unless ok
        sha256: str | None         # model file hash carried over from the handle

    score_b(handle, feats, *, dataset, guard=True) -> BaselineScores
        feats: sequence of feature dicts (the output of statement_features.extract_features).
        Each document is scored by fraud_detect.nodes.ml_statement.assess on with_guard(bundle, guard).
        Never raises; any load or scoring error becomes BASELINE_FAILED naming "6b" and `dataset`.
    score_a(handle, rows, *, dataset) -> BaselineScores
        rows: DataFrame in features_a.rows_to_frame schema (account_id, timestamp, direction,
        amount_sgd, balance_after_sgd, y, split, ...). One probability per row, same order. The
        builder and the columns come from fraud_detect.nodes.ml_transaction._feature_spec(bundle);
        the pipeline is bundle["pipeline"] fed exactly feature_frame[columns]. Never raises.
    deployed_cutoffs(handle) -> tuple[float, float]     # (medium, high) the live node runs with
        "6b": bundle["thresholds"] when present, else (0.5, 0.8). "6a": always (0.5, 0.8) (finding F9).
    matched_cutoffs(scores, is_genuine, rule) -> cutoffs.Cutoffs | None
        The decisive "matched" basis: cutoffs.choose_cutoffs(scores.probs, is_genuine, rule) on the
        Current model's (guard off) validation scores. None unless scores.status is STATUS_OK.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detect.ml import statement_features as sf
from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.retraining.cutoffs import CutoffRule, choose_cutoffs
from fraud_detect.nodes import ml_statement, ml_transaction

from fraud_detect.ml.retraining import baseline  # noqa: E402  (module does not exist yet)

DATASET = "heldout_seed"


# ----------------------------------------------------------------------------- fakes
def _feat(breaks=0.0, pages=2, text=60, money=70):
    f = {name: 0.0 for name in sf.FEATURE_NAMES}
    f.update(n_pages=pages, n_text_objects=text, n_money_tokens=money,
             balance_reconcile_breaks=breaks)
    return f


def _bundle_b(with_ranges=True, thresholds=True):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, len(sf.FEATURE_NAMES))), columns=sf.FEATURE_NAMES)
    y = (X["balance_reconcile_breaks"] > 0).astype(int)
    b = {"model": LogisticRegression().fit(X, y), "feature_names": list(sf.FEATURE_NAMES)}
    if thresholds:
        b["thresholds"] = {"medium": 0.35, "high": 0.75}
    if with_ranges:
        b["size_ranges"] = {"n_pages": [1, 3], "n_text_objects": [10, 100], "n_money_tokens": [10, 120]}
    return b


class _Spy:
    """A pipeline stand-in that records the columns it was fed."""

    def __init__(self, fail=False):
        self.seen = []
        self.fail = fail

    def predict_proba(self, X):
        if self.fail:
            raise RuntimeError("cannot score")
        self.seen.append(list(X.columns))
        p = 1.0 / (1.0 + np.exp(-X.iloc[:, 0].to_numpy(dtype=float) / 100.0))
        return np.column_stack([1 - p, p])


def _rows(n=12):
    return pd.DataFrame({
        "corpus_key": "c", "doc_id": "d1", "row_index": range(n), "account_id": "a1",
        "timestamp": [f"2026-03-{i + 1:02d}" for i in range(n)],
        "direction": ["CR" if i % 2 == 0 else "DR" for i in range(n)],
        "amount_sgd": [100.0 + 10 * i for i in range(n)],
        "balance_after_sgd": [5000.0 + 7 * i for i in range(n)],
        "y": [0] * n, "split": "test", "open_date": None})


def _handle_a(bundle):
    return baseline.wrap("6a", bundle)


# ----------------------------------------------------------------------------- loading
def test_missing_file_is_no_baseline(tmp_path):
    h = baseline.load_current("6b", tmp_path / "models" / "nope.joblib")
    assert (h.status, h.bundle, h.sha256) == (baseline.NO_BASELINE, None, None)
    assert baseline.NO_BASELINE == "no baseline"
    s = baseline.score_b(h, [_feat()], dataset=DATASET)
    assert s.status == baseline.NO_BASELINE and s.probs is None and s.abstain is None


def test_load_current_ok_records_file_hash(tmp_path):
    p = tmp_path / "statement_forgery_clf.joblib"
    joblib.dump(_bundle_b(), p)
    h = baseline.load_current("6b", p)
    assert h.status == baseline.STATUS_OK and isinstance(h.bundle, dict)
    assert h.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert h.node == "6b" and h.path == p


def test_unreadable_model_file_is_baseline_failed_naming_node_and_dataset(tmp_path):
    p = tmp_path / "broken.joblib"
    p.write_bytes(b"this is not a joblib file")
    h = baseline.load_current("6b", p)
    assert h.status == baseline.UNLOADABLE and h.bundle is None
    s = baseline.score_b(h, [_feat()], dataset=DATASET)
    assert s.status == baseline.BASELINE_FAILED == "baseline failed"
    assert "6b" in s.detail and DATASET in s.detail
    assert s.probs is None


def test_node_6a_loader_is_used_for_6a(tmp_path):
    p = tmp_path / "txn.joblib"
    p.write_bytes(b"garbage")
    h = baseline.load_current("6a", p)
    assert h.status == baseline.UNLOADABLE
    s = baseline.score_a(h, _rows(), dataset="forgery_test")
    assert s.status == baseline.BASELINE_FAILED and "6a" in s.detail and "forgery_test" in s.detail


# ----------------------------------------------------------------------------- Node 6b
def test_6b_scores_equal_live_assess_with_guard_on():
    bundle = _bundle_b()
    feats = [_feat(-1.0), _feat(2.0), _feat(1.5, pages=20), _feat(0.2, money=500)]
    s = baseline.score_b(baseline.wrap("6b", bundle), feats, dataset=DATASET)
    assert s.status == baseline.STATUS_OK
    live = [ml_statement.assess(f, bundle) for f in feats]
    assert s.abstain.tolist() == [r["abstain"] for r in live] == [False, False, True, True]
    for got, r in zip(s.probs, live):
        if r["abstain"]:
            assert np.isnan(got)
        else:
            assert got == pytest.approx(r["prob"], abs=1e-12)


def test_guard_off_is_a_copy_without_size_ranges_and_does_not_touch_the_original():
    bundle = _bundle_b()
    off = baseline.with_guard(bundle, False)
    assert off is not bundle and "size_ranges" not in off
    assert "size_ranges" in bundle                       # original untouched
    assert off["model"] is bundle["model"] and off["thresholds"] == bundle["thresholds"]
    assert baseline.with_guard(bundle, True) is bundle


def test_6b_guard_off_scores_every_document_like_an_unguarded_bundle():
    bundle = _bundle_b()
    feats = [_feat(1.0, pages=30), _feat(-1.0, pages=2), _feat(0.5, money=900)]
    s = baseline.score_b(baseline.wrap("6b", bundle), feats, dataset=DATASET, guard=False)
    assert not s.abstain.any() and not np.isnan(s.probs).any()
    bare = {k: v for k, v in bundle.items() if k != "size_ranges"}
    assert s.probs.tolist() == pytest.approx([ml_statement.assess(f, bare)["prob"] for f in feats])
    on = baseline.score_b(baseline.wrap("6b", bundle), feats, dataset=DATASET, guard=True)
    assert on.abstain.tolist() == [True, False, True]


def test_6b_scoring_error_is_baseline_failed_not_an_exception():
    bundle = _bundle_b()
    bad = [{"n_pages": 2}]                                # lacks the other feature names
    s = baseline.score_b(baseline.wrap("6b", bundle), bad, dataset="forgery_ood")
    assert s.status == baseline.BASELINE_FAILED
    assert "6b" in s.detail and "forgery_ood" in s.detail and s.probs is None


# ----------------------------------------------------------------------------- Node 6a
def test_feature_spec_pinning_behaviour():
    """Fails if the private ml_transaction._feature_spec is renamed or changes behaviour."""
    assert callable(getattr(ml_transaction, "_feature_spec", None))
    assert list(inspect.signature(ml_transaction._feature_spec).parameters) == ["model"]
    pdf = {"feature_set": "pdf", "feature_columns": ["amount_sgd", "log_amount"], "pipeline": object()}
    build, cols = ml_transaction._feature_spec(pdf)
    assert build is tf.build_pdf_features and cols == ["amount_sgd", "log_amount"]
    legacy = {"feature_columns": ["amount_sgd"], "pipeline": object()}
    build, cols = ml_transaction._feature_spec(legacy)
    assert build is tf.build_features and cols == ["amount_sgd"]
    build, cols = ml_transaction._feature_spec({"pipeline": object()})
    assert build is tf.build_features and cols == tf.FEATURE_COLUMNS


def test_6a_is_fed_exactly_the_columns_it_was_trained_on():
    spy = _Spy()
    cols = ["amount_sgd", "log_amount", "direction"]
    bundle = {"pipeline": spy, "feature_set": "pdf", "feature_columns": cols}
    rows = _rows()
    s = baseline.score_a(_handle_a(bundle), rows, dataset="forgery_test")
    assert s.status == baseline.STATUS_OK and len(s.probs) == len(rows)
    assert spy.seen == [cols]
    assert not s.abstain.any()
    # same numbers as the live path would give for the same frame
    feat = tf.build_pdf_features(rows.assign(is_fraud=0, split="train"))
    assert s.probs.tolist() == pytest.approx(spy.predict_proba(feat[cols])[:, 1].tolist())


def test_6a_legacy_full_feature_bundle_uses_build_features_columns():
    spy = _Spy()
    bundle = {"pipeline": spy, "feature_columns": ["amount_sgd", "log_amount"]}
    s = baseline.score_a(_handle_a(bundle), _rows(), dataset="test")
    assert s.status == baseline.STATUS_OK
    assert spy.seen == [["amount_sgd", "log_amount"]]


def test_6a_scoring_failure_is_baseline_failed():
    bundle = {"pipeline": _Spy(fail=True), "feature_set": "pdf", "feature_columns": ["amount_sgd"]}
    s = baseline.score_a(_handle_a(bundle), _rows(), dataset="heldout_layout")
    assert s.status == baseline.BASELINE_FAILED
    assert "6a" in s.detail and "heldout_layout" in s.detail


# ----------------------------------------------------------------------------- cutoffs
def test_deployed_cutoffs():
    assert baseline.deployed_cutoffs(baseline.wrap("6b", _bundle_b())) == (0.35, 0.75)
    assert baseline.deployed_cutoffs(baseline.wrap("6b", _bundle_b(thresholds=False))) == (0.5, 0.8)
    a = {"pipeline": _Spy(), "feature_set": "pdf", "feature_columns": ["amount_sgd"],
         "thresholds": {"medium": 0.1, "high": 0.2}}
    assert baseline.deployed_cutoffs(_handle_a(a)) == (0.5, 0.8)      # the live node ignores the file


def test_matched_cutoffs_use_guard_off_validation_scores_and_the_same_rule():
    handle = baseline.wrap("6b", _bundle_b())
    rng = np.random.default_rng(3)
    feats = [_feat(float(v), pages=int(p)) for v, p in
             zip(rng.normal(size=40), rng.choice([2, 2, 2, 25], size=40))]
    is_genuine = np.array([i % 2 == 0 for i in range(40)])
    scores = baseline.score_b(handle, feats, dataset="validation", guard=False)
    rule = CutoffRule(target_fa_medium=20.0, target_fa_high=10.0)
    got = baseline.matched_cutoffs(scores, is_genuine, rule)
    want = choose_cutoffs(scores.probs, is_genuine, rule)
    assert (got.medium, got.high) == (want.medium, want.high)
    assert got.high >= got.medium


def test_matched_cutoffs_none_without_a_baseline(tmp_path):
    h = baseline.load_current("6b", tmp_path / "missing.joblib")
    s = baseline.score_b(h, [_feat()], dataset="validation", guard=False)
    assert baseline.matched_cutoffs(s, np.array([True]), CutoffRule()) is None


def test_baseline_module_does_not_import_torch_or_write_files():
    src = Path(baseline.__file__).read_text()
    assert "torch" not in src
    for token in ("open(", ".write_text(", ".write_bytes(", "joblib.dump"):
        assert token not in src
