"""Node 6a's per-transaction model is switched off in the live app by default.

Covers the switch (``load_live_model``), that the pipeline then raises no
``ML_TXN_FRAUD_LIKELIHOOD`` findings while the GNN lookup in the same node still
works, and that a model bundle is fed the feature set it was trained on.
"""
from __future__ import annotations

import joblib
import pytest

from fraud_detect.ml import transaction_features as tf
from fraud_detect.nodes import ml_transaction as mt
from fraud_detect.pipeline import run_pipeline
from fraud_detect.synthetic import generate_clean


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "txn.joblib"
    joblib.dump({"pipeline": "stub", "feature_set": "pdf",
                 "feature_columns": tf.PDF_FEATURE_COLUMNS}, path)
    return str(path)


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "garbage"])
def test_live_model_is_off_unless_explicitly_enabled(model_file, value):
    env = {} if value is None else {mt.ENABLE_ENV: value}
    assert mt.load_live_model(model_file, env=env) is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " on "])
def test_live_model_loads_when_enabled(model_file, value):
    bundle = mt.load_live_model(model_file, env={mt.ENABLE_ENV: value})
    assert bundle is not None and bundle["feature_set"] == "pdf"


def test_enabled_but_missing_file_is_still_fail_safe(tmp_path):
    assert mt.load_live_model(str(tmp_path / "absent.joblib"), env={mt.ENABLE_ENV: "1"}) is None


def test_web_app_starts_with_the_model_off(monkeypatch):
    import importlib

    monkeypatch.delenv(mt.ENABLE_ENV, raising=False)
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    import fraud_detect.webapp as webapp
    webapp = importlib.reload(webapp)
    assert webapp.TXN_MODEL is None


def test_gnn_lookup_still_runs_with_the_model_off(tmp_path):
    gen = generate_clean(str(tmp_path))
    store = {"account_scores": {"201812345A": 0.9}, "edge_scores": {}, "meta": {}}
    report = run_pipeline(gen.path, txn_model=None, gnn_scores=store)
    codes = {f.code for f in report.findings}
    assert "ML_TXN_GNN_RISK" in codes
    assert "ML_TXN_FRAUD_LIKELIHOOD" not in codes


def test_feature_spec_follows_the_bundle():
    build, cols = mt._feature_spec({"feature_set": "pdf", "feature_columns": tf.PDF_FEATURE_COLUMNS})
    assert build is tf.build_pdf_features and cols == tf.PDF_FEATURE_COLUMNS
    build, cols = mt._feature_spec({"feature_columns": tf.FEATURE_COLUMNS})   # older bundles
    assert build is tf.build_features and cols == tf.FEATURE_COLUMNS
    build, cols = mt._feature_spec(object())                                   # bare estimator
    assert build is tf.build_features and cols == tf.FEATURE_COLUMNS


def test_pdf_features_ignore_time_of_day():
    import pandas as pd

    base = {"account_id": "A", "direction": "CR", "amount_sgd": 100.0, "balance_after_sgd": 100.0,
            "is_fraud": 0, "split": "train", "open_date": None}
    rows = pd.DataFrame([{**base, "timestamp": "2025-01-01T01:00:00Z"},
                         {**base, "timestamp": "2025-01-01T23:00:00Z"}])
    feats = tf.build_pdf_features(rows)
    assert list(feats.columns[: len(tf.PDF_FEATURE_COLUMNS)]) == tf.PDF_FEATURE_COLUMNS
    assert feats["gap_hours"].tolist() == [0.0, 0.0]   # same date -> no gap, as on a statement
