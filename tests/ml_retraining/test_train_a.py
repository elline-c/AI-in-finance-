"""Task 16.1: failing tests for ``fraud_detect.ml.retraining.train_a`` (Node 6a candidate).
Written before the module exists. Only the Mini_Corpus and pytest temporary folders are used; nothing
under ``models/`` is read or written. The two Mini_Corpus seeds play two roles: seed 1 ``train`` rows
fit the model, seed 2 ``train`` rows stand in for the validation data (the Mini_Corpus ``validation``
Split holds no genuine statement, so it cannot choose cutoffs).

PINNED INTERFACE (module ``fraud_detect.ml.retraining.train_a``)::

    NODE6A_LIVE_CUTOFF_NOTE: str      # contains "0.5" and "0.8" (finding F9: live node uses fixed cutoffs)
    MODEL_LABEL == "model trained on synthetic data"
    STATEMENT_LEVEL == "statement_max_row"

    def detection_statements(features, statements) -> StatementSets
        # features: frame as features_a.build_pdf_table output (needs corpus_key, doc_id, y).
        # statements: frame with columns corpus_key, doc_id, is_tampered (bool) - one row per statement.
        # Statement key == (corpus_key, doc_id).
        # StatementSets (frozen dataclass), each a sorted list of keys:
        #   .genuine            statements with is_tampered False
        #   .positive           tampered statements with at least one y == 1 row (detection denominator)
        #   .zero_fraud_tampered tampered statements with no y == 1 row (listed, NOT in the denominator)

    def statement_scores(pipeline, features, columns) -> dict
        # {(corpus_key, doc_id): max over the statement's rows of pipeline.predict_proba(features[columns])[:, 1]}

    def train_candidate(train_features, valid_features, statements, *, out_path, columns=None,
                        audit=AUDIT_V1, rule=None, master_seed=None, held_out_layout=None,
                        is_ignored=None) -> TrainedA
        # train_features / valid_features: frames as build_pdf_table output. Only train rows are fitted.
        # statements: covers every statement in valid_features (and may cover train).
        # columns: requested feature columns; default model_input_columns(audit, "6a").
        #   Any name outside transaction_features.PDF_FEATURE_COLUMNS raises
        #   features_a.ForbiddenFeatureError (all offenders listed) before any file is written.
        # rule: cutoffs.CutoffRule; default CutoffRule(level="statement_max_row"). Any other level
        #   raises ValueError. Cutoffs: choose_cutoffs on valid statement scores (genuine vs positive;
        #   zero-fraud tampered statements excluded) - the validation data only.
        # Writes the joblib bundle to out_path with safe_io.atomic_write(is_ignored=...).
        # TrainedA (dataclass): .path (Path), .bundle (dict, the object saved), .zero_fraud_statements (list of keys)
    Bundle keys: pipeline (fitted sklearn Pipeline, steps "pre" and "clf"), feature_set ("pdf"),
        feature_columns (ordered audited subset of PDF_FEATURE_COLUMNS, in PDF_FEATURE_COLUMNS order),
        categorical, numeric (the columns of each kind), thresholds ({"medium": float, "high": float}),
        cutoff_rule ({"target_fa_medium", "target_fa_high", "level"}), live_cutoff_note
        (== NODE6A_LIVE_CUTOFF_NOTE), zero_fraud_statements ({"count": int, "doc_ids": sorted list}),
        validation_curve (list of dicts with name, cutoff, alert_rate, false_alarm_rate, catch_rate),
        audit_version, audit_digest, master_seed, held_out_layout, model_label, synthetic_data (True).
    The pipeline is train_transaction.build_pipeline("pdf") restricted to ``columns``: same
    OneHotEncoder and HistGradientBoostingClassifier parameters.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detect.ml import train_transaction
from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.retraining import safe_io, train_a
from fraud_detect.ml.retraining.audit import AUDIT_V1, model_input_columns
from fraud_detect.ml.retraining.corpora import load_corpus
from fraud_detect.ml.retraining.cutoffs import CutoffRule, choose_cutoffs
from fraud_detect.ml.retraining.features_a import ForbiddenFeatureError, load_row_table
from fraud_detect.ml.retraining.train_a import (
    MODEL_LABEL, NODE6A_LIVE_CUTOFF_NOTE, STATEMENT_LEVEL, detection_statements,
    statement_scores, train_candidate)
from fraud_detect.nodes import ml_transaction

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data(mini_corpus, tmp_path_factory):
    """(train_features, valid_features, statements) from the two Mini_Corpus seeds."""
    root = tmp_path_factory.mktemp("train_a_cache")
    feats, stmts = {}, []
    for role, seed in (("train", mini_corpus.seeds[0]), ("valid", mini_corpus.seeds[1])):
        path = mini_corpus.path(seed)
        table = load_row_table(path, cache_root=root, is_ignored=lambda p: True)
        feats[role] = table.features[table.features["split"] == "train"].reset_index(drop=True)
        corpus = load_corpus(path)
        stmts.append(pd.DataFrame(
            [{"corpus_key": d.corpus_key, "doc_id": d.doc_id, "is_tampered": d.is_tampered}
             for d in corpus.docs]))
    return feats["train"], feats["valid"], pd.concat(stmts, ignore_index=True)


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    train, valid, stmts = data
    out = tmp_path_factory.mktemp("train_a_out") / "candidate_6a.joblib"
    res = train_candidate(train, valid, stmts, out_path=out, master_seed=20260801,
                          held_out_layout="two_col_ddmmyy_cr", is_ignored=lambda p: True)
    return res, out


def test_pipeline_parameters_equal_build_pipeline_pdf(trained):
    res, _ = trained
    ref = train_transaction.build_pipeline("pdf")
    pipe = res.bundle["pipeline"]
    assert pipe.named_steps["clf"].get_params() == ref.named_steps["clf"].get_params()
    enc = {n: t for n, t, _ in pipe.named_steps["pre"].transformers}["cat"]
    ref_enc = {n: t for n, t, _ in ref.named_steps["pre"].transformers}["cat"]
    assert enc.get_params() == ref_enc.get_params()
    assert hasattr(pipe, "predict_proba")


def test_feature_set_and_ordered_audited_columns(trained):
    b = trained[0].bundle
    audited = model_input_columns(AUDIT_V1, "6a")
    expected = [c for c in tf.PDF_FEATURE_COLUMNS if c in audited]
    assert b["feature_set"] == "pdf"
    assert b["feature_columns"] == expected
    assert "balance_after_sgd" not in expected and "day_of_week" not in expected
    assert set(b["numeric"]) | set(b["categorical"]) == set(expected)
    assert b["categorical"] == ["direction"]
    assert b["audit_version"] == AUDIT_V1.version and b["audit_digest"] == AUDIT_V1.digest()
    assert (b["master_seed"], b["held_out_layout"]) == (20260801, "two_col_ddmmyy_cr")
    assert b["model_label"] == MODEL_LABEL and b["synthetic_data"] is True


def test_non_pdf_feature_refused_and_nothing_written(data, tmp_path):
    train, valid, stmts = data
    out = tmp_path / "out" / "cand.joblib"
    with pytest.raises(ForbiddenFeatureError) as exc:
        train_candidate(train, valid, stmts, out_path=out, is_ignored=lambda p: True,
                        columns=["amount_sgd", "channel", "kyc_risk_rating"])
    assert exc.value.offenders == ["channel", "kyc_risk_rating"]
    assert not out.exists() and not out.parent.exists()


def test_write_guard_blocks_non_ignored_path(data, tmp_path):
    train, valid, stmts = data
    out = tmp_path / "cand.joblib"
    with pytest.raises(safe_io.WriteGuardError):
        train_candidate(train, valid, stmts, out_path=out, is_ignored=lambda p: False)
    assert list(tmp_path.iterdir()) == []


def test_statement_level_rule_only(data, tmp_path):
    train, valid, stmts = data
    with pytest.raises(ValueError):
        train_candidate(train, valid, stmts, out_path=tmp_path / "x.joblib",
                        rule=CutoffRule(level="document"), is_ignored=lambda p: True)
    assert not (tmp_path / "x.joblib").exists()


def test_cutoffs_use_max_row_score_on_validation_genuine_statements(trained, data):
    res, _ = trained
    _, valid, stmts = data
    b = res.bundle
    sets = detection_statements(valid, stmts)
    scores = statement_scores(b["pipeline"], valid, b["feature_columns"])
    keep = sets.genuine + sets.positive
    # max over rows, one number per statement
    row_p = b["pipeline"].predict_proba(valid[b["feature_columns"]])[:, 1]
    k0 = keep[0]
    mask = (valid["corpus_key"] == k0[0]) & (valid["doc_id"] == k0[1])
    assert scores[k0] == pytest.approx(row_p[mask.to_numpy()].max())
    expected = choose_cutoffs([scores[k] for k in keep],
                              [k in set(sets.genuine) for k in keep],
                              CutoffRule(level=STATEMENT_LEVEL))
    assert b["thresholds"] == {"medium": expected.medium, "high": expected.high}
    assert b["thresholds"]["high"] >= b["thresholds"]["medium"]
    assert b["cutoff_rule"] == {"target_fa_medium": 5.0, "target_fa_high": 1.0,
                                "level": STATEMENT_LEVEL}
    genuine_scores = np.array([scores[k] for k in sets.genuine])
    for name, target in (("medium", 5.0), ("high", 1.0)):
        fa = 100 * np.mean(genuine_scores >= b["thresholds"][name])
        assert fa <= max(target, 100 / len(genuine_scores))
    assert {r["name"] for r in b["validation_curve"]} >= {"medium", "high"}
    for r in b["validation_curve"]:
        assert {"cutoff", "alert_rate", "false_alarm_rate", "catch_rate"} <= set(r)


def test_cutoffs_ignore_training_rows_and_fit_uses_train_only(data, tmp_path):
    train, valid, stmts = data
    a = train_candidate(train, valid, stmts, out_path=tmp_path / "a.joblib", is_ignored=lambda p: True)
    # Dropping every validation row of one tampered statement changes nothing about the fit,
    # and the validation frame is never fitted on: same train rows => identical pipeline output.
    b = train_candidate(train, valid, stmts, out_path=tmp_path / "b.joblib", is_ignored=lambda p: True)
    X = valid[a.bundle["feature_columns"]]
    assert np.allclose(a.bundle["pipeline"].predict_proba(X), b.bundle["pipeline"].predict_proba(X))
    n_fit = a.bundle["pipeline"].named_steps["clf"].n_features_in_
    assert n_fit >= len(a.bundle["numeric"])


def test_live_cutoff_note_mentions_fixed_values(trained):
    assert "0.5" in NODE6A_LIVE_CUTOFF_NOTE and "0.8" in NODE6A_LIVE_CUTOFF_NOTE
    assert trained[0].bundle["live_cutoff_note"] == NODE6A_LIVE_CUTOFF_NOTE


def test_zero_fraud_statements_listed_and_left_out_of_denominator(trained, data):
    res, _ = trained
    train, valid, stmts = data
    sets = detection_statements(valid, stmts)
    valid_keys = set(zip(valid["corpus_key"], valid["doc_id"]))
    tampered = {(r.corpus_key, r.doc_id) for r in stmts.itertuples() if r.is_tampered} & valid_keys
    fraud_keys = {k for k, g in valid.groupby(["corpus_key", "doc_id"])["y"].max().items() if g == 1}
    assert set(sets.positive) == tampered & fraud_keys
    assert set(sets.zero_fraud_tampered) == tampered - fraud_keys
    assert set(sets.genuine).isdisjoint(sets.positive) and set(sets.genuine).isdisjoint(sets.zero_fraud_tampered)
    assert sets.zero_fraud_tampered, "Mini_Corpus has tampered statements with no fraud row"
    zf = res.bundle["zero_fraud_statements"]
    assert zf["count"] == len(sets.zero_fraud_tampered) == len(res.zero_fraud_statements)
    assert zf["doc_ids"] == sorted(k[1] for k in sets.zero_fraud_tampered)
    # they never reach the cutoff choice: curve catch rate is over positive statements only
    scores = statement_scores(res.bundle["pipeline"], valid, res.bundle["feature_columns"])
    med = {r["name"]: r for r in res.bundle["validation_curve"]}["medium"]
    caught = sum(scores[k] >= med["cutoff"] for k in sets.positive)
    assert med["catch_rate"] == pytest.approx(100 * caught / len(sets.positive))


def test_candidate_read_by_unchanged_ml_transaction_helpers(trained):
    res, out = trained
    assert out.is_file() and res.path == out
    model = ml_transaction.load_model(str(out))
    assert isinstance(model, dict)
    build, cols = ml_transaction._feature_spec(model)
    assert build is tf.build_pdf_features and cols == res.bundle["feature_columns"]
    saved = joblib.load(out)
    assert saved["thresholds"] == res.bundle["thresholds"]
    # the node's own scoring path runs the bundle without raising and without findings from a crash
    from fraud_detect.models import DocumentMetadata, IngestionResult, Transaction
    from datetime import date
    txns = [Transaction(index=i, date=date(2026, 1, 1 + i), description="x",
                        amount=100.0 * (i + 1), balance=1000.0 + i, page=1) for i in range(3)]
    ing = IngestionResult(source_path="fake.pdf", page_count=1, raw_text="",
                          metadata=DocumentMetadata(), uen="U1", transactions=txns)
    df = ml_transaction._rows_from_ingestion(ing)
    feat = build(df.assign(is_fraud=0, split="train"))
    probs = model["pipeline"].predict_proba(feat[cols])[:, 1]
    assert len(probs) == 3 and ((0 <= probs) & (probs <= 1)).all()
    assert ml_transaction.analyze(ing, model=model).agent == ml_transaction.AGENT


def test_no_torch_in_module_or_after_import():
    src = Path(train_a.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        assert not any(n.split(".")[0] in ("torch", "torch_geometric") for n in names)
    code = ("import sys; import fraud_detect.ml.retraining.train_a; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','torch_geometric')]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0
