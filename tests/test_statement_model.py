"""Node 6b (statement forgery): the reconciliation feature, the model-input list,
and the node's thresholds and abstain guard.

Uses small hand-built inputs and stub models, so it needs no trained model file.
"""
from __future__ import annotations

import numpy as np
import pytest

from fraud_detect.ml import statement_features as sf
from fraud_detect.models import Severity
from fraud_detect.nodes import ml_statement as ms


# --------------------------------------------------------------------------- #
# Balance reconciliation feature
# --------------------------------------------------------------------------- #
def test_reconcile_clean_rows_have_no_breaks():
    lines = [
        "BALANCE BROUGHT FORWARD 6,361.33",
        "01/07/2026 ATM WITHDRAWAL 133 200.00 6,161.33",          # unsigned, out
        "02/07/2026 FAST TRANSFER IN REF1 100.00 6,261.33",       # unsigned, in
    ]
    assert sf._reconcile_rows(lines) == (0, 0.0)


def test_reconcile_catches_an_altered_amount_in_the_first_row():
    """The opening balance comes from the brought-forward line, so row 1 is checked."""
    lines = ["BALANCE BROUGHT FORWARD 6,361.33",
             "01/07/2026 ATM WITHDRAWAL 133 500.00 6,161.33"]       # 500 edited from 200
    breaks, rate = sf._reconcile_rows(lines)
    assert breaks == 1 and rate == 1.0


def test_reconcile_ignores_the_header_summary_line():
    """A summary line of several numbers and no date must not be read as a row."""
    lines = ["6,361.33 7,668.44 1,750.74 443.63",
             "BALANCE BROUGHT FORWARD 6,361.33",
             "01/07/2026 ATM WITHDRAWAL 133 200.00 6,161.33"]
    assert sf._reconcile_rows(lines)[0] == 0


@pytest.mark.parametrize("lines", [
    ["BALANCE BROUGHT FORWARD SGD 28,249.88",                       # signed amounts, SGD prefix
     "2026-05-01 NETS/CARD NIMBUS 1713 -SGD 13.47 SGD 28,236.41"],
    ["BALANCE BROUGHT FORWARD 7,021.27 CR",                         # CR suffix on balances
     "01-05-26 ATM WITHDRAWAL 723 50.00 6,971.27 CR"],
    ["BALANCE BROUGHT FORWARD S$3,703.86",                          # S$ prefix, day-month dates
     "1 Sep GIRO FIBRENEST REF732555 S$76.38 S$3,627.48"],
    ["Opening Balance: 42,000.00",                                  # this project's own layout
     "04/01/2025 M1 Limited -SGD2,945.32 SGD39,054.68"],
])
def test_reconcile_handles_each_layout_style(lines):
    assert sf._reconcile_rows(lines) == (0, 0.0)


def test_reconcile_with_no_rows_is_zero():
    assert sf._reconcile_rows(["Statement Period: 01/07/2026 - 31/07/2026"]) == (0, 0.0)


# --------------------------------------------------------------------------- #
# Model inputs
# --------------------------------------------------------------------------- #
def test_layout_artefact_features_are_not_model_inputs():
    assert "duplicate_amount_overlap" not in sf.FEATURE_NAMES
    assert "n_distinct_date_formats" not in sf.FEATURE_NAMES
    assert set(sf.SIZE_FEATURES) <= set(sf.FEATURE_NAMES)


def test_extraction_still_reports_the_features_the_evidence_text_uses(tmp_path):
    from fraud_detect.synthetic import generate_clean

    feats = sf.extract_features(generate_clean(str(tmp_path)).path)
    assert "duplicate_amount_overlap" in feats and "n_distinct_date_formats" in feats
    assert set(sf.FEATURE_NAMES) <= set(feats)


def test_clean_synthetic_statement_has_no_balance_breaks(tmp_path):
    from fraud_detect.synthetic import generate_clean

    feats = sf.extract_features(generate_clean(str(tmp_path)).path)
    assert feats["balance_reconcile_breaks"] == 0


def test_altered_balance_is_detected_in_a_generated_statement(tmp_path):
    from fraud_detect.synthetic import generate_tampered

    feats = sf.extract_features(generate_tampered(str(tmp_path), seed=42, issues=["ALTERED_BALANCE"]).path)
    assert feats["balance_reconcile_breaks"] >= 1


# --------------------------------------------------------------------------- #
# Node thresholds and abstain guard
# --------------------------------------------------------------------------- #
class _Stub:
    """Stand-in classifier returning a fixed probability and recording its input."""

    def __init__(self, prob):
        self.prob, self.seen = prob, None

    def predict_proba(self, X):
        self.seen = list(X.columns)
        return np.array([[1 - self.prob, self.prob]])


def _feats(**over):
    base = {n: 1.0 for n in sf.FEATURE_NAMES}
    base.update({"n_pages": 1.0, "n_text_objects": 2000.0, "n_money_tokens": 60.0,
                 "n_distinct_date_formats": 2.0, "duplicate_amount_overlap": 3.0})
    base.update(over)
    return base


def _bundle(prob, **extra):
    return {"model": _Stub(prob), "feature_names": list(sf.FEATURE_NAMES),
            "thresholds": {"medium": 0.7, "high": 0.9},
            "size_ranges": {"n_pages": [1, 3], "n_text_objects": [1100, 6300], "n_money_tokens": [23, 179]},
            **extra}


@pytest.mark.parametrize("prob, expected", [(0.95, "HIGH"), (0.9, "HIGH"), (0.75, "MEDIUM"),
                                            (0.7, "MEDIUM"), (0.69, None), (0.5, None)])
def test_bundle_thresholds_set_the_severity(prob, expected):
    r = ms.assess(_feats(), _bundle(prob))
    assert not r["abstain"] and r["prob"] == pytest.approx(prob)
    sev = "HIGH" if r["prob"] >= r["high"] else "MEDIUM" if r["prob"] >= r["medium"] else None
    assert sev == expected


@pytest.mark.parametrize("over", [{"n_pages": 6.0}, {"n_text_objects": 9000.0}, {"n_money_tokens": 400.0}])
def test_node_abstains_on_documents_larger_than_its_training_data(over):
    r = ms.assess(_feats(**over), _bundle(0.99))
    assert r["abstain"] and r["prob"] is None and len(r["outside"]) == 1


def test_forgery_signals_do_not_trigger_the_guard():
    """Many balance breaks are evidence of forgery, not of an unfamiliar document."""
    r = ms.assess(_feats(balance_reconcile_breaks=40.0, balance_reconcile_rate=0.9, has_editing_software=1.0),
                  _bundle(0.99))
    assert not r["abstain"]


def test_model_receives_exactly_the_bundles_feature_list():
    b = _bundle(0.8, feature_names=["n_pages", "balance_reconcile_breaks"])
    ms.assess(_feats(), b)
    assert b["model"].seen == ["n_pages", "balance_reconcile_breaks"]


def test_older_bundles_keep_the_original_cutoffs_and_have_no_guard():
    old = {"model": _Stub(0.85), "feature_names": list(sf.FEATURE_NAMES)}     # no thresholds / ranges
    r = ms.assess(_feats(n_pages=9.0), old)
    assert not r["abstain"] and (r["medium"], r["high"]) == (0.5, 0.8)


def test_bare_estimator_is_still_accepted():
    r = ms.assess(_feats(), _Stub(0.6))
    assert not r["abstain"] and r["prob"] == pytest.approx(0.6)


def test_analyze_emits_no_finding_when_abstaining_or_below_threshold(tmp_path):
    from fraud_detect.nodes import ingestion
    from fraud_detect.synthetic import generate_clean

    ing = ingestion.ingest(generate_clean(str(tmp_path)).path)
    real = sf.extract_features(ing.source_path)
    too_big = _bundle(0.99, size_ranges={"n_pages": [1, 1]})                  # clean sample has 2 pages
    assert not ms.analyze(ing, too_big).findings
    assert not ms.analyze(ing, _bundle(0.6, size_ranges={})).findings         # below medium 0.7
    flagged = ms.analyze(ing, _bundle(0.95, size_ranges={})).findings
    assert [(f.code, f.severity) for f in flagged] == [("ML_STATEMENT_FORGERY_LIKELIHOOD", Severity.HIGH)]
    assert real["n_pages"] >= 2


def test_analyze_without_a_model_is_silent(tmp_path):
    from fraud_detect.nodes import ingestion
    from fraud_detect.synthetic import generate_clean

    assert not ms.analyze(ingestion.ingest(generate_clean(str(tmp_path)).path), None).findings
