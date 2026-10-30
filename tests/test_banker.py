"""Tests for the banker / loan-officer approval app.

Covers the separation-of-duties invariant (a banker only sees and can only
decide on fraud-cleared cases) and that lending decisions are recorded with an
audit trail.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from fraud_detect import banker_app, webapp
from fraud_detect.models import CreditDecision, ReviewDecision
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_clean, generate_tampered


@pytest.fixture()
def apps(tmp_path):
    store = CaseStore(os.path.join(tmp_path, "cases.json"))
    banker_app.store = store
    webapp.store = store
    return banker_app.app, webapp.app, store


@pytest.fixture(scope="module")
def samples():
    d = tempfile.mkdtemp(prefix="banker_samples_")
    return {
        "clean": generate_clean(d).path,
        "tampered": generate_tampered(d).path,
    }


def _screen(store, path):
    report = run_pipeline(path)
    store.upsert_report(report)
    return report


def test_queue_shows_cleared_hides_uncleared(apps, samples):
    _, _, store = apps
    clean = _screen(store, samples["clean"])      # AUTO_CLEAR
    tam = _screen(store, samples["tampered"])      # HUMAN_REVIEW, not cleared
    q = TestClient(banker_app.app).get("/").text
    assert clean.case_id in q
    assert tam.case_id not in q


def test_cleared_application_is_viewable(apps, samples):
    _, _, store = apps
    clean = _screen(store, samples["clean"])
    r = TestClient(banker_app.app).get(f"/application/{clean.case_id}")
    assert r.status_code == 200
    assert "Lending decision" in r.text
    assert "Statement financials" in r.text


def test_uncleared_application_is_blocked(apps, samples):
    _, _, store = apps
    tam = _screen(store, samples["tampered"])
    r = TestClient(banker_app.app).get(f"/application/{tam.case_id}")
    assert r.status_code == 403


def test_decision_on_uncleared_case_is_refused(apps, samples):
    _, _, store = apps
    tam = _screen(store, samples["tampered"])
    r = TestClient(banker_app.app).post(
        f"/application/{tam.case_id}/decide",
        data={"decision": "approve", "officer": "X", "rationale": "nope"},
    )
    assert r.status_code == 409  # fraud not cleared


def test_approve_records_decision_and_audit(apps, samples):
    _, _, store = apps
    clean = _screen(store, samples["clean"])
    bc = TestClient(banker_app.app)
    r = bc.post(f"/application/{clean.case_id}/decide", data={
        "decision": "approve", "officer": "Daniel Lim",
        "rationale": "Strong cash flow.", "approved_amount": "SGD 450,000",
        "conditions": "Quarterly statements",
    })
    assert r.status_code == 200
    case = store.get(clean.case_id)
    assert case.credit.decision == CreditDecision.APPROVED
    assert case.credit.approved_amount == 450000.0
    assert case.credit.officer == "Daniel Lim"
    assert any(e["event"] == "CREDIT_APPROVED" for e in case.audit_log)


def test_analyst_clearing_makes_case_visible(apps, samples):
    _, _, store = apps
    tam = _screen(store, samples["tampered"])
    bc = TestClient(banker_app.app)
    assert tam.case_id not in bc.get("/").text          # hidden while uncleared
    store.record_decision(tam.case_id, ReviewDecision.APPROVED,
                          reviewer="Jane", reason="false alarm")
    assert tam.case_id in bc.get("/").text              # visible once cleared


def test_banker_never_overrides_fraud_rejection(apps, samples):
    """A fraud REJECTION must keep the case out of the banker stage entirely."""
    _, _, store = apps
    tam = _screen(store, samples["tampered"])
    store.record_decision(tam.case_id, ReviewDecision.REJECTED,
                          reviewer="Jane", reason="confirmed tampering")
    bc = TestClient(banker_app.app)
    assert tam.case_id not in bc.get("/").text
    assert bc.get(f"/application/{tam.case_id}").status_code == 403
    assert bc.post(f"/application/{tam.case_id}/decide", data={
        "decision": "approve", "officer": "X", "rationale": "y"}).status_code == 409
