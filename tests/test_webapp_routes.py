"""HTTP-route tests for the analyst console (fraud_detect/webapp.py).

These exercise the webapp-level controls FEAT-003 wired up, at the FastAPI
route layer using ``fastapi.testclient.TestClient``:

  * ITEM 1 - the server-side "I have reviewed all pipeline stages" gate on
    POST /case/{id}/decide (400 without the acknowledgement; the decision must
    stay PENDING), and acceptance once acknowledged with a valid reviewer.
  * ITEM 5 - reviewer-roster validation on /decide and /request-info (an unknown
    reviewer is rejected with 400; a roster reviewer is accepted).
  * ITEM 4 - POST /case/{id}/revert: a fresh decision reverts to PENDING (303);
    once the 30-minute window has elapsed the store raises ValueError, mapped to
    HTTP 400.
  * ITEM 6 - POST /case/{id}/escalate: a blank internal note is rejected (400);
    a valid note + roster senior officer + roster reviewer records the senior
    assignment and redirects (303).
  * ITEM 7 - POST /upload and POST /screen-sample are gone, so they return
    404/405.

Isolation: a fixture monkeypatches ``fraud_detect.webapp.store`` to a CaseStore
on a pytest tmp path and seeds a single hand-built RiskReport via
``upsert_report``, so these tests never touch the real data/cases.json. Valid
reviewer / senior-officer values come from the live rosters
(reviewer_choices() / senior_choices()), so the tests track the roster.

NOTE: importing fraud_detect.webapp pulls in FastAPI, which is not installed in
the planning sandbox (no internet); these route tests therefore run on the
developer's laptop after ``pip install -r requirements.txt``. They are written
green-by-construction and mirror the style of tests/test_detection.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fraud_detect.models import (
    ConfidenceBand,
    Finding,
    ReviewDecision,
    RiskBand,
    RiskReport,
    Routing,
    Severity,
)
from fraud_detect.reviewers import reviewer_choices, senior_choices
from fraud_detect.store import CaseStore, REVERT_WINDOW_MINUTES

CASE_ID = "CASE-ROUTE-TEST"

# A roster reviewer / senior officer picked from the live rosters so the tests
# stay valid if a name is renamed. The display strings use a plain hyphen and no
# HTML-special characters, so webapp._scrub() leaves them byte-for-byte intact
# and is_valid_reviewer()/is_valid_senior() match exactly.
VALID_REVIEWER = reviewer_choices()[0]
VALID_SENIOR = senior_choices()[0]
UNKNOWN_REVIEWER = "Nobody McTamper - Impostor"


def _seed_report() -> RiskReport:
    """A minimal but realistic HUMAN_REVIEW report with findings across a couple
    of pipeline stages, so _group_by_stage returns more than one stage and the
    acknowledgement message's stage count is meaningful."""
    return RiskReport(
        case_id=CASE_ID,
        source_path="data/uploads/route_test.pdf",
        account_holder="Harbourline Logistics Pte Ltd",
        uen="201812345A",
        statement_period="2024-01-01 to 2024-06-30",
        risk_score=60,
        risk_band=RiskBand.HIGH,
        confidence_score=55,
        confidence_band=ConfidenceBand.HIGH,
        routing=Routing.HUMAN_REVIEW,
        routing_reason="Mixed findings require a human reviewer.",
        findings=[
            Finding(
                agent="Node2-VisualForensics",
                code="VF_FONT_MISMATCH",
                title="Font mismatch on an amount",
                severity=Severity.HIGH,
                weight=20.0,
                evidence="Row 4 amount uses a different font from the rest.",
                page=1,
                field="row 4",
            ),
            Finding(
                agent="Node3-FinancialLogic",
                code="FL_BALANCE_BREAK",
                title="Running balance does not reconcile",
                severity=Severity.HIGH,
                weight=25.0,
                evidence="Balance jumps by 12,000 with no matching transaction.",
                page=1,
                field="row 5",
            ),
        ],
        summary="Two high-severity findings; routed to a human.",
    )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient whose webapp uses an isolated, tmp-path CaseStore seeded with
    one case. Patching the module-level ``store`` means every route the app
    serves reads/writes the temp store, never the real data/cases.json."""
    from fastapi.testclient import TestClient

    from fraud_detect import webapp

    store = CaseStore(str(tmp_path / "cases.json"),
                      audit_path=str(tmp_path / "audit_chain.jsonl"))
    store.upsert_report(_seed_report())
    monkeypatch.setattr(webapp, "store", store)

    # follow_redirects=False so we can assert the 303 redirects explicitly.
    with TestClient(webapp.app, follow_redirects=False) as c:
        # expose the isolated store to the tests for state assertions
        c.store = store  # type: ignore[attr-defined]
        yield c


def _decision(client) -> ReviewDecision:
    return client.store.get(CASE_ID).review.decision  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# ITEM 1 - the review-all-stages gate on /decide
# --------------------------------------------------------------------------- #
def test_decide_without_acknowledgement_is_rejected_and_stays_pending(client):
    """No stages_reviewed value -> 400, and NO decision is recorded (the gate
    runs BEFORE store.record_decision, so the case stays PENDING)."""
    assert _decision(client) == ReviewDecision.PENDING

    resp = client.post(f"/case/{CASE_ID}/decide", data={
        "decision": "approve",
        "reviewer": VALID_REVIEWER,
        "reason": "Looks fine to me.",
        # stages_reviewed deliberately omitted
    })
    assert resp.status_code == 400
    # the decision must NOT have been recorded
    assert _decision(client) == ReviewDecision.PENDING


def test_decide_with_acknowledgement_and_valid_reviewer_records_decision(client):
    """Acknowledgement present + roster reviewer -> 303 redirect and the
    decision is persisted."""
    resp = client.post(f"/case/{CASE_ID}/decide", data={
        "decision": "approve",
        "reviewer": VALID_REVIEWER,
        "reason": "Statement reconciles once re-examined.",
        "stages_reviewed": "1",
    })
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/case/{CASE_ID}"
    assert _decision(client) == ReviewDecision.APPROVED


# --------------------------------------------------------------------------- #
# ITEM 5 - reviewer-roster validation
# --------------------------------------------------------------------------- #
def test_decide_with_unknown_reviewer_is_rejected(client):
    resp = client.post(f"/case/{CASE_ID}/decide", data={
        "decision": "reject",
        "reviewer": UNKNOWN_REVIEWER,
        "reason": "Trying to sign off under a bogus name.",
        "stages_reviewed": "1",
    })
    assert resp.status_code == 400
    assert _decision(client) == ReviewDecision.PENDING


def test_request_info_with_unknown_reviewer_is_rejected(client):
    resp = client.post(f"/case/{CASE_ID}/request-info", data={
        "reviewer": UNKNOWN_REVIEWER,
        "message": "Please re-upload clearer pages.",
        "internal_note": "unknown officer attempt",
    })
    assert resp.status_code == 400
    # case untouched - still PENDING (not moved to NEEDS_INFO)
    assert _decision(client) == ReviewDecision.PENDING


def test_request_info_with_valid_reviewer_is_accepted(client):
    resp = client.post(f"/case/{CASE_ID}/request-info", data={
        "reviewer": VALID_REVIEWER,
        "message": "Could you re-upload pages 2 and 3?",
        "internal_note": "scan quality low on those pages",
    })
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/case/{CASE_ID}"
    assert _decision(client) == ReviewDecision.NEEDS_INFO


# --------------------------------------------------------------------------- #
# ITEM 4 - 30-minute revert window at the route layer
# --------------------------------------------------------------------------- #
def test_revert_within_window_reopens_to_pending(client):
    """Decide first, then revert immediately -> 303 and the case is PENDING
    again (re-opened for a human to decide)."""
    client.post(f"/case/{CASE_ID}/decide", data={
        "decision": "reject",
        "reviewer": VALID_REVIEWER,
        "reason": "Rejected by mistake.",
        "stages_reviewed": "1",
    })
    assert _decision(client) == ReviewDecision.REJECTED

    resp = client.post(f"/case/{CASE_ID}/revert", data={
        "reviewer": VALID_REVIEWER,
        "note": "Mis-click: this application should have been approved.",
    })
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/case/{CASE_ID}"
    assert _decision(client) == ReviewDecision.PENDING


def test_revert_after_window_is_rejected(client):
    """A decision older than REVERT_WINDOW_MINUTES can no longer be reverted:
    the store raises ValueError and the route maps it to HTTP 400, leaving the
    final decision intact."""
    client.post(f"/case/{CASE_ID}/decide", data={
        "decision": "reject",
        "reviewer": VALID_REVIEWER,
        "reason": "Rejected.",
        "stages_reviewed": "1",
    })
    assert _decision(client) == ReviewDecision.REJECTED

    # Backdate the recorded decision time to just outside the 30-minute window.
    store = client.store  # type: ignore[attr-defined]
    data = store._read()
    expired = datetime.utcnow() - timedelta(minutes=REVERT_WINDOW_MINUTES + 1)
    data[CASE_ID]["review"]["reviewed_at"] = expired.isoformat()
    store._write(data)

    resp = client.post(f"/case/{CASE_ID}/revert", data={
        "reviewer": VALID_REVIEWER,
        "note": "Too late to undo this one.",
    })
    assert resp.status_code == 400
    # the decision still stands - revert did NOT re-open the case
    assert _decision(client) == ReviewDecision.REJECTED


# --------------------------------------------------------------------------- #
# ITEM 6 - senior-officer escalation
# --------------------------------------------------------------------------- #
def test_escalate_with_blank_note_is_rejected(client):
    """A junior officer MUST justify the reroute; a blank note -> 400 and no
    senior officer is assigned."""
    resp = client.post(f"/case/{CASE_ID}/escalate", data={
        "reviewer": VALID_REVIEWER,
        "senior_officer": VALID_SENIOR,
        "note": "   ",   # whitespace only -> blank after _scrub
    })
    assert resp.status_code == 400
    assert client.store.get(CASE_ID).review.assigned_senior_officer is None  # type: ignore[attr-defined]


def test_escalate_valid_assigns_senior_officer(client):
    resp = client.post(f"/case/{CASE_ID}/escalate", data={
        "reviewer": VALID_REVIEWER,
        "senior_officer": VALID_SENIOR,
        "note": "Complex counterparty web beyond my experience; please take over.",
    })
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/case/{CASE_ID}"
    review = client.store.get(CASE_ID).review  # type: ignore[attr-defined]
    assert review.assigned_senior_officer == VALID_SENIOR
    assert review.escalated_by == VALID_REVIEWER
    # escalation does not itself decide the case
    assert review.decision == ReviewDecision.PENDING


def test_escalate_with_unknown_senior_officer_is_rejected(client):
    resp = client.post(f"/case/{CASE_ID}/escalate", data={
        "reviewer": VALID_REVIEWER,
        "senior_officer": "Ghost Senior - Nonexistent",
        "note": "Please take over.",
    })
    assert resp.status_code == 400
    assert client.store.get(CASE_ID).review.assigned_senior_officer is None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# ITEM 7 - the removed self-service screening routes
# --------------------------------------------------------------------------- #
def test_upload_route_is_gone(client):
    """POST /upload was removed as an internal control; the route no longer
    exists, so FastAPI answers 404 (unknown path) or 405 (method)."""
    resp = client.post("/upload", data={})
    assert resp.status_code in (404, 405)


def test_screen_sample_route_is_gone(client):
    resp = client.post("/screen-sample", data={})
    assert resp.status_code in (404, 405)


# --------------------------------------------------------------------------- #
# Regression: _group_by_stage still partitions findings by pipeline stage.
# A light assertion so a future refactor of the stage grouping is caught here.
# --------------------------------------------------------------------------- #
def test_group_by_stage_partitions_mixed_findings():
    from fraud_detect.webapp import _group_by_stage

    report = _seed_report()
    stages = _group_by_stage(report.findings)
    # Always returns the five known stages in pipeline order (incl. empty ones).
    titles = [title for title, _blurb, _matched in stages]
    assert titles[0].startswith("1 ")
    assert any(t.startswith("2 ") for t in titles)
    assert any(t.startswith("3 ") for t in titles)

    by_title = {title: matched for title, _blurb, matched in stages}
    vf = next(m for t, m in by_title.items() if t.startswith("2 "))
    fl = next(m for t, m in by_title.items() if t.startswith("3 "))
    assert {f.code for f in vf} == {"VF_FONT_MISMATCH"}
    assert {f.code for f in fl} == {"FL_BALANCE_BREAK"}
    # every seeded finding lands in exactly one stage (no loss, no duplication)
    total = sum(len(m) for _t, _b, m in stages)
    assert total == len(report.findings)
