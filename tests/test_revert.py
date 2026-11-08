"""Tests for the 30-minute revert window (ITEM 4).

A reviewer can undo an APPROVED / REJECTED decision, but only within
REVERT_WINDOW_MINUTES of the decision. The boundary is exercised deterministically
via the injectable ``now`` argument. The revert must be an append-only audit
event: the prior HUMAN_APPROVED / HUMAN_REJECTED entry stays in both logs and a
new HUMAN_REVERTED entry is appended, and the case is re-opened to PENDING.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fraud_detect.models import (
    ConfidenceBand,
    ReviewDecision,
    RiskBand,
    RiskReport,
    Routing,
)
from fraud_detect.store import REVERT_WINDOW_MINUTES, CaseStore, revert_window_remaining


@pytest.fixture()
def store(tmp_path):
    return CaseStore(
        str(tmp_path / "cases.json"),
        audit_path=str(tmp_path / "audit_chain.jsonl"),
    )


def _seed_report(case_id: str = "CASE-REVERT") -> RiskReport:
    """A minimal, hand-built RiskReport (no pipeline needed)."""
    return RiskReport(
        case_id=case_id,
        source_path="/tmp/example.pdf",
        risk_score=40,
        risk_band=RiskBand.MEDIUM,
        confidence_score=70,
        confidence_band=ConfidenceBand.HIGH,
        routing=Routing.HUMAN_REVIEW,
        routing_reason="medium risk",
    )


def _decided_case(store: CaseStore, decision: ReviewDecision,
                  case_id: str = "CASE-REVERT"):
    store.upsert_report(_seed_report(case_id))
    return store.record_decision(
        case_id, decision, reviewer="Jane Tan - Credit Risk Analyst",
        reason="initial decision",
    )


def test_revert_just_under_window_is_allowed(store):
    case = _decided_case(store, ReviewDecision.REJECTED)
    reviewed_at = case.review.reviewed_at
    # 29m59s later -> still inside the 30-minute window.
    now = reviewed_at + timedelta(minutes=REVERT_WINDOW_MINUTES) - timedelta(seconds=1)
    reverted = store.revert_decision(
        case.case_id, reviewer="Jane Tan - Credit Risk Analyst",
        note="mis-clicked reject", now=now,
    )
    assert reverted is not None
    assert reverted.review.decision == ReviewDecision.PENDING


def test_revert_exactly_at_window_boundary_raises(store):
    # At exactly 30 minutes the window has elapsed (guard uses >=), so a revert
    # at the boundary is already too late.
    case = _decided_case(store, ReviewDecision.APPROVED)
    now = case.review.reviewed_at + timedelta(minutes=REVERT_WINDOW_MINUTES)
    with pytest.raises(ValueError):
        store.revert_decision(
            case.case_id, reviewer="Jane Tan - Credit Risk Analyst",
            note="boundary", now=now,
        )


def test_revert_just_over_window_raises(store):
    case = _decided_case(store, ReviewDecision.APPROVED)
    now = case.review.reviewed_at + timedelta(minutes=REVERT_WINDOW_MINUTES, seconds=1)
    with pytest.raises(ValueError):
        store.revert_decision(
            case.case_id, reviewer="Jane Tan - Credit Risk Analyst",
            note="too late", now=now,
        )


def test_revert_of_pending_case_raises(store):
    store.upsert_report(_seed_report("CASE-PENDING"))
    with pytest.raises(ValueError):
        store.revert_decision(
            "CASE-PENDING", reviewer="Jane Tan - Credit Risk Analyst",
            note="nothing to revert", now=datetime.utcnow(),
        )


def test_revert_of_needs_info_case_raises(store):
    store.upsert_report(_seed_report("CASE-INFO"))
    store.request_info("CASE-INFO", reviewer="Jane Tan - Credit Risk Analyst",
                       message_to_customer="please send payslips")
    with pytest.raises(ValueError):
        store.revert_decision(
            "CASE-INFO", reviewer="Jane Tan - Credit Risk Analyst",
            note="not final", now=datetime.utcnow(),
        )


def test_revert_preserves_prior_audit_and_appends_new_event(store):
    case = _decided_case(store, ReviewDecision.REJECTED)
    now = case.review.reviewed_at + timedelta(minutes=5)
    reverted = store.revert_decision(
        case.case_id, reviewer="Jane Tan - Credit Risk Analyst",
        note="mistake", now=now,
    )

    # Embedded mirror: the prior HUMAN_REJECTED is still present, with a new
    # HUMAN_REVERTED appended after it (history is append-only, not rewritten).
    events = [e["event"] for e in reverted.audit_log]
    assert "HUMAN_REJECTED" in events
    assert events[-1] == "HUMAN_REVERTED"

    # Hash-chain: both events present for this case, in order.
    chain_events = [e.event for e in store._audit.events_for(case.case_id)]
    assert "HUMAN_REJECTED" in chain_events
    assert chain_events[-1] == "HUMAN_REVERTED"


def test_revert_preserves_training_feedback(store):
    store.upsert_report(_seed_report("CASE-FB"))
    store.record_decision(
        "CASE-FB", ReviewDecision.REJECTED,
        reviewer="Jane Tan - Credit Risk Analyst", reason="r",
        feedback_for_training="useful signal",
    )
    case = store.get("CASE-FB")
    now = case.review.reviewed_at + timedelta(minutes=1)
    reverted = store.revert_decision(
        "CASE-FB", reviewer="Jane Tan - Credit Risk Analyst",
        note="redo", now=now,
    )
    assert reverted.review.feedback_for_training == "useful signal"


def test_revert_window_remaining_helper():
    report = _seed_report("CASE-WIN")
    from fraud_detect.models import Case, ReviewRecord

    reviewed_at = datetime.utcnow()
    case = Case(
        case_id="CASE-WIN",
        report=report,
        review=ReviewRecord(
            case_id="CASE-WIN",
            decision=ReviewDecision.APPROVED,
            reviewer="Jane Tan - Credit Risk Analyst",
            reviewed_at=reviewed_at,
        ),
    )
    # Inside the window: allowed, with seconds remaining > 0.
    allowed, remaining = revert_window_remaining(
        case, now=reviewed_at + timedelta(minutes=10))
    assert allowed is True
    assert 0 < remaining <= REVERT_WINDOW_MINUTES * 60

    # Past the window: not allowed, no remaining time.
    allowed, remaining = revert_window_remaining(
        case, now=reviewed_at + timedelta(minutes=REVERT_WINDOW_MINUTES, seconds=1))
    assert allowed is False
    assert remaining == 0

    # A PENDING case can never be reverted.
    case.review.decision = ReviewDecision.PENDING
    allowed, remaining = revert_window_remaining(case, now=reviewed_at)
    assert allowed is False
    assert remaining == 0
