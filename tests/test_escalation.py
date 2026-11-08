"""Tests for senior-officer escalation (ITEM 6).

A junior reviewer routes a complex case to a named senior officer with a
mandatory internal note. The assignment is recorded on the ReviewRecord and an
append-only ESCALATED_TO_SENIOR audit event is written to BOTH the embedded
mirror and the hash-chain. Escalation never itself decides the case.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from fraud_detect.models import (
    ConfidenceBand,
    ReviewDecision,
    RiskBand,
    RiskReport,
    Routing,
)
from fraud_detect.reviewers import is_valid_senior, senior_choices
from fraud_detect.store import CaseStore


@pytest.fixture()
def store(tmp_path):
    return CaseStore(
        str(tmp_path / "cases.json"),
        audit_path=str(tmp_path / "audit_chain.jsonl"),
    )


def _seed_report(case_id: str = "CASE-ESC") -> RiskReport:
    return RiskReport(
        case_id=case_id,
        source_path="/tmp/example.pdf",
        risk_score=60,
        risk_band=RiskBand.HIGH,
        confidence_score=50,
        confidence_band=ConfidenceBand.MEDIUM,
        routing=Routing.HUMAN_REVIEW,
        routing_reason="high risk, complex",
    )


def test_escalation_records_assignment_and_note(store):
    store.upsert_report(_seed_report())
    senior = senior_choices()[0]
    case = store.escalate_to_senior(
        "CASE-ESC",
        reviewer="Marcus Lee - Fraud Review Officer",
        senior_officer=senior,
        note="outside my experience; cross-border structuring",
        now=datetime.utcnow(),
    )
    assert case is not None
    assert case.review.assigned_senior_officer == senior
    assert case.review.escalated_by == "Marcus Lee - Fraud Review Officer"
    assert case.review.escalation_note == "outside my experience; cross-border structuring"
    assert case.review.escalated_at is not None


def test_escalation_keeps_pending_decision_open(store):
    store.upsert_report(_seed_report())
    case = store.escalate_to_senior(
        "CASE-ESC",
        reviewer="Marcus Lee - Fraud Review Officer",
        senior_officer=senior_choices()[0],
        note="please review",
    )
    # The machine never decides; escalation re-routes but leaves it PENDING.
    assert case.review.decision == ReviewDecision.PENDING


def test_escalation_appends_event_to_both_logs(store):
    store.upsert_report(_seed_report())
    case = store.escalate_to_senior(
        "CASE-ESC",
        reviewer="Marcus Lee - Fraud Review Officer",
        senior_officer=senior_choices()[0],
        note="needs senior eyes",
    )
    # Embedded mirror.
    assert case.audit_log[-1]["event"] == "ESCALATED_TO_SENIOR"
    assert "to=" in case.audit_log[-1]["detail"]
    # Hash-chain.
    chain_events = [e.event for e in store._audit.events_for("CASE-ESC")]
    assert chain_events[-1] == "ESCALATED_TO_SENIOR"


def test_escalation_preserves_prior_decision_fields(store):
    store.upsert_report(_seed_report())
    store.record_decision(
        "CASE-ESC", ReviewDecision.APPROVED,
        reviewer="Marcus Lee - Fraud Review Officer", reason="looked fine",
    )
    case = store.escalate_to_senior(
        "CASE-ESC",
        reviewer="Marcus Lee - Fraud Review Officer",
        senior_officer=senior_choices()[0],
        note="second opinion please",
    )
    # A senior can re-decide; the existing final decision is left intact.
    assert case.review.decision == ReviewDecision.APPROVED
    assert case.review.assigned_senior_officer == senior_choices()[0]


def test_escalation_on_missing_case_returns_none(store):
    assert store.escalate_to_senior(
        "NO-SUCH-CASE",
        reviewer="Marcus Lee - Fraud Review Officer",
        senior_officer=senior_choices()[0],
        note="x",
    ) is None


def test_senior_officer_roster_validation():
    assert is_valid_senior(senior_choices()[0]) is True
    assert is_valid_senior("Someone - Not On Roster") is False
