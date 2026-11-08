"""Tests for the customer portal's applicant-entered details (ApplicationDetails).

Covers the customer-portal improvements:
  * the applicant's TYPED form values are persisted on the Case and can be read
    back (so the confirmation / status pages show what the applicant entered,
    e.g. "ABC Trading", rather than the name parsed from the bank statement),
  * older cases stored WITHOUT an ``application`` field still load (backward
    compatible / Optional-with-default), and
  * the Singapore-time (SGT, UTC+8) timestamp helper formats a UTC datetime
    correctly and is labelled SGT.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fraud_detect.customer_app import _format_sgt
from fraud_detect.models import (
    ApplicationDetails,
    Case,
    ReviewRecord,
    RiskBand,
    ConfidenceBand,
    RiskReport,
    Routing,
)
from fraud_detect.store import CaseStore


def _make_report(case_id: str = "CASE-TEST01") -> RiskReport:
    """A minimal statement-parsed report whose account_holder differs from the
    name the applicant types, so we can prove the typed value wins."""
    return RiskReport(
        case_id=case_id,
        source_path="/tmp/stmt.pdf",
        account_holder="Meridian Trading Company",   # parsed FROM the statement
        uen="200000000Z",
        statement_period="01 Mar 2025 - 31 Mar 2025",
        risk_score=10,
        risk_band=RiskBand.LOW,
        confidence_score=90,
        confidence_band=ConfidenceBand.VERY_HIGH,
        routing=Routing.AUTO_CLEAR,
        routing_reason="low risk",
    )


def test_application_details_persist_typed_values(tmp_path):
    store = CaseStore(str(tmp_path / "cases.json"))
    store.upsert_report(_make_report())

    details = ApplicationDetails(
        company_name="ABC Trading",          # what the APPLICANT typed
        uen="201234567A",
        contact_email="ops@abc.example",
        loan_amount="SGD 500,000",
        loan_reason="Business expansion / new premises",
    )
    store.set_application_details("CASE-TEST01", details)

    case = store.get("CASE-TEST01")
    assert case is not None
    assert case.application is not None
    # Typed company name is preserved and is NOT the statement-parsed name.
    assert case.application.company_name == "ABC Trading"
    assert case.application.company_name != case.report.account_holder
    assert case.application.loan_reason == "Business expansion / new premises"
    assert case.application.loan_amount == "SGD 500,000"


def test_set_application_details_unknown_case_returns_none(tmp_path):
    store = CaseStore(str(tmp_path / "cases.json"))
    result = store.set_application_details("CASE-DOES-NOT-EXIST",
                                           ApplicationDetails(company_name="X"))
    assert result is None


def test_old_case_without_application_field_still_loads(tmp_path):
    """A Case persisted before ApplicationDetails existed must still validate."""
    store = CaseStore(str(tmp_path / "cases.json"))
    case = store.upsert_report(_make_report("CASE-OLD001"))
    # upsert_report never sets `application`, so this mirrors a legacy record.
    loaded = store.get("CASE-OLD001")
    assert loaded is not None
    assert loaded.application is None
    assert loaded.report.account_holder == "Meridian Trading Company"
    assert case.case_id == loaded.case_id


def test_application_details_defaults_are_optional():
    details = ApplicationDetails()
    assert details.company_name is None
    assert details.uen is None
    assert details.contact_email is None
    assert details.loan_amount is None
    assert details.loan_reason is None


def test_format_sgt_adds_eight_hours_and_labels_sgt():
    # 2025-03-01 09:30 UTC -> 2025-03-01 17:30 SGT
    utc = datetime(2025, 3, 1, 9, 30, tzinfo=timezone.utc)
    assert _format_sgt(utc) == "01 Mar 2025, 17:30 SGT"


def test_format_sgt_treats_naive_datetime_as_utc():
    naive = datetime(2025, 3, 1, 20, 0)  # stored naive-UTC, as created_at is
    # 20:00 UTC + 8h = 04:00 next day SGT
    assert _format_sgt(naive) == "02 Mar 2025, 04:00 SGT"
