"""Tests for ``display_company_name`` (analyst & banker display preference).

The analyst and loan-officer consoles must show the company name the APPLICANT
TYPED into the customer portal (``case.application.company_name``) in preference
to the name PARSED from the uploaded statement (``case.report.account_holder``).
Older cases have no ``application`` and must still fall back to the parsed name
without raising. These are pure-function tests (no I/O, no network).
"""
from __future__ import annotations

from fraud_detect.models import (
    ApplicationDetails,
    Case,
    ConfidenceBand,
    ReviewRecord,
    RiskBand,
    RiskReport,
    Routing,
    display_company_name,
)


def _report(account_holder):
    """A minimal valid RiskReport carrying the statement-parsed account holder."""
    return RiskReport(
        case_id="CASE-TEST01",
        source_path="/tmp/stmt.pdf",
        account_holder=account_holder,
        risk_score=10,
        risk_band=RiskBand.LOW,
        confidence_score=90,
        confidence_band=ConfidenceBand.VERY_HIGH,
        routing=Routing.AUTO_CLEAR,
        routing_reason="clean",
    )


def _case(account_holder, application):
    return Case(
        case_id="CASE-TEST01",
        report=_report(account_holder),
        review=ReviewRecord(case_id="CASE-TEST01"),
        application=application,
    )


def test_prefers_applicant_typed_name_over_parsed():
    """Typed "ABC Trading" wins over the statement-parsed "Meridian"."""
    case = _case("Meridian", ApplicationDetails(company_name="ABC Trading"))
    assert display_company_name(case) == "ABC Trading"


def test_falls_back_to_parsed_when_application_is_none():
    """Older case with no application -> show the statement-parsed holder."""
    case = _case("Meridian", None)
    assert display_company_name(case) == "Meridian"


def test_falls_back_to_parsed_when_company_name_blank():
    """An application with no company_name still falls back to the parsed name."""
    case = _case("Meridian", ApplicationDetails(company_name=None, uen="X"))
    assert display_company_name(case) == "Meridian"


def test_empty_company_name_string_falls_back():
    """An empty-string typed name is treated as absent (falsy) -> fall back."""
    case = _case("Meridian", ApplicationDetails(company_name=""))
    assert display_company_name(case) == "Meridian"


def test_returns_none_when_neither_available():
    """Neither a typed name nor a parsed holder -> None (caller renders a dash)."""
    case = _case(None, None)
    assert display_company_name(case) is None
