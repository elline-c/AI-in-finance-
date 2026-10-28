"""Phase 5 test cases.

Proves the detection agents catch the deliberately-planted issues (Phase 3),
that a clean statement is auto-cleared, and that borderline / unreadable inputs
are routed to a human. Each planted-issue kind maps to the reason code(s) that
should fire.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date

import pytest

from fraud_detect.models import (
    DocumentMetadata,
    IngestionResult,
    RiskBand,
    Routing,
)
from fraud_detect.nodes import financial_logic
from fraud_detect.pipeline import run_pipeline
from fraud_detect.synthetic import (
    generate_clean,
    generate_medium,
    generate_tampered,
    generate_unreadable,
)

# planted-issue kind -> reason code(s) that should be raised when it is present
PLANTED_TO_CODES = {
    "ALTERED_BALANCE": {"FL_BALANCE_BREAK"},
    "ALTERED_NUMBER": {"FL_BALANCE_BREAK", "VF_FONT_MISMATCH"},
    "FORMAT_INCONSISTENCY": {"VF_DATE_FORMAT_MIX", "VF_CURRENCY_MIX"},
    "METADATA_MISMATCH": {"VF_EDIT_SOFTWARE", "VF_CREATION_BEFORE_PERIOD"},
    "PERIOD_GAP": {"FL_PERIOD_GAP"},
    "SUSPICIOUS_TXN": {"FL_SUDDEN_DEPOSIT", "FL_UNVERIFIED_COUNTERPARTY"},
    "STRUCTURING": {"FL_STRUCTURING"},
    "UEN_MISMATCH": {"FL_UEN_NAME_MISMATCH"},
}


@pytest.fixture(scope="module")
def workdir():
    d = tempfile.mkdtemp(prefix="fraud_samples_")
    yield d


def _codes(report):
    return {f.code for f in report.findings}


def test_clean_is_auto_cleared(workdir):
    gen = generate_clean(workdir)
    r = run_pipeline(gen.path)
    assert r.risk_band == RiskBand.LOW
    assert r.routing == Routing.AUTO_CLEAR
    assert len(r.findings) == 0


def test_fully_tampered_is_very_high_and_escalated(workdir):
    gen = generate_tampered(workdir)
    r = run_pipeline(gen.path)
    assert r.risk_band == RiskBand.VERY_HIGH
    assert r.routing == Routing.HUMAN_REVIEW
    assert r.risk_score >= 76


@pytest.mark.parametrize("issue", list(PLANTED_TO_CODES.keys()))
def test_each_planted_issue_is_detected(workdir, issue):
    """Generate a statement with ONLY this issue and assert its code(s) fire."""
    gen = generate_tampered(workdir, seed=42, issues=[issue])
    r = run_pipeline(gen.path)
    codes = _codes(r)
    expected = PLANTED_TO_CODES[issue]
    assert codes & expected, (
        f"planted {issue}: expected one of {expected}, got {sorted(codes)}"
    )


def test_medium_risk_routes_to_human(workdir):
    gen = generate_medium(workdir)
    r = run_pipeline(gen.path)
    # mild formatting only -> not LOW, escalated to a human, but not VERY_HIGH
    assert r.risk_band in (RiskBand.MEDIUM, RiskBand.HIGH)
    assert r.routing == Routing.HUMAN_REVIEW


def test_unreadable_scan_flags_data_quality_and_escalates(workdir):
    gen = generate_unreadable(workdir)
    r = run_pipeline(gen.path)
    codes = _codes(r)
    assert any(code.startswith("DQ_") for code in codes)
    assert r.routing == Routing.HUMAN_REVIEW


def test_machine_never_auto_rejects(workdir):
    """The most severe automated outcome must be HUMAN_REVIEW, never a rejection."""
    for gen in (generate_clean(workdir), generate_tampered(workdir),
                generate_medium(workdir), generate_unreadable(workdir)):
        r = run_pipeline(gen.path)
        assert r.routing in (Routing.AUTO_CLEAR, Routing.HUMAN_REVIEW)


# --------------------------------------------------------------------------- #
# Node 3 — statement-period length check (FL_PERIOD_LENGTH)
# --------------------------------------------------------------------------- #
def _ingestion_with_period(start, end):
    """Build a minimal IngestionResult carrying only a declared statement
    period, for exercising the financial-logic period-length rule directly."""
    return IngestionResult(
        source_path="test.pdf",
        page_count=1,
        raw_text="",
        metadata=DocumentMetadata(),
        statement_period_start=start,
        statement_period_end=end,
    )


def _fl_codes(ing):
    return {f.code for f in financial_logic.analyze(ing).findings}


def test_period_too_short_flags_period_length():
    # ~3 months -> below the 6-month floor
    ing = _ingestion_with_period(date(2023, 1, 1), date(2023, 4, 1))
    assert "FL_PERIOD_LENGTH" in _fl_codes(ing)


def test_period_too_long_flags_period_length():
    # ~14 months -> above the 12-month ceiling
    ing = _ingestion_with_period(date(2023, 1, 1), date(2024, 3, 1))
    assert "FL_PERIOD_LENGTH" in _fl_codes(ing)


def test_valid_period_does_not_flag_period_length():
    # 8 months -> inside the 6-12 month window
    ing = _ingestion_with_period(date(2023, 1, 1), date(2023, 9, 1))
    assert "FL_PERIOD_LENGTH" not in _fl_codes(ing)


def test_boundary_periods_do_not_flag_period_length():
    # exactly 6 months and exactly 12 months are both compliant
    six = _ingestion_with_period(date(2023, 1, 1), date(2023, 7, 1))
    twelve = _ingestion_with_period(date(2023, 1, 1), date(2024, 1, 1))
    assert "FL_PERIOD_LENGTH" not in _fl_codes(six)
    assert "FL_PERIOD_LENGTH" not in _fl_codes(twelve)


@pytest.mark.parametrize("start,end", [
    (None, date(2023, 9, 1)),
    (date(2023, 1, 1), None),
    (None, None),
])
def test_missing_period_dates_do_not_crash_or_flag(start, end):
    ing = _ingestion_with_period(start, end)
    # must not raise and must not emit the period-length code
    assert "FL_PERIOD_LENGTH" not in _fl_codes(ing)
