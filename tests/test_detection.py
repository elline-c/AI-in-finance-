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


# --------------------------------------------------------------------------- #
# FEAT-003 — multi-month segmentation, month coverage, and ORDER-INDEPENDENCE
#
# These exercise the FEAT-002 logic end-to-end: ingestion segmentation of a
# combined multi-month PDF, and the month-coverage / ordering checks in
# financial_logic (FL_MONTH_MISSING / FL_MONTH_DUPLICATE / FL_MONTH_OUT_OF_ORDER).
# The order-independence test is the key proof that a jumbled submission order
# cannot change which month a statement is identified as.
# --------------------------------------------------------------------------- #
from fraud_detect.models import StatementSegment  # noqa: E402
from fraud_detect.nodes import ingestion  # noqa: E402
from fraud_detect.synthetic import (  # noqa: E402
    generate_combined_multi_month,
    generate_monthly_set,
)


def _ingestion_from_segments(segments):
    """Build a minimal IngestionResult carrying only the per-month segments,
    for driving the month-coverage checks directly (mirrors PR#7's hand-built
    IngestionResult style)."""
    return IngestionResult(
        source_path="combined.pdf",
        page_count=max((s.page_end or 0) for s in segments) if segments else 1,
        raw_text="",
        metadata=DocumentMetadata(),
        segments=list(segments),
    )


def _combined_ingestion_from_files(results):
    """Aggregate several single-file ingestion results into one IngestionResult
    whose ``segments`` are one-per-file, so the multi-file coverage checks run
    over the whole submitted set. Month identity for each segment comes solely
    from that file's parsed period dates (NOT its position in ``results``)."""
    segments = []
    for r in results:
        # each standalone monthly file parses to exactly one segment
        segments.append(r.segments[0] if r.segments else StatementSegment(
            period_start=r.statement_period_start,
            period_end=r.statement_period_end,
        ))
    return _ingestion_from_segments(segments)


# ---- (1) SEGMENTATION ------------------------------------------------------ #
def test_combined_pdf_segments_match_ground_truth(workdir):
    combined = generate_combined_multi_month(
        workdir, year=2024, months=[1, 2, 3, 4, 5, 6], jumbled=False)
    ing = ingestion.ingest(combined.path)

    assert len(ing.segments) == combined.month_count == 6
    # each physical section is one page, parsed in document order
    for seg, truth in zip(ing.segments, combined.segments):
        assert seg.page_start == truth.page_start
        assert seg.page_end == truth.page_end
        assert seg.period_start == truth.period_start
        assert seg.period_end == truth.period_end
    # chronological layout -> no ordering anomaly
    assert "FL_MONTH_OUT_OF_ORDER" not in _fl_codes(ing)


# ---- (2) MISSING MONTH ----------------------------------------------------- #
def test_missing_month_flags_fl_month_missing(workdir):
    # Jan, Feb, Apr -> March is omitted from the interior of the range.
    mset = generate_monthly_set(
        workdir, year=2024, months=[1, 2, 4], kind="missing")
    results = ingestion.ingest_many(mset.paths)
    ing = _combined_ingestion_from_files(results)

    findings = financial_logic.analyze(ing).findings
    missing = [f for f in findings if f.code == "FL_MONTH_MISSING"]
    assert missing, "expected FL_MONTH_MISSING for the omitted month"
    # evidence must name the actual missing month (March 2024)
    assert any("March 2024" in f.evidence for f in missing)


# ---- (3) DUPLICATE MONTH --------------------------------------------------- #
def test_duplicate_month_flags_fl_month_duplicate(workdir):
    # February produced twice -> duplicate coverage of that calendar month.
    mset = generate_monthly_set(
        workdir, year=2024, months=[1, 2, 2, 3], kind="duplicate")
    results = ingestion.ingest_many(mset.paths)
    ing = _combined_ingestion_from_files(results)

    findings = financial_logic.analyze(ing).findings
    dupes = [f for f in findings if f.code == "FL_MONTH_DUPLICATE"]
    assert dupes, "expected FL_MONTH_DUPLICATE for the repeated month"
    assert any("February 2024" in f.evidence for f in dupes)


# ---- (4) OUT-OF-ORDER ------------------------------------------------------ #
def test_out_of_order_combined_pdf_flags_fl_month_out_of_order(workdir):
    combined = generate_combined_multi_month(
        workdir, year=2024, months=[1, 2, 3, 4, 5, 6], jumbled=True)
    ing = ingestion.ingest(combined.path)

    assert len(ing.segments) == combined.month_count == 6
    codes = _fl_codes(ing)
    assert "FL_MONTH_OUT_OF_ORDER" in codes
    ooo = [f for f in financial_logic.analyze(ing).findings
           if f.code == "FL_MONTH_OUT_OF_ORDER"]
    # evidence names months and the page where the inversion appears
    assert any("Page" in f.evidence and "2024" in f.evidence for f in ooo)


# ---- (5) ORDER-INDEPENDENCE (requirement D) -------------------------------- #
def test_month_identification_is_order_independent(workdir):
    """The SAME set of monthly files screened in two different submission
    orders must yield identical per-file month identification AND identical
    coverage finding codes. This proves a jumbled upload order cannot change
    which month a statement is identified as."""
    mset = generate_monthly_set(
        workdir, year=2024, months=[1, 2, 4], kind="orderproof")
    paths = mset.paths

    forward = list(paths)
    reversed_order = list(reversed(paths))
    assert forward != reversed_order  # genuinely different input orders

    res_fwd = ingestion.ingest_many(forward)
    res_rev = ingestion.ingest_many(reversed_order)

    # Per-FILE month identification: map each file path to the (year, month)
    # its parsed period dates declare. Must be identical regardless of order.
    def identity_map(paths_order, results):
        out = {}
        for p, r in zip(paths_order, results):
            start = r.statement_period_start
            out[p] = (start.year, start.month) if start else None
        return out

    id_fwd = identity_map(forward, res_fwd)
    id_rev = identity_map(reversed_order, res_rev)
    assert id_fwd == id_rev
    # and the identification must match what the generator declared per file
    expected = {f.path: (f.year, f.month) for f in mset.files}
    assert id_fwd == expected

    # Coverage finding codes must be identical across the two orderings.
    codes_fwd = _fl_codes(_combined_ingestion_from_files(res_fwd))
    codes_rev = _fl_codes(_combined_ingestion_from_files(res_rev))
    assert codes_fwd == codes_rev
    # the gap (March) is still detected irrespective of order
    assert "FL_MONTH_MISSING" in codes_fwd
