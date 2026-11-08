"""Tests for the Human-decision checklist + per-item / per-stage confidence.

Matches tests/test_detection.py style: hand-built RiskReport objects for the
unit-level band rules, plus an end-to-end run over the synthetic generators
(generate_clean / generate_unreadable) through run_pipeline.

Confidence is BANDS only (High / Medium / Low / n/a) with a non-empty rationale
(explainability); no numeric probabilities are asserted because none are
fabricated.
"""
from __future__ import annotations

import tempfile
from datetime import date

import pytest

from fraud_detect.models import (
    ConfidenceBand,
    Finding,
    RiskBand,
    RiskReport,
    Routing,
    Severity,
    Transaction,
)
from fraud_detect.nodes.checklist import (
    BAND_HIGH,
    BAND_LOW,
    BAND_NA,
    CHECKLIST_SIDE_NOTE,
    build_checklist,
    stage_confidence,
    stage_confidence_map,
)
from fraud_detect.pipeline import run_pipeline
from fraud_detect.synthetic import generate_clean, generate_unreadable


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _finding(code, agent="Node1:Ingestion", severity=Severity.MEDIUM, field=None):
    return Finding(
        agent=agent, code=code, title=code, severity=severity, weight=8,
        evidence=f"synthetic {code}", field=field,
    )


def _report(*, findings=None, account_holder="Acme Pte Ltd", uen="201812345A",
            account_number="123-456-789", opening_balance=1000.0,
            closing_balance=2000.0, declared_currency="SGD",
            transactions=None, period=True):
    """Build a minimal-but-valid RiskReport with sensible clean defaults."""
    if transactions is None:
        transactions = [Transaction(index=0, amount=100.0, balance=1100.0)]
    return RiskReport(
        case_id="CASE-TEST",
        source_path="test.pdf",
        account_holder=account_holder,
        uen=uen,
        statement_period="2024-01-01 to 2024-08-31",
        account_number=account_number,
        statement_period_start=date(2024, 1, 1) if period else None,
        statement_period_end=date(2024, 8, 31) if period else None,
        risk_score=1,
        risk_band=RiskBand.LOW,
        confidence_score=85,
        confidence_band=ConfidenceBand.VERY_HIGH,
        routing=Routing.AUTO_CLEAR,
        routing_reason="clean",
        findings=list(findings or []),
        transactions=list(transactions),
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        declared_currency=declared_currency,
    )


def _by_key(items):
    return {i.key: i for i in items}


# --------------------------------------------------------------------------- #
# Clean report -> all compulsory items ticked, High confidence, no Low flags
# --------------------------------------------------------------------------- #
def test_clean_report_all_compulsory_present_and_high():
    items = _by_key(build_checklist(_report()))
    compulsory = [i for i in items.values() if i.compulsory]
    assert compulsory, "expected some compulsory items"
    for item in compulsory:
        assert item.present is True, f"{item.key} should be ticked"
        assert item.confidence_band == BAND_HIGH, f"{item.key} should be High"
    # no item is Low on a clean parse
    assert all(i.confidence_band != BAND_LOW for i in items.values())
    # every rationale is non-empty (explainability)
    assert all(i.rationale for i in items.values())


def test_clean_report_covers_expected_keys():
    items = _by_key(build_checklist(_report()))
    for key in (
        "account_holder", "uen", "statement_period", "opening_balance",
        "transactions_present", "account_number", "closing_balance",
        "declared_currency", "period_covers_6_to_12_months",
        "running_balances_reconcile",
    ):
        assert key in items, f"missing checklist item {key}"


# --------------------------------------------------------------------------- #
# Missing holder -> that item crossed with band n/a
# --------------------------------------------------------------------------- #
def test_missing_holder_is_crossed_and_na():
    report = _report(account_holder=None, findings=[_finding("DQ_MISSING_HOLDER")])
    items = _by_key(build_checklist(report))
    holder = items["account_holder"]
    assert holder.present is False
    assert holder.confidence_band == BAND_NA
    assert holder.rationale  # non-empty
    # the still-present UEN item stays ticked (only holder was missing). It drops
    # to Low because an unrelated DQ_MISSING finding affects the same stage? No:
    # DQ_MISSING_HOLDER is not in uen's related_missing and is not a readability
    # code, so UEN is Medium (present with unrelated DQ findings).
    assert items["uen"].present is True


# --------------------------------------------------------------------------- #
# Unreadable report -> present items drop to Low
# --------------------------------------------------------------------------- #
def test_unreadable_report_present_items_drop_to_low():
    report = _report(findings=[_finding("DQ_UNREADABLE_ROW", severity=Severity.LOW)])
    items = _by_key(build_checklist(report))
    # account_holder is present (truthy, no DQ_MISSING_HOLDER) but the source was
    # unreadable -> Low band.
    assert items["account_holder"].present is True
    assert items["account_holder"].confidence_band == BAND_LOW
    assert "verify manually" in items["account_holder"].rationale


def test_absent_optional_item_is_na():
    report = _report(account_number=None)
    acct = _by_key(build_checklist(report))["account_number"]
    assert acct.present is False
    assert acct.confidence_band == BAND_NA


# --------------------------------------------------------------------------- #
# Per-stage confidence
# --------------------------------------------------------------------------- #
def test_stage_confidence_node1_clean_is_high():
    report = _report()
    band, rationale = stage_confidence(
        "1 · Ingestion & data quality", [], report)
    assert band == BAND_HIGH
    assert rationale


def test_stage_confidence_node1_readability_is_low():
    f = _finding("DQ_NO_TEXT", severity=Severity.HIGH)
    report = _report(findings=[f])
    band, rationale = stage_confidence(
        "1 · Ingestion & data quality", [f], report)
    assert band == BAND_LOW
    assert rationale


def test_stage_confidence_node1_missing_field_is_medium():
    from fraud_detect.nodes.checklist import BAND_MEDIUM
    f = _finding("DQ_MISSING_UEN")
    report = _report(uen=None, findings=[f])
    band, _ = stage_confidence("1 · Ingestion & data quality", [f], report)
    assert band == BAND_MEDIUM


def test_stage_confidence_downstream_low_when_source_unreadable():
    f_dq = _finding("DQ_UNREADABLE_ROW", severity=Severity.LOW)
    f_fl = _finding("FL_BALANCE_BREAK", agent="Node3:FinancialLogic")
    report = _report(findings=[f_dq, f_fl])
    band, rationale = stage_confidence(
        "3 · Financial logic", [f_fl], report)
    assert band == BAND_LOW
    assert rationale


# --------------------------------------------------------------------------- #
# End-to-end through the pipeline on the synthetic generators
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def workdir():
    return tempfile.mkdtemp(prefix="fraud_checklist_")


def test_pipeline_clean_statement_checklist_high(workdir):
    gen = generate_clean(workdir)
    report = run_pipeline(gen.path)
    items = _by_key(build_checklist(report))
    # a clean statement -> all compulsory items ticked and High, nothing Low
    for item in (i for i in items.values() if i.compulsory):
        assert item.present is True
        assert item.confidence_band == BAND_HIGH
    assert all(i.confidence_band != BAND_LOW for i in items.values())


def test_pipeline_clean_statement_stage_map_high(workdir):
    gen = generate_clean(workdir)
    report = run_pipeline(gen.path)
    # mirror the console's stage grouping just enough to key the map by title
    from fraud_detect.webapp import _group_by_stage
    stages = _group_by_stage(report.findings)
    cmap = stage_confidence_map(stages, report)
    assert cmap  # non-empty
    for title, (band, rationale) in cmap.items():
        assert band in (BAND_HIGH, "Medium", BAND_LOW)
        assert rationale
    # the ingestion stage on a clean statement is High
    ingest_title = next(t for t in cmap if t.startswith("1 "))
    assert cmap[ingest_title][0] == BAND_HIGH


def test_pipeline_unreadable_statement_stage_low(workdir):
    gen = generate_unreadable(workdir)
    report = run_pipeline(gen.path)
    from fraud_detect.webapp import _group_by_stage
    stages = _group_by_stage(report.findings)
    cmap = stage_confidence_map(stages, report)
    ingest_title = next(t for t in cmap if t.startswith("1 "))
    # unreadable input -> ingestion stage is Low
    assert cmap[ingest_title][0] == BAND_LOW


def test_side_note_recommends_requesting_all_info():
    assert "request ALL information" in CHECKLIST_SIDE_NOTE
    assert "em dash" not in CHECKLIST_SIDE_NOTE
    # plain hyphen only, no em dash character
    assert "\u2014" not in CHECKLIST_SIDE_NOTE
