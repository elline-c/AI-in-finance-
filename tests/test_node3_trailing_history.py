"""Trailing-history trust rules for Node 3 gap learning (store.trailing_history)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from fraud_detect.models import (
    ConfidenceBand, DocumentMetadata, IngestionResult, ReviewDecision,
    RiskBand, RiskReport, Routing, Transaction,
)
from fraud_detect.nodes import financial_logic
from fraud_detect.store import CaseStore

UEN = "201812345A"
HOLDER = "Meridian Trading Pte Ltd"
# Quiet every ~10 days: a long normal rhythm for this company.
SPARSE_DAYS = [0, 10, 21, 30, 41, 51, 61, 72, 81, 92, 102, 112]


def _txns(days, start=date(2025, 1, 1)):
    return [Transaction(index=i + 1, date=start + timedelta(days=d), description="X",
                        amount=1.0, page=1) for i, d in enumerate(days)]


def _report(case_id, *, uen=UEN, holder=HOLDER, days=SPARSE_DAYS,
            routing=Routing.HUMAN_REVIEW, created=None):
    return RiskReport(
        case_id=case_id, source_path=f"{case_id}.pdf", account_holder=holder, uen=uen,
        risk_score=40, risk_band=RiskBand.MEDIUM, confidence_score=60,
        confidence_band=ConfidenceBand.HIGH, routing=routing, routing_reason="t",
        transactions=_txns(days),
        created_at=created or datetime(2025, 6, 1),
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    monkeypatch.delenv("TRAILING_HISTORY_REQUIRE_CHAIN", raising=False)
    return CaseStore(str(tmp_path / "cases.json"), str(tmp_path / "chain.jsonl"))


def _approve(store, report, decision=ReviewDecision.APPROVED):
    store.upsert_report(report)
    store.record_decision(report.case_id, decision, "alice", "checked")


def _hist(store, **kw):
    return store.trailing_history(UEN, HOLDER, **kw)


# --------------------------------------------------------------------------- #
# Who counts
# --------------------------------------------------------------------------- #
def test_human_approved_case_is_used(store):
    _approve(store, _report("CASE-A"))
    h = _hist(store)
    assert h.case_ids == ("CASE-A",)
    assert len(h.gap_days) == len(SPARSE_DAYS) - 1


@pytest.mark.parametrize("decision", [ReviewDecision.REJECTED])
def test_rejected_case_is_not_used(store, decision):
    _approve(store, _report("CASE-R"), decision)
    assert _hist(store).case_ids == ()


def test_auto_cleared_case_is_not_used(store):
    store.upsert_report(_report("CASE-AUTO", routing=Routing.AUTO_CLEAR))   # PENDING
    assert _hist(store).case_ids == ()


def test_pending_and_needs_info_cases_are_not_used(store):
    store.upsert_report(_report("CASE-P"))
    store.upsert_report(_report("CASE-N"))
    store.request_info("CASE-N", "alice", "please send more")
    assert _hist(store).case_ids == ()


def test_different_uen_is_not_used(store):
    _approve(store, _report("CASE-U", uen="200923456B"))
    assert _hist(store).case_ids == ()


def test_different_holder_is_not_used(store):
    _approve(store, _report("CASE-H", holder="Skyline Logistics Pte Ltd"))
    assert _hist(store).case_ids == ()


def test_holder_match_ignores_case_and_punctuation(store):
    _approve(store, _report("CASE-M", holder="MERIDIAN TRADING PTE. LTD."))
    assert _hist(store).case_ids == ("CASE-M",)


@pytest.mark.parametrize("uen,holder", [(None, HOLDER), (UEN, None), ("", HOLDER)])
def test_missing_uen_or_holder_gives_no_history(store, uen, holder):
    _approve(store, _report("CASE-A"))
    assert store.trailing_history(uen, holder).case_ids == ()


def test_the_current_case_is_excluded(store):
    _approve(store, _report("CASE-A"))
    assert _hist(store, exclude_case_id="CASE-A").case_ids == ()


def test_only_the_most_recent_statements_are_used(store):
    for i in range(8):
        _approve(store, _report(f"CASE-{i}", created=datetime(2025, 1, 1 + i)))
    h = _hist(store, max_statements=3)
    assert h.case_ids == ("CASE-7", "CASE-6", "CASE-5")


def test_gaps_are_not_joined_across_statements(store):
    # Two statements a year apart: the year-long space between them must not appear.
    _approve(store, _report("CASE-1", days=[0, 5, 10], created=datetime(2025, 1, 1)))
    _approve(store, _report("CASE-2", days=[0, 5, 10], created=datetime(2025, 2, 1)))
    assert sorted(_hist(store).gap_days) == [5, 5, 5, 5]


# --------------------------------------------------------------------------- #
# Chain cross-check
# --------------------------------------------------------------------------- #
def test_edited_decision_without_a_chain_event_is_not_trusted(store):
    store.upsert_report(_report("CASE-F"))               # only SCREENED in the chain
    raw = json.loads(store.path.read_text())
    raw["CASE-F"]["review"]["decision"] = "APPROVED"     # forged by editing the file
    store.path.write_text(json.dumps(raw))
    assert _hist(store).case_ids == ()
    assert _hist(store, require_chain=False).case_ids == ("CASE-F",)


def test_approval_later_reversed_in_the_chain_is_not_trusted(store):
    _approve(store, _report("CASE-X"))
    store.record_decision("CASE-X", ReviewDecision.REJECTED, "bob", "changed mind")
    raw = json.loads(store.path.read_text())
    raw["CASE-X"]["review"]["decision"] = "APPROVED"     # file says approved, chain says rejected
    store.path.write_text(json.dumps(raw))
    assert _hist(store).case_ids == ()


def test_broken_chain_gives_no_history(store):
    _approve(store, _report("CASE-A"))
    lines = store._audit.path.read_text().splitlines()
    lines[0] = lines[0].replace("SCREENED", "SCREENEX")  # tamper with an early entry
    store._audit.path.write_text("\n".join(lines) + "\n")
    store._chain_check = None                            # drop the cached verdict
    assert _hist(store).case_ids == ()
    assert _hist(store, require_chain=False).case_ids == ("CASE-A",)


def test_chain_check_can_be_switched_off_by_setting(store, monkeypatch):
    store.upsert_report(_report("CASE-F"))
    raw = json.loads(store.path.read_text())
    raw["CASE-F"]["review"]["decision"] = "APPROVED"
    store.path.write_text(json.dumps(raw))
    monkeypatch.setenv("TRAILING_HISTORY_REQUIRE_CHAIN", "0")
    assert _hist(store).case_ids == ("CASE-F",)


# --------------------------------------------------------------------------- #
# Effect on Node 3, and the feedback loop
# --------------------------------------------------------------------------- #
def _statement(days):
    return IngestionResult(source_path="new.pdf", page_count=1, raw_text="",
                           metadata=DocumentMetadata(), uen=UEN, account_holder=HOLDER,
                           transactions=_txns(days))


def test_approval_teaches_node3_and_the_next_statement_is_not_flagged(store):
    # Dense-looking statement with a 12-day silence: flagged when judged alone.
    new_days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23]
    alone = financial_logic.analyze(_statement(new_days))
    assert "FL_PERIOD_GAP" in [f.code for f in alone.findings]

    # An analyst approves an earlier statement showing a ~10-day rhythm.
    _approve(store, _report("CASE-OLD"))
    h = _hist(store)
    taught = financial_logic.analyze(
        _statement(new_days), trailing_gap_days=list(h.gap_days),
        trailing_source=h.describe())
    assert "FL_PERIOD_GAP" not in [f.code for f in taught.findings]


def test_evidence_names_the_cases_the_limit_came_from(store):
    _approve(store, _report("CASE-OLD"))
    h = _hist(store)
    days = list(range(0, 15)) + list(range(80, 90))      # a 66-day silence
    rep = financial_logic.analyze(
        _statement(days), trailing_gap_days=list(h.gap_days), trailing_source=h.describe())
    gap = [f for f in rep.findings if f.code == "FL_PERIOD_GAP"][0]
    assert "CASE-OLD" in gap.evidence
    assert "approved earlier statement" in gap.evidence


def test_pipeline_survives_a_failing_history_lookup(store, monkeypatch):
    """A broken history lookup must not stop screening."""
    from fraud_detect import pipeline

    def boom(*a, **k):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(store, "trailing_history", boom)
    monkeypatch.setattr(pipeline.ingestion, "ingest", lambda p: _statement([0, 5, 9]))
    report = pipeline.run_pipeline("new.pdf", store=store)
    assert report.case_id.startswith("CASE-")
