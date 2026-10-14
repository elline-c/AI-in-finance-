"""Tests for Node 5 — Cross-Case Memory Agent.

Verifies that:
  * a first screening raises NO memory findings (nothing seen before),
  * after a human REJECTS a case, screening a statement with the same
    suspicious transactions raises MEM_RECURRING_KNOWN_BAD citing the prior case,
  * after a case is CLEARED, a repeat raises MEM_SIMILAR_PASSED_BEFORE,
  * the machine still never auto-rejects (routing is at most HUMAN_REVIEW),
  * fingerprint normalization collapses entity-suffix / near-amount variants.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from fraud_detect.models import ReviewDecision, Routing
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore, fingerprint, normalize_counterparty
from fraud_detect.synthetic import generate_tampered


@pytest.fixture()
def store(tmp_path):
    return CaseStore(str(tmp_path / "cases.json"))


def _reapply(src_path: str, tmp_path) -> str:
    """Re-submit the SAME statement under a new path (a doctored statement being
    re-used on a fresh application). Copying the exact file guarantees identical
    suspicious transactions, so the memory node sees a true recurrence."""
    from pathlib import Path

    src = Path(src_path)
    dst = Path(tmp_path) / ("reapply_" + src.name)
    dst.write_bytes(src.read_bytes())
    return str(dst)


def _mem_codes(report):
    return {f.code for f in report.findings if f.agent.startswith("Node5")}


def test_first_screening_has_no_memory_findings(store, tmp_path):
    gen = generate_tampered(str(tmp_path))
    report = run_pipeline(gen.path, store=store)
    store.upsert_report(report)
    assert _mem_codes(report) == set()


def test_rejected_pattern_is_flagged_on_recurrence(store, tmp_path):
    gen1 = generate_tampered(str(tmp_path))
    r1 = run_pipeline(gen1.path, store=store)
    store.upsert_report(r1)
    store.record_decision(r1.case_id, ReviewDecision.REJECTED,
                          reviewer="Tester", reason="confirmed")

    # Re-submit the same statement under a different filename (doctored file
    # reused on a fresh application) so fingerprints are guaranteed identical.
    r2 = run_pipeline(_reapply(gen1.path, tmp_path), store=store)

    assert "MEM_RECURRING_KNOWN_BAD" in _mem_codes(r2)
    # Every memory finding must cite the prior case id (auditability).
    for f in r2.findings:
        if f.agent.startswith("Node5"):
            assert r1.case_id in f.evidence


def test_cleared_pattern_reports_passed_before(store, tmp_path):
    gen1 = generate_tampered(str(tmp_path))
    r1 = run_pipeline(gen1.path, store=store)
    store.upsert_report(r1)
    store.record_decision(r1.case_id, ReviewDecision.APPROVED,
                          reviewer="Tester", reason="explained")

    r2 = run_pipeline(_reapply(gen1.path, tmp_path), store=store)

    assert "MEM_SIMILAR_PASSED_BEFORE" in _mem_codes(r2)


def test_memory_never_auto_rejects(store, tmp_path):
    gen1 = generate_tampered(str(tmp_path))
    r1 = run_pipeline(gen1.path, store=store)
    store.upsert_report(r1)
    store.record_decision(r1.case_id, ReviewDecision.REJECTED,
                          reviewer="Tester", reason="confirmed")

    r2 = run_pipeline(_reapply(gen1.path, tmp_path), store=store)
    # The strongest automated outcome is still HUMAN_REVIEW.
    assert r2.routing in (Routing.HUMAN_REVIEW, Routing.AUTO_CLEAR)
    assert r2.routing != "REJECTED"


def test_pipeline_without_store_runs_no_memory(tmp_path):
    gen = generate_tampered(str(tmp_path))
    report = run_pipeline(gen.path)  # no store -> no memory node
    assert _mem_codes(report) == set()
    assert report.transaction_fingerprints  # fingerprints still attached


def test_normalization_collapses_entity_suffixes():
    a = normalize_counterparty("Offshore Nominee A Pte Ltd")
    b = normalize_counterparty("Offshore Nominee A")
    assert a == b == "offshore nominee a"


def test_fingerprint_amount_banding_matches_near_amounts():
    fa = fingerprint("Shell Co", 150_000.0)
    fb = fingerprint("Shell Co", 149_500.0)
    # Both land in the same 5k bucket -> same match key.
    assert fa.key() == fb.key()
