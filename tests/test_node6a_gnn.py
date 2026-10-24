"""Node 6a GNN-wiring tests (Task 7.1).

Exercises the GNN signal added to ``fraud_detect.nodes.ml_transaction.analyze``
with a stub precomputed-scores store passed directly as ``gnn_scores`` (no torch,
no trained artifact needed — the store is a plain dict). The GNN path is
independent of the GBM path, so these tests pass ``model=None`` and assert the
account-level ``ML_TXN_GNN_RISK`` finding behaves per spec:

- Elevated account -> exactly one ML_TXN_GNN_RISK finding at the right severity
  (HIGH >= 0.80, MEDIUM >= 0.50, below 0.50 -> no finding).
- No store (gnn_scores=None) -> current behaviour unchanged (no GNN finding).
- The node never auto-rejects: Findings are evidence only.
"""
from __future__ import annotations

import pytest

from fraud_detect.models import DocumentMetadata, IngestionResult, Severity, Transaction
from fraud_detect.nodes import ml_transaction
from fraud_detect.nodes.ml_transaction import AGENT_GNN


def _ingestion(uen: str = "53333333A", with_txns: bool = True) -> IngestionResult:
    """Minimal screened statement for one account under review."""
    txns = []
    if with_txns:
        txns = [
            Transaction(index=0, description="Transfer in", amount=149_000.0, balance=149_000.0, page=1),
            Transaction(index=1, description="Transfer out", amount=-148_500.0, balance=500.0, page=1),
        ]
    return IngestionResult(
        source_path="/tmp/stmt.pdf",
        page_count=1,
        raw_text="",
        metadata=DocumentMetadata(),
        uen=uen,
        transactions=txns,
    )


def _stub_store(account_scores: dict) -> dict:
    """A plain scores store matching the Task 5 format."""
    return {"account_scores": account_scores, "edge_scores": {}, "meta": {}}


def _gnn_findings(report):
    return [f for f in report.findings if f.code == "ML_TXN_GNN_RISK"]


# --- elevated account -> exactly one finding at the right severity ----------

def test_high_score_yields_one_high_gnn_finding():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"53333333A": 0.87})

    report = ml_transaction.analyze(ing, model=None, gnn_scores=store)

    gnn = _gnn_findings(report)
    assert len(gnn) == 1
    f = gnn[0]
    assert f.code == "ML_TXN_GNN_RISK"
    assert f.agent == AGENT_GNN == "Node6a:MLTransaction:GNN"
    assert f.severity == Severity.HIGH


def test_score_at_high_boundary_is_high():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"53333333A": 0.80})

    gnn = _gnn_findings(ml_transaction.analyze(ing, model=None, gnn_scores=store))
    assert len(gnn) == 1
    assert gnn[0].severity == Severity.HIGH


def test_medium_score_yields_one_medium_gnn_finding():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"53333333A": 0.63})

    gnn = _gnn_findings(ml_transaction.analyze(ing, model=None, gnn_scores=store))
    assert len(gnn) == 1
    assert gnn[0].severity == Severity.MEDIUM


def test_score_at_medium_boundary_is_medium():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"53333333A": 0.50})

    gnn = _gnn_findings(ml_transaction.analyze(ing, model=None, gnn_scores=store))
    assert len(gnn) == 1
    assert gnn[0].severity == Severity.MEDIUM


def test_below_medium_score_yields_no_gnn_finding():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"53333333A": 0.49})

    gnn = _gnn_findings(ml_transaction.analyze(ing, model=None, gnn_scores=store))
    assert gnn == []


# --- fail-safe: no store / unknown key / no transactions --------------------

def test_no_store_leaves_behaviour_unchanged():
    """gnn_scores=None must behave exactly as the pre-GNN node (no GNN finding)."""
    ing = _ingestion(uen="53333333A")

    report = ml_transaction.analyze(ing, model=None, gnn_scores=None)

    assert _gnn_findings(report) == []
    # model=None + no store -> no findings at all, identical to today.
    assert report.findings == []
    assert report.agent == "Node6a:MLTransaction"


def test_unknown_account_key_yields_no_gnn_finding():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"99999999Z": 0.95})  # a different account

    gnn = _gnn_findings(ml_transaction.analyze(ing, model=None, gnn_scores=store))
    assert gnn == []


def test_no_transactions_yields_no_gnn_finding():
    """A GNN finding requires the account to appear in a screened statement."""
    ing = _ingestion(uen="53333333A", with_txns=False)
    store = _stub_store({"53333333A": 0.95})

    report = ml_transaction.analyze(ing, model=None, gnn_scores=store)
    assert _gnn_findings(report) == []


# --- governance: evidence only, never an auto-reject ------------------------

def test_gnn_finding_is_evidence_only_never_rejects():
    ing = _ingestion(uen="53333333A")
    store = _stub_store({"53333333A": 0.91})

    report = ml_transaction.analyze(ing, model=None, gnn_scores=store)
    gnn = _gnn_findings(report)
    assert len(gnn) == 1
    f = gnn[0]

    # The finding carries only evidence fields; nothing on it marks a rejection.
    assert f.severity in (Severity.MEDIUM, Severity.HIGH)
    assert "never an automatic rejection" in f.evidence
    assert "graph" in f.evidence.lower()
    # Finding has no decision/routing field -> it cannot auto-reject by contract.
    assert not hasattr(f, "decision")
    assert not hasattr(f, "routing")
