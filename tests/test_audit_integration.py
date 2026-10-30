"""End-to-end integration test — screen + human decision chain verifies.

Drives a screening (``CaseStore.upsert_report``) then a human decision
(``CaseStore.record_decision``) through a real ``CaseStore`` and asserts that
the signed, hash-chained audit trail records exactly the two expected events in
order (``SCREENED`` then ``HUMAN_REJECTED``) and that verification reports the
right integrity status in both the UNSIGNED and SIGNED configurations.

This exercises the task 8.1/8.2 wiring (CaseStore -> AuditChain) rather than the
chain primitives in isolation: the ``event``/``detail``/``case_id`` recorded on
the chain must match what the cases.json ``audit_log`` mirror recorded, and the
human decision — never the system — is what produces the ``HUMAN_REJECTED``
event (governance: the chain records, it never auto-rejects).

**Validates: Requirements 3.1, 3.2, 9.2**
"""
from __future__ import annotations

import pytest

from fraud_detect.audit import LocalFileSigner
from fraud_detect.audit.signer import AUDIT_SIGNING_KEY_ENV
from fraud_detect.audit.verify import VerificationStatus
from fraud_detect.models import ReviewDecision
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_tampered

REVIEWER = "IntegrationTester"
REASON = "confirmed tampering on review"


def _screen_and_reject(store: CaseStore, tmp_path):
    """Run a real screening then a human REJECT through ``store``.

    Returns the ``case_id`` so the caller can inspect the chain for that case.
    The report is produced by the real pipeline on a generated tampered sample
    (same approach as tests/test_memory.py), so it is a genuinely valid
    ``RiskReport`` with a ``case_id`` — the lightest consistent way to get one.
    """
    gen = generate_tampered(str(tmp_path))
    report = run_pipeline(gen.path)
    case = store.upsert_report(report)
    decided = store.record_decision(
        case.case_id, ReviewDecision.REJECTED, reviewer=REVIEWER, reason=REASON,
    )
    assert decided is not None
    # Governance (Req 9.2 spirit): the REJECTED state is set by the human
    # decision, not fabricated by the system. The recorded decision reflects
    # exactly what the human chose.
    assert decided.review.decision == ReviewDecision.REJECTED
    assert decided.review.reviewer == REVIEWER
    return case.case_id


def _assert_two_event_chain(store: CaseStore, case_id: str):
    """Assert the chain holds exactly SCREENED then HUMAN_REJECTED for ``case_id``.

    Cross-checks each chain event's ``detail`` against the cases.json
    ``audit_log`` mirror so the authoritative chain and the UI mirror agree
    (Req 3.1/3.2/3.3).
    """
    events = store._audit.events_for(case_id)
    assert len(events) == 2, f"expected 2 events, got {len(events)}"

    screened, rejected = events
    assert screened.event == "SCREENED"
    assert rejected.event == "HUMAN_REJECTED"
    assert screened.case_id == case_id
    assert rejected.case_id == case_id

    # The human's decision — reviewer + reason — is reflected in the recorded
    # HUMAN_REJECTED event (Req 9.2: the chain records the human decision).
    assert REVIEWER in rejected.detail
    assert REASON in rejected.detail

    # The chain detail must match what the cases.json audit_log mirror recorded.
    case = store.get(case_id)
    assert case is not None
    mirror = {entry["event"]: entry["detail"] for entry in case.audit_log}
    assert mirror["SCREENED"] == screened.detail
    assert mirror["HUMAN_REJECTED"] == rejected.detail

    return screened, rejected


def test_unsigned_chain_records_two_events_and_verifies_intact_unsigned(
    tmp_path, monkeypatch
):
    """No key configured -> chain is written UNSIGNED but fully intact.

    The screen + reject flow records SCREENED then HUMAN_REJECTED, and
    verification reports INTACT_UNSIGNED (never signed). monkeypatch.delenv
    ensures no leaked key path from the environment flips this to signed.
    """
    # Ensure UNSIGNED: no signing key visible to this CaseStore.
    monkeypatch.delenv(AUDIT_SIGNING_KEY_ENV, raising=False)

    store = CaseStore(
        str(tmp_path / "cases.json"),
        audit_path=str(tmp_path / "audit_chain.jsonl"),
    )

    case_id = _screen_and_reject(store, tmp_path)
    _assert_two_event_chain(store, case_id)

    # UNSIGNED events carry no signature / key_id.
    for event in store._audit.events_for(case_id):
        assert event.signature is None
        assert event.key_id is None

    result = store._audit.verify()
    assert result.status == VerificationStatus.INTACT_UNSIGNED
    assert result.intact is True
    assert result.signed is False
    assert result.total_events == 2
    assert result.unsigned_events == 2


def test_signed_chain_records_two_events_and_verifies_intact_signed(
    tmp_path, monkeypatch
):
    """Key configured -> chain is written SIGNED and verifies INTACT_SIGNED.

    The signing key is generated to a path OUTSIDE the data dir (Req 5.3) and
    pointed at via monkeypatch.setenv, so a NEW CaseStore builds a signer from
    the environment. Both recorded events must carry a signature and the
    signer's key_id, and verification must report INTACT_SIGNED.
    """
    # Generate an Ed25519 key OUTSIDE the data dir and point the env var at it.
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    key_path = key_dir / "audit_signing_key.pem"
    signer = LocalFileSigner.generate(str(key_path))
    monkeypatch.setenv(AUDIT_SIGNING_KEY_ENV, str(key_path))

    # NEW CaseStore so __init__ builds the signer from the environment.
    data_dir = tmp_path / "data"
    store = CaseStore(
        str(data_dir / "cases.json"),
        audit_path=str(data_dir / "audit_chain.jsonl"),
    )

    case_id = _screen_and_reject(store, tmp_path)
    _assert_two_event_chain(store, case_id)

    # Every recorded event carries a signature + the signer's key_id.
    for event in store._audit.events_for(case_id):
        assert event.signature is not None and event.signature != ""
        assert event.key_id == signer.key_id

    result = store._audit.verify()
    assert result.status == VerificationStatus.INTACT_SIGNED
    assert result.intact is True
    assert result.signed is True
    assert result.total_events == 2
    assert result.unsigned_events == 0
