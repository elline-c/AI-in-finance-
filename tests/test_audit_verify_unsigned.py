"""Property-based tests for verifying UNSIGNED audit chains.

Feature: immutable-audit-trail, Property 7: Unsigned chains are reported as
UNSIGNED, never signed.

For any non-empty sequence of appended events written with NO signer, the chain
verifies as ``INTACT_UNSIGNED``: ``intact`` is True, ``signed`` is False, every
appended event is counted as unsigned (``unsigned_events == total_events ==``
the number appended), and the hash links are intact (never ``BROKEN_CHAIN``).
Every event read back carries ``signature is None`` and ``key_id is None``.

A critical nuance is also covered: an UNSIGNED chain must *still* be reported as
``INTACT_UNSIGNED`` even when a verifier (signer) IS available. The events carry
no signature, so there is nothing to validate and the chain must never be
reported as ``INTACT_SIGNED`` just because a key happens to be present.

Validates: Requirements 6.1, 6.2, 7.3

The strategy generates a non-empty list of ``(case_id, event, detail)`` tuples.
Each Hypothesis example appends them to a FRESH chain on its own isolated temp
file, created inside the test body, so examples never collide on a shared file.
String fields include unicode to exercise the canonical serializer.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import LocalFileSigner
from fraud_detect.audit.verify import VerificationStatus

# A non-empty list of (case_id, event, detail) tuples. Text includes unicode to
# exercise ensure_ascii=False in the canonical serializer.
_events = st.lists(
    st.tuples(st.text(), st.text(), st.text()),
    min_size=1,
    max_size=12,
)


@settings(max_examples=150)
@given(records=_events)
def test_unsigned_chain_verifies_as_intact_unsigned(records):
    """Property 7: a chain written with no signer verifies as INTACT_UNSIGNED,
    with ``signed == False`` and every appended event counted as unsigned, and
    the hash links intact (never BROKEN_CHAIN). Every event carries a null
    ``signature``/``key_id``."""
    # Fresh, isolated temp file INSIDE the test body so Hypothesis examples do
    # not collide on a shared path.
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"
        chain = AuditChain(path=str(chain_path))  # no signer -> UNSIGNED

        for case_id, event, detail in records:
            chain.append(case_id=case_id, event=event, detail=detail)

        n = len(records)
        result = chain.verify()

        # Reported plainly as intact-but-unsigned, never signed.
        assert result.status == VerificationStatus.INTACT_UNSIGNED
        assert result.intact is True
        assert result.signed is False

        # Every appended event is unsigned; counts line up with what we wrote.
        assert result.total_events == n
        assert result.unsigned_events == n
        assert result.unsigned_events == result.total_events

        # Hash links are intact: no break reported.
        assert result.first_broken_seq is None
        assert result.first_bad_signature_seq is None

        # Every event read back carries no signature material.
        events = chain._read_all_events()
        assert len(events) == n
        for ev in events:
            assert ev.signature is None
            assert ev.key_id is None


@settings(max_examples=150)
@given(records=_events)
def test_unsigned_chain_stays_unsigned_even_with_a_verifier(records):
    """Property 7 (nuance): an UNSIGNED chain verified through an AuditChain that
    DOES have a signer is still INTACT_UNSIGNED — never INTACT_SIGNED. The events
    carry no signature, so a present key must not promote the chain to signed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"

        # Write the chain WITHOUT a signer -> all events UNSIGNED.
        writer = AuditChain(path=str(chain_path))  # no signer
        for case_id, event, detail in records:
            writer.append(case_id=case_id, event=event, detail=detail)

        n = len(records)

        # Now point a NEW AuditChain at the same file, this time WITH a signer
        # available for verification. The events still have no signature.
        signer = LocalFileSigner.generate(str(Path(tmp_dir) / "key.pem"))
        verifier_chain = AuditChain(path=str(chain_path), signer=signer)
        result = verifier_chain.verify()

        # Still reported as intact-but-unsigned; a present verifier must not make
        # an unsigned chain look signed.
        assert result.status == VerificationStatus.INTACT_UNSIGNED
        assert result.status != VerificationStatus.INTACT_SIGNED
        assert result.intact is True
        assert result.signed is False
        assert result.total_events == n
        assert result.unsigned_events == n
        assert result.first_broken_seq is None
        assert result.first_bad_signature_seq is None
