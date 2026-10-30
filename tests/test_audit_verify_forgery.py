"""Property-based test for the headline audit-chain security property.

Feature: immutable-audit-trail, Property 10: Rewrite-with-rehash forgery is
still caught.

This is the property signing exists for. A plain hash chain can be forged by an
attacker who edits an event's content and recomputes every downstream
``entry_hash`` so the chain walks cleanly again -- a pure-hash verifier would be
fooled. Signing closes that gap: the signature commits to ``entry_hash``, so a
rewrite that recomputes hashes can make pass-1 (the hash chain) pass, but the
attacker cannot produce a valid signature over the new signed payload without the
private key. Verification therefore catches the forgery in pass-2 and returns
``SIGNATURE_INVALID`` rather than ``INTACT_SIGNED``.

The attacker simulated here CAN recompute hashes (the hashing helpers are public)
but CANNOT re-sign (no private key), so the rewritten event keeps its old,
now-mismatching signature.

Validates: Requirements 7.7

Strategy: Hypothesis builds an intact SIGNED chain of >= 1 event on a fresh temp
file per example (reusing one module-scoped ``LocalFileSigner`` because keygen is
expensive and the property does not depend on the specific key), then picks an
index ``i`` to tamper. It changes event[i]'s ``detail``, recomputes event[i]'s
``entry_hash``, and re-links the chain from ``i`` onward so the hash chain is
internally consistent again -- while leaving every signature untouched. Verifying
the rewritten chain must return ``SIGNATURE_INVALID`` at ``seq == i`` with
``first_broken_seq is None`` (the hash chain was made consistent, so the failure
is purely a signature failure).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit import canonical
from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.models import AuditEvent
from fraud_detect.audit.signer import LocalFileSigner
from fraud_detect.audit.verify import VerificationStatus

# A non-empty list of (case_id, event, detail) tuples. Text includes unicode to
# exercise the canonical serializer (ensure_ascii=False) across hash and signed
# payloads. ``data=st.data()`` is used in the test body to pick the tamper index
# against the concrete number of events built.
_event_specs = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=40),   # case_id (non-empty to be meaningful)
        st.text(min_size=0, max_size=40),   # event
        st.text(min_size=0, max_size=200),  # detail
    ),
    min_size=1,
    max_size=20,
)


@pytest.fixture(scope="module")
def module_signer() -> LocalFileSigner:
    """A single Ed25519 signer generated once for the whole module.

    Ed25519 keygen per Hypothesis example would dominate the runtime. The forgery
    property does not depend on which key signs the honest chain -- only that the
    attacker lacks that key -- so one signer is reused across all examples. Each
    example still gets a brand-new temp chain file so no state leaks between runs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield LocalFileSigner.generate(str(Path(tmp) / "key.pem"))


def _forge_with_rehash(events: list[AuditEvent], i: int) -> list[AuditEvent]:
    """Simulate the attacker: edit event ``i``'s content, recompute its and all
    downstream ``entry_hash``/``prev_hash`` so the hash chain is consistent, but
    keep every (now-mismatching) signature -- the attacker cannot re-sign.

    Returns a NEW list of :class:`AuditEvent`; the input is not mutated.
    """
    forged: list[AuditEvent] = [ev.model_copy() for ev in events]

    # 1. Change event[i]'s content (its detail) to something different, then
    #    recompute its entry_hash from the new hash payload. The attacker CAN do
    #    this -- the hashing helpers are public.
    tampered = forged[i]
    tampered.detail = tampered.detail + "\u2622 forged"  # guaranteed different
    tampered.entry_hash = canonical.compute_entry_hash(
        canonical.hash_payload(tampered)
    )
    # Signature is deliberately left as-is: it still signs the OLD payload, so it
    # no longer validates over the new signed bytes (which include entry_hash).

    # 2. Re-link the chain from i+1 onward so prev_hash matches the prior
    #    (recomputed) entry_hash and each downstream entry_hash is consistent.
    #    This makes pass-1 (the hash chain) walk cleanly again.
    for j in range(i + 1, len(forged)):
        forged[j].prev_hash = forged[j - 1].entry_hash
        forged[j].entry_hash = canonical.compute_entry_hash(
            canonical.hash_payload(forged[j])
        )
        # Downstream signatures are also left untouched (attacker cannot re-sign),
        # but the FIRST failing signature is event i, which is what we assert on.

    return forged


@settings(max_examples=150, deadline=None)
@given(specs=_event_specs, data=st.data())
def test_rewrite_with_rehash_forgery_is_caught(specs, data, module_signer):
    """Property 10: an attacker who edits an event and recomputes all downstream
    hashes (so the hash chain walks cleanly) but cannot re-sign is caught at the
    signature pass -- verify() returns SIGNATURE_INVALID (not INTACT_SIGNED), at
    the tampered event's seq, with no hash-chain break reported."""
    # Fresh, isolated temp file INSIDE the test body so examples never collide.
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"
        chain = AuditChain(path=str(chain_path), signer=module_signer)

        # Build an intact SIGNED chain.
        for case_id, event, detail in specs:
            chain.append(case_id=case_id, event=event, detail=detail)

        events = chain._read_all_events()
        assert len(events) == len(specs)

        # Sanity: the honest chain verifies as INTACT_SIGNED before tampering.
        assert chain.verify().status == VerificationStatus.INTACT_SIGNED

        # Pick which event the attacker rewrites.
        i = data.draw(st.integers(min_value=0, max_value=len(events) - 1))

        forged = _forge_with_rehash(events, i)

        # Write the forged chain back to the JSONL file (full rewrite, as an
        # attacker with file access would do).
        chain_path.write_text(
            "".join(ev.model_dump_json() + "\n" for ev in forged),
            encoding="utf-8",
        )

        # The forgery is caught at the signature pass, NOT accepted as signed.
        result = chain.verify()
        assert result.status == VerificationStatus.SIGNATURE_INVALID
        assert result.status != VerificationStatus.INTACT_SIGNED

        # Failure is at the tampered event, purely a signature failure: the hash
        # chain was made consistent so no break is reported.
        assert result.first_bad_signature_seq == i
        assert result.first_broken_seq is None
        assert result.intact is True   # hash chain walks cleanly
        assert result.signed is False  # but it is not validly signed
