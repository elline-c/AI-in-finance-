"""Property-based test for the signed-chain verification round-trip.

Feature: immutable-audit-trail, Property 5: Signed chain round-trip verifies as
INTACT_SIGNED.

A chain built with a configured signer, after appending any non-empty sequence
of events, verifies as ``INTACT_SIGNED``: the hash chain walks cleanly and every
event's Ed25519 signature validates under the public key for its ``key_id``. The
result's ``total_events`` equals the number appended, ``unsigned_events`` is 0,
and every event read back carries a non-null ``signature`` and a ``key_id`` equal
to the signer's.

Validates: Requirements 4.1, 4.3, 7.2

Strategy: Hypothesis generates a non-empty list of ``(case_id, event, detail)``
tuples. Each example builds a chain on its OWN fresh temp file created inside the
test body (so examples never share chain state), appends all records with a
single module-scoped signer, then calls ``chain.verify()``.

Isolation + performance notes: Ed25519 key generation is comparatively
expensive, so a single signer is generated once per module (``module_signer``)
and reused across all examples -- the property under test does not depend on
which key is used. Each example still gets a brand-new temp chain file so no
signed-chain state leaks between examples.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import LocalFileSigner
from fraud_detect.audit.verify import VerificationStatus

# A non-empty list of (case_id, event, detail) tuples. Text includes unicode to
# exercise the canonical serializer (ensure_ascii=False) and the signed payload.
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

    Generating a keypair per Hypothesis example would dominate the runtime, so
    the signed round-trip property reuses this one signer across all examples.
    The property (a signed chain verifies as INTACT_SIGNED) does not depend on
    which specific key is used.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield LocalFileSigner.generate(str(Path(tmp) / "key.pem"))


@settings(max_examples=150, deadline=None)
@given(specs=_event_specs)
def test_signed_chain_round_trip_verifies_intact_signed(specs, module_signer):
    """Property 5: a chain built with a configured signer verifies as
    INTACT_SIGNED, with total_events equal to the number appended, zero unsigned
    events, and every event carrying a non-null signature and the signer's
    key_id."""
    # Fresh, isolated temp file INSIDE the test body so Hypothesis examples do
    # not collide on a shared path.
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"
        chain = AuditChain(path=str(chain_path), signer=module_signer)

        for case_id, event, detail in specs:
            chain.append(case_id=case_id, event=event, detail=detail)

        result = chain.verify()

        # Overall verdict: hash chain intact AND every event validly signed.
        assert result.status == VerificationStatus.INTACT_SIGNED
        assert result.intact is True
        assert result.signed is True

        # Counts: every appended event is present and none are unsigned.
        assert result.total_events == len(specs)
        assert result.unsigned_events == 0

        # Every event read back is genuinely signed under the module signer.
        events = chain._read_all_events()
        assert len(events) == len(specs)
        for ev in events:
            assert ev.signature is not None
            assert ev.key_id == module_signer.key_id
