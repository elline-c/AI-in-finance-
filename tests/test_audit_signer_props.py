"""Property-based tests for the audit chain's signing backend.

Feature: immutable-audit-trail, Property 6: Signatures resolve by key_id and
wrong keys fail.

For any payload and any generated keypair, a signature produced by key A
validates under A and fails under a different key B; a mismatched/unknown
``key_id`` yields a verification failure. This is what lets verification resolve
the correct public key per event (by ``key_id``) and reject signatures that do
not belong to the resolved key.

Validates: Requirements 4.4, 5.4

Key generation is relatively expensive, so two module-scoped signers (A and B,
with distinct keys) are generated once via ``LocalFileSigner.generate`` into a
temporary directory and reused across all data/byte-varying examples. The
data-varying properties still run >= 100 examples.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.signer import LocalFileSigner

# An unknown key_id that no generated signer will ever report. The real key_id
# is "ed25519:" + 12 hex chars; this fixed value is astronomically unlikely to
# collide with a freshly generated key's fingerprint.
UNKNOWN_KEY_ID = "ed25519:deadbeef0000"


@pytest.fixture(scope="module")
def signers():
    """Generate two signers (A and B) with distinct keys, once per module.

    Keys are written to a temp dir (mirroring a real on-disk key file) and the
    directory is cleaned up when the module's tests finish.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        signer_a = LocalFileSigner.generate(str(tmp_path / "key_a.pem"))
        signer_b = LocalFileSigner.generate(str(tmp_path / "key_b.pem"))
        # Two freshly generated Ed25519 keys must have different fingerprints;
        # if this ever failed the whole premise of the test would be invalid.
        assert signer_a.key_id != signer_b.key_id
        yield signer_a, signer_b


# Arbitrary byte payloads, including the empty payload, to stand in for the
# canonical signed bytes the chain actually signs.
_data = st.binary(min_size=0, max_size=512)


@settings(max_examples=150)
@given(data=_data)
def test_round_trip_signature_validates_under_same_key(signers, data):
    """Property 6 (round-trip): a signature from signer A validates under A's
    own key_id."""
    signer_a, _ = signers
    sig = signer_a.sign(data)
    assert signer_a.verify(data, sig, key_id=signer_a.key_id) is True


@settings(max_examples=150)
@given(data=_data)
def test_signature_from_a_does_not_verify_under_b(signers, data):
    """Property 6 (wrong key): a signature produced by A does NOT verify under a
    different signer B (B does not hold A's key_id, so resolution fails)."""
    signer_a, signer_b = signers
    sig_a = signer_a.sign(data)
    # B cannot resolve A's key_id -> verification is False.
    assert signer_b.verify(data, sig_a, key_id=signer_b.key_id) is False
    # And asking B to verify under A's key_id also fails (B doesn't know it).
    assert signer_b.verify(data, sig_a, key_id=signer_a.key_id) is False


@settings(max_examples=150)
@given(data=_data)
def test_unknown_key_id_fails_and_resolves_to_none(signers, data):
    """Property 6 (unknown key_id): verifying with a key_id the signer doesn't
    know returns False, and public_key_for(unknown_id) returns None."""
    signer_a, _ = signers
    sig_a = signer_a.sign(data)
    assert signer_a.verify(data, sig_a, key_id=UNKNOWN_KEY_ID) is False
    assert signer_a.public_key_for(UNKNOWN_KEY_ID) is None
    # The signer's own key_id resolves to a real public key (sanity anchor).
    assert signer_a.public_key_for(signer_a.key_id) is not None


@settings(max_examples=150)
@given(data=_data, suffix=st.binary(min_size=1, max_size=16))
def test_tampered_data_does_not_verify(signers, data, suffix):
    """Property 6 (tampered data): a signature over ``data`` does not verify
    over ``tampered`` when tampered != data."""
    signer_a, _ = signers
    sig = signer_a.sign(data)
    tampered = data + suffix  # appending >=1 byte guarantees tampered != data
    assert tampered != data
    assert signer_a.verify(tampered, sig, key_id=signer_a.key_id) is False
