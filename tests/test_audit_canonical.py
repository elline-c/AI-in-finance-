"""Property-based tests for the audit chain's canonical serialization + hashing.

Feature: immutable-audit-trail, Property 3: Deterministic canonical hashing.

For any payload, the canonical serialization and the recomputed ``entry_hash``
are identical across repeated computations and equal to the stored hash. This
is what lets the hash chain (and, transitively, the Ed25519 signature that
commits to ``entry_hash``) be reproduced across repeated computations and
across process boundaries.

Validates: Requirements 2.3, 2.4

The strategies generate varied hash payloads with the shape the chain uses
(``seq, prev_hash, ts, case_id, event, detail``). String fields include
unicode to exercise ``ensure_ascii=False`` in the serializer.
"""
from __future__ import annotations

import json
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.canonical import (
    GENESIS_HASH,
    canonical_bytes,
    compute_entry_hash,
    hash_payload,
    signed_payload,
)

# A 64-lowercase-hex string, the shape of prev_hash / entry_hash.
_hex64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")


@st.composite
def hash_payloads(draw):
    """Generate a hash payload: seq (>=0), a 64-hex prev_hash, and unicode-
    capable ts / case_id / event / detail strings."""
    return {
        "seq": draw(st.integers(min_value=0)),
        "prev_hash": draw(_hex64),
        "ts": draw(st.text()),
        "case_id": draw(st.text()),
        "event": draw(st.text()),
        "detail": draw(st.text()),
    }


@settings(max_examples=200)
@given(payload=hash_payloads())
def test_canonical_bytes_is_byte_identical_across_calls(payload):
    """Property 3 (part 1): canonical_bytes is deterministic — repeated calls
    on the same payload return byte-identical output."""
    first = canonical_bytes(payload)
    second = canonical_bytes(payload)
    assert first == second
    assert isinstance(first, bytes)


@settings(max_examples=200)
@given(payload=hash_payloads())
def test_entry_hash_is_deterministic_and_64_lower_hex(payload):
    """Property 3 (part 2): compute_entry_hash is identical across repeated
    calls and is always 64 lowercase hex characters."""
    first = compute_entry_hash(payload)
    second = compute_entry_hash(payload)
    assert first == second
    assert _HEX64_RE.match(first), f"not 64 lowercase hex: {first!r}"


@settings(max_examples=200)
@given(payload=hash_payloads())
def test_key_order_independence(payload):
    """Property 3 (part 3): building the same logical payload with keys in a
    different insertion order yields identical canonical_bytes and identical
    entry_hash, because the serializer sorts keys."""
    # Same logical content, reversed insertion order.
    reordered = {k: payload[k] for k in reversed(list(payload.keys()))}
    assert list(reordered.keys()) != list(payload.keys()) or len(payload) <= 1

    assert canonical_bytes(reordered) == canonical_bytes(payload)
    assert compute_entry_hash(reordered) == compute_entry_hash(payload)


@settings(max_examples=200)
@given(payload=hash_payloads())
def test_canonical_bytes_matches_direct_json_dumps(payload):
    """Property 3 (part 4 — cross-process determinism proxy): canonical_bytes
    equals an independent json.dumps with the same deterministic options, so
    any process computing the bytes the same documented way agrees."""
    expected = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert canonical_bytes(payload) == expected


@settings(max_examples=200)
@given(payload=hash_payloads())
def test_recomputed_entry_hash_equals_stored_hash(payload):
    """Property 3 (stored-hash equality): once an event stores the computed
    entry_hash, recomputing it over the hash payload (as verification does)
    reproduces the stored value exactly."""
    stored = compute_entry_hash(payload)

    # Simulate a full event (hash payload + stored entry_hash + signature
    # fields) the way an AuditEvent-shaped object would carry it.
    event = dict(payload)
    event["entry_hash"] = stored
    event["signature"] = None
    event["key_id"] = None

    # hash_payload must strip back to exactly the hash fields...
    recomputed = compute_entry_hash(hash_payload(event))
    assert recomputed == stored

    # ...and signed_payload must be the hash payload plus the stored hash,
    # with signature/key_id excluded.
    sp = signed_payload(event)
    assert sp == {**hash_payload(event), "entry_hash": stored}
    assert "signature" not in sp
    assert "key_id" not in sp


def test_genesis_hash_is_64_zeros():
    """GENESIS_HASH is the 64-zero prev_hash used for seq == 0."""
    assert GENESIS_HASH == "0" * 64
    assert _HEX64_RE.match(GENESIS_HASH)
