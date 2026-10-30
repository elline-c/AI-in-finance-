"""Property-based tests for the ``AuditEvent`` field-format invariant.

Feature: immutable-audit-trail, Property 4: Field-format invariant.

These validate Correctness Property 4 from the design: every constructible
``AuditEvent`` has a non-negative integer ``seq`` and 64-lowercase-hex
``prev_hash``/``entry_hash``, and malformed values for those fields are
rejected with a pydantic ``ValidationError``.

Validates: Requirements 2.5

Positive property:
    For VALID generated inputs (``seq >= 0``; ``prev_hash``/``entry_hash`` a
    64-lowercase-hex string; arbitrary unicode for ``ts``/``case_id``/``event``/
    ``detail``; ``signature``/``key_id`` either ``None`` or a string),
    ``AuditEvent(**fields)`` constructs successfully and round-trips — every
    stored field equals the corresponding input.

Negative properties (each raises ``ValidationError``):
    * a negative ``seq``;
    * a ``prev_hash`` or ``entry_hash`` that is the wrong length, contains
      uppercase hex, or contains non-hex characters.
"""
from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from fraud_detect.audit.models import AuditEvent

# A minimum of 100 iterations per property, per the testing strategy.
_SETTINGS = settings(max_examples=100)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

# 64 lowercase hex characters — the exact shape accepted for prev_hash/entry_hash.
hex64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)

# Arbitrary unicode strings for the free-text payload fields.
unicode_text = st.text()

# signature / key_id: either None (UNSIGNED) or an arbitrary string.
optional_str = st.one_of(st.none(), st.text())


@st.composite
def valid_event_fields(draw: st.DrawFn) -> dict:
    """Generate a dict of VALID AuditEvent field values."""
    return {
        "seq": draw(st.integers(min_value=0)),
        "prev_hash": draw(hex64),
        "entry_hash": draw(hex64),
        "ts": draw(unicode_text),
        "case_id": draw(unicode_text),
        "event": draw(unicode_text),
        "detail": draw(unicode_text),
        "signature": draw(optional_str),
        "key_id": draw(optional_str),
    }


@st.composite
def invalid_hex(draw: st.DrawFn) -> str:
    """Generate a string that is NOT 64 lowercase hex characters."""
    kind = draw(st.sampled_from(["wrong_length", "uppercase", "non_hex"]))
    if kind == "wrong_length":
        # Any length of lowercase hex except exactly 64 (includes empty).
        value = draw(
            st.text(alphabet="0123456789abcdef", min_size=0, max_size=128).filter(
                lambda s: len(s) != 64
            )
        )
        return value
    if kind == "uppercase":
        # 64 chars with at least one uppercase hex letter.
        prefix = draw(st.text(alphabet="0123456789abcdef", min_size=0, max_size=63))
        upper = draw(st.sampled_from(list("ABCDEF")))
        suffix_len = 63 - len(prefix)
        suffix = draw(
            st.text(
                alphabet="0123456789abcdef",
                min_size=suffix_len,
                max_size=suffix_len,
            )
        )
        return prefix + upper + suffix
    # non_hex: 64 chars with at least one non-hex character.
    prefix = draw(st.text(alphabet="0123456789abcdef", min_size=0, max_size=63))
    # characters outside the hex alphabet
    bad = draw(st.sampled_from(list("ghijklmnopqrstuvwxyz!@# _-/")))
    suffix_len = 63 - len(prefix)
    suffix = draw(
        st.text(alphabet="0123456789abcdef", min_size=suffix_len, max_size=suffix_len)
    )
    return prefix + bad + suffix


# --------------------------------------------------------------------------- #
# Positive property: valid inputs construct and round-trip
# --------------------------------------------------------------------------- #


@_SETTINGS
@given(fields=valid_event_fields())
def test_valid_fields_construct_and_round_trip(fields: dict) -> None:
    """Valid inputs build an AuditEvent whose stored fields equal the inputs."""
    event = AuditEvent(**fields)

    for name, expected in fields.items():
        assert getattr(event, name) == expected

    # And the invariant holds on the constructed object.
    assert event.seq >= 0
    assert _HEX64_RE.fullmatch(event.prev_hash)
    assert _HEX64_RE.fullmatch(event.entry_hash)


# --------------------------------------------------------------------------- #
# Negative property: negative seq is rejected
# --------------------------------------------------------------------------- #


@_SETTINGS
@given(seq=st.integers(max_value=-1), prev_hash=hex64, entry_hash=hex64)
def test_negative_seq_rejected(seq: int, prev_hash: str, entry_hash: str) -> None:
    """A negative ``seq`` raises ValidationError."""
    with pytest.raises(ValidationError):
        AuditEvent(
            seq=seq,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            ts="2026-01-01T00:00:00Z",
            case_id="CASE-1",
            event="SCREENED",
            detail="detail",
        )


# --------------------------------------------------------------------------- #
# Negative property: malformed prev_hash is rejected
# --------------------------------------------------------------------------- #


@_SETTINGS
@given(prev_hash=invalid_hex(), entry_hash=hex64)
def test_malformed_prev_hash_rejected(prev_hash: str, entry_hash: str) -> None:
    """A prev_hash that is wrong length / uppercase / non-hex is rejected."""
    assert not _HEX64_RE.fullmatch(prev_hash)
    with pytest.raises(ValidationError):
        AuditEvent(
            seq=0,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            ts="2026-01-01T00:00:00Z",
            case_id="CASE-1",
            event="SCREENED",
            detail="detail",
        )


# --------------------------------------------------------------------------- #
# Negative property: malformed entry_hash is rejected
# --------------------------------------------------------------------------- #


@_SETTINGS
@given(prev_hash=hex64, entry_hash=invalid_hex())
def test_malformed_entry_hash_rejected(prev_hash: str, entry_hash: str) -> None:
    """An entry_hash that is wrong length / uppercase / non-hex is rejected."""
    assert not _HEX64_RE.fullmatch(entry_hash)
    with pytest.raises(ValidationError):
        AuditEvent(
            seq=0,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            ts="2026-01-01T00:00:00Z",
            case_id="CASE-1",
            event="SCREENED",
            detail="detail",
        )
