"""Canonical serialization and hashing primitives for the audit chain.

This module owns the deterministic, reproducible byte representation that both
the hash chain and the Ed25519 signature are computed over. Everything here is
pure and stdlib-only so that the same bytes (and therefore the same
``entry_hash`` and the same signed bytes) are produced across repeated
computations and across process boundaries.

Two canonical byte strings exist, both built with the same serializer:

1. **Hash payload** -> hashed to produce ``entry_hash``. It contains
   ``seq, prev_hash, ts, case_id, event, detail`` and deliberately **excludes**
   ``entry_hash`` itself (a field cannot hash over itself) while **including**
   ``prev_hash`` (that is what chains events together).
2. **Signed payload** -> the bytes the signature is computed over. It is the
   hash payload **plus** ``entry_hash`` and **excludes** ``signature`` and
   ``key_id``. Because it commits to ``entry_hash`` (which depends on
   ``prev_hash``), a valid signature transitively commits to the whole chain
   prefix.

The builders accept either a mapping or any object exposing the required
fields as attributes, so they work both for the ``AuditEvent`` model (task 2.2)
and for the raw-field append path (task 4.2).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

__all__ = [
    "GENESIS_HASH",
    "canonical_bytes",
    "hash_payload",
    "signed_payload",
    "compute_entry_hash",
]

# ``prev_hash`` of the very first event (``seq == 0``): 64 zero hex chars.
GENESIS_HASH: str = "0" * 64

# Fields that make up the hash payload, in no particular order (canonical_bytes
# sorts keys deterministically anyway).
_HASH_FIELDS = ("seq", "prev_hash", "ts", "case_id", "event", "detail")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize ``payload`` to deterministic, reproducible bytes.

    Uses sorted keys, compact separators, ``ensure_ascii=False`` and UTF-8 so
    the output is identical across repeated computations and across process
    boundaries.
    """
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _field(event: Any, name: str) -> Any:
    """Read ``name`` from ``event``, whether it is a mapping or an object."""
    if isinstance(event, Mapping):
        return event[name]
    return getattr(event, name)


def hash_payload(event: Any) -> dict:
    """Build the hash payload for ``event``.

    Returns a dict with exactly ``seq, prev_hash, ts, case_id, event, detail``.
    This EXCLUDES ``entry_hash`` (a field cannot hash over itself) and the
    signature fields. ``event`` may be a mapping or an object exposing those
    fields as attributes.
    """
    return {name: _field(event, name) for name in _HASH_FIELDS}


def signed_payload(event: Any) -> dict:
    """Build the signed payload for ``event``.

    Returns the hash payload PLUS ``entry_hash``. This INCLUDES ``entry_hash``
    and EXCLUDES ``signature`` and ``key_id`` -- it is exactly the payload the
    Ed25519 signature is computed over. ``event`` may be a mapping or an object.
    """
    payload = hash_payload(event)
    payload["entry_hash"] = _field(event, "entry_hash")
    return payload


def compute_entry_hash(payload: Mapping[str, Any]) -> str:
    """Return the hex SHA-256 over the canonical bytes of ``payload``.

    ``payload`` is expected to be a hash payload (the dict returned by
    :func:`hash_payload`). The result is 64 lowercase hex characters.
    """
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()
