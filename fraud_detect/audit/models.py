"""The ``AuditEvent`` data model — one immutable line of the audit chain.

An ``AuditEvent`` is the unit written to ``data/audit_chain.jsonl`` (one JSON
object per line). It carries three groups of fields:

* **chain structural fields** — ``seq``, ``prev_hash``, ``entry_hash`` — which
  link each event to its predecessor (``prev_hash``) and commit to its own
  content (``entry_hash``);
* **payload fields** — ``ts``, ``case_id``, ``event``, ``detail`` — the record
  of what actually happened; and
* **signature fields** — ``signature`` and ``key_id`` — the optional Ed25519
  signature (base64) and the id of the key that produced it. Both are ``None``
  for an UNSIGNED event (no signing key was configured when it was written).

The model validates only the structural field *formats* that must hold for
every event regardless of how it is produced: ``seq`` is a non-negative
integer, and ``prev_hash``/``entry_hash`` are exactly 64 lowercase hex
characters. The cross-field signing rules ("``key_id`` present iff
``signature`` present", "signature validates over the signed payload") live in
the verifier (task 6), not here — the model stays permissive so UNSIGNED events
can always be constructed.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, field_validator

from . import canonical

__all__ = ["AuditEvent"]

# 64 lowercase hex characters — the shape of both ``prev_hash`` and
# ``entry_hash`` (a hex SHA-256 digest, or ``GENESIS_HASH`` for ``prev_hash``
# of the genesis event, which is also 64 lowercase hex chars).
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class AuditEvent(BaseModel):
    """One immutable, hash-chained (optionally signed) audit event."""

    # --- chain structural fields ---
    seq: int                 # monotonically increasing; 0 for the genesis event
    prev_hash: str           # hex sha256 of the prior entry_hash; GENESIS_HASH at seq 0
    entry_hash: str          # hex sha256 over the canonical hash payload

    # --- payload: what actually happened ---
    ts: str                  # UTC ISO-8601 timestamp
    case_id: str             # which case this event belongs to
    event: str               # "SCREENED" | "HUMAN_APPROVED" | "HUMAN_REJECTED" | ...
    detail: str              # human-readable detail string

    # --- signature fields (None => UNSIGNED) ---
    signature: Optional[str] = None  # base64 Ed25519 signature over the signed payload
    key_id: Optional[str] = None     # id of the key that produced `signature`

    @field_validator("seq")
    @classmethod
    def _seq_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seq must be a non-negative integer")
        return value

    @field_validator("prev_hash", "entry_hash")
    @classmethod
    def _hex64(cls, value: str) -> str:
        if not _HEX64_RE.fullmatch(value):
            raise ValueError(
                "must be exactly 64 lowercase hexadecimal characters"
            )
        return value

    def compute_hash(self) -> str:
        """Recompute this event's ``entry_hash`` from its hash payload.

        Convenience over :mod:`fraud_detect.audit.canonical`: hashes the
        canonical hash payload (``seq, prev_hash, ts, case_id, event, detail``)
        and returns the hex SHA-256. The verifier uses this to check that the
        stored ``entry_hash`` matches the content.
        """
        return canonical.compute_entry_hash(canonical.hash_payload(self))
