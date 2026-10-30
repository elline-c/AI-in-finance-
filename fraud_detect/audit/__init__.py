"""Signed, hash-chained, append-only audit trail for the fraud_detect system.

This subpackage makes the audit record tamper-evident and forgery-resistant.
Audit events are written to a dedicated append-only JSONL store at
``data/audit_chain.jsonl`` that is never rewritten or truncated in place, while
the mutable case snapshot continues to live in ``data/cases.json``.

Each event is cryptographically linked to its predecessor by a hash chain, so
any edit, deletion, reorder, or insertion of a past event breaks the chain and
is detectable by re-walking it. Each event is additionally signed (Ed25519) over
a canonical payload that commits to the chained ``entry_hash``, so a full-file
rewrite with recomputed hashes cannot forge a verifiable chain without the
private key. Signing is pluggable behind a ``Signer`` interface so the
prototype's local-file key can be swapped for a production KMS/HSM backend.

Public names are exported here as the later implementation tasks land
(``AuditEvent``, ``AuditChain``, ``VerificationResult``, the ``Signer``
interface, etc.).
"""

from .chain import AuditChain
from .models import AuditEvent
from .signer import LocalFileSigner, Signer, build_signer_from_env
from .verify import VerificationResult, VerificationStatus

__all__: list[str] = [
    "AuditEvent",
    "AuditChain",
    "Signer",
    "LocalFileSigner",
    "build_signer_from_env",
    "VerificationResult",
    "VerificationStatus",
]
