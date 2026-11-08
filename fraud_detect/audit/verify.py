"""Verification result model for the audit chain.

The verifier (task 6.2) re-walks the append-only chain and reports whether the
audit trail is intact, signed, broken, or carries an invalid signature. This
module defines the *result* of that check — the :class:`VerificationResult`
model and the :class:`VerificationStatus` enum of the four possible outcomes —
so that both the verifier and the CLI (task 7) share one typed contract.

The four statuses are:

* **INTACT_SIGNED** — the hash chain walks cleanly and every event's Ed25519
  signature validates under the public key for its ``key_id``.
* **INTACT_UNSIGNED** — the hash chain walks cleanly but some or all events
  carry no signature (no key was configured when they were written). Reported
  plainly so an operator is never left believing an unsigned chain was signed.
* **BROKEN_CHAIN** — a hash-link / sequence / malformed-line failure. Carries
  ``first_broken_seq`` and a human-readable ``reason``.
* **SIGNATURE_INVALID** — the hash chain is intact but a signed event's
  signature does not validate (tampered-then-rehashed content, wrong/unknown
  ``key_id``, or a forged signature). Carries ``first_bad_signature_seq`` and a
  ``reason``.

The ``intact`` and ``signed`` boolean flags are derived from the status by the
convenience constructors below so they can never drift out of sync with it:
``intact`` is True for INTACT_SIGNED / INTACT_UNSIGNED; ``signed`` is True only
for INTACT_SIGNED (every event validly signed). This module defines only the
model — the actual re-walk logic lives in :meth:`AuditChain.verify` (task 6.2).
"""

from __future__ import annotations

import argparse
from enum import Enum
from typing import Optional

from pydantic import BaseModel

__all__ = ["VerificationStatus", "VerificationResult"]


class VerificationStatus(str, Enum):
    """The four possible outcomes of re-walking the audit chain."""

    INTACT_SIGNED = "INTACT_SIGNED"
    INTACT_UNSIGNED = "INTACT_UNSIGNED"
    BROKEN_CHAIN = "BROKEN_CHAIN"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"


class VerificationResult(BaseModel):
    """The outcome of verifying an audit chain.

    ``intact`` means the hash chain walks cleanly; ``signed`` distinguishes a
    fully-signed chain from an intact-but-UNSIGNED one. Prefer the classmethod
    constructors (:meth:`intact_signed`, :meth:`intact_unsigned`,
    :meth:`broken_chain`, :meth:`signature_invalid`) so the ``intact``/``signed``
    flags stay consistent with ``status``.
    """

    status: VerificationStatus         # one of the four VerificationStatus values
    intact: bool                       # True for INTACT_SIGNED / INTACT_UNSIGNED
    signed: bool                       # True only when every event is validly signed
    total_events: int                  # number of events walked
    unsigned_events: int = 0           # count of events carrying no signature
    first_broken_seq: Optional[int] = None       # seq of first hash-chain break
    first_bad_signature_seq: Optional[int] = None  # seq of first invalid signature
    reason: Optional[str] = None       # human-readable explanation; None when fully OK

    # ------------------------------------------------------------------ #
    # Convenience constructors — keep intact/signed consistent with status
    # ------------------------------------------------------------------ #
    @classmethod
    def intact_signed(cls, total: int) -> "VerificationResult":
        """Hash chain walks cleanly and every event is validly signed.

        An empty chain (``total == 0``) is vacuously signed and uses this too.
        """
        return cls(
            status=VerificationStatus.INTACT_SIGNED,
            intact=True,
            signed=True,
            total_events=total,
            unsigned_events=0,
            first_broken_seq=None,
            first_bad_signature_seq=None,
            reason=None,
        )

    @classmethod
    def intact_unsigned(cls, total: int, unsigned: int) -> "VerificationResult":
        """Hash chain walks cleanly but some/all events carry no signature."""
        return cls(
            status=VerificationStatus.INTACT_UNSIGNED,
            intact=True,
            signed=False,
            total_events=total,
            unsigned_events=unsigned,
            first_broken_seq=None,
            first_bad_signature_seq=None,
            reason="chain intact but UNSIGNED",
        )

    @classmethod
    def broken_chain(
        cls, seq: int, reason: str, total: int
    ) -> "VerificationResult":
        """A hash-link / sequence / malformed-line failure at ``seq``."""
        return cls(
            status=VerificationStatus.BROKEN_CHAIN,
            intact=False,
            signed=False,
            total_events=total,
            unsigned_events=0,
            first_broken_seq=seq,
            first_bad_signature_seq=None,
            reason=reason,
        )

    @classmethod
    def signature_invalid(
        cls, seq: int, reason: str, total: int
    ) -> "VerificationResult":
        """Hash chain intact, but a signed event's signature does not validate."""
        return cls(
            status=VerificationStatus.SIGNATURE_INVALID,
            intact=True,
            signed=False,
            total_events=total,
            unsigned_events=0,
            first_broken_seq=None,
            first_bad_signature_seq=seq,
            reason=reason,
        )


# ====================================================================== #
# Verification CLI (task 7.1)
# ====================================================================== #
#
# Operator-facing entry point invoked as ``python -m fraud_detect.audit.verify``
# (Requirement 8.1). It verifies the chain and reports the VerificationResult
# with a strict exit-code contract so scripts and pipelines can branch on the
# outcome and, crucially, can tell an UNSIGNED chain apart from a signed one.
#
# Exit-code contract (Requirements 8.2, 8.3, 8.4):
#
#   0  INTACT_SIGNED      -- hash chain walks cleanly AND every event is validly
#                            signed. This is the ONLY success code; exit 0 is
#                            reserved for a fully signed, intact chain.
#   1  BROKEN_CHAIN /     -- a hash-link / sequence / malformed-line failure, or
#      SIGNATURE_INVALID     a signature that does not validate. Integrity failed.
#   2  INTACT_UNSIGNED    -- hash chain walks cleanly but some/all events carry
#                            no signature (no key was configured when written).
#                            Surfaced LOUDLY with its own distinct code so it is
#                            never mistaken for signed-OK (0) or broken (1).
#   3  ERROR              -- could not run verification at all (e.g. a signing key
#                            is configured but cannot be loaded -- fail-loud per
#                            Requirement 6.3/6.4; we never treat that as a pass or
#                            as a plain UNSIGNED result).
#
# ``main`` is kept importable and testable: it accepts an ``argv`` list and
# returns an int, so task 7.2 can assert each exit code without spawning a
# subprocess.

# Exit codes (module-level so tests and callers can reference them by name).
EXIT_INTACT_SIGNED = 0
EXIT_INTEGRITY_FAILED = 1   # BROKEN_CHAIN or SIGNATURE_INVALID
EXIT_INTACT_UNSIGNED = 2    # distinct, loud: unsigned chain is never "signed"
EXIT_ERROR = 3              # verification could not run (e.g. bad signing key)

_EPILOG = """\
exit codes:
  0  INTACT_SIGNED      chain intact and every event validly signed (the only OK)
  1  BROKEN_CHAIN /     chain broken or a signature did not validate
     SIGNATURE_INVALID
  2  INTACT_UNSIGNED    chain intact but UNSIGNED -- surfaced distinctly so it is
                        never mistaken for a signed chain
  3  ERROR              verification could not run (e.g. configured signing key
                        could not be loaded -- fail-loud, never a silent pass)
"""


def _format_result(result: "VerificationResult", path: str) -> str:
    """Render a human-readable multi-line summary of ``result`` for the CLI.

    Includes the status, the chain path, totals, and the relevant locating
    fields (``first_broken_seq`` / ``first_bad_signature_seq`` / ``reason``)
    only when they apply, so the output is not cluttered with ``None`` lines.
    """
    lines = [
        f"audit chain: {path}",
        f"status:      {result.status.value}",
        f"intact:      {result.intact}",
        f"signed:      {result.signed}",
        f"total_events:    {result.total_events}",
        f"unsigned_events: {result.unsigned_events}",
    ]
    if result.first_broken_seq is not None:
        lines.append(f"first_broken_seq:        {result.first_broken_seq}")
    if result.first_bad_signature_seq is not None:
        lines.append(f"first_bad_signature_seq: {result.first_bad_signature_seq}")
    if result.reason:
        lines.append(f"reason:      {result.reason}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    """CLI entry point: verify the audit chain and return the contract exit code.

    Mirrors the argparse style of ``statement_guard/cli.py`` (accepts an
    ``argv`` list, returns an ``int``). Builds the signer from the environment
    via :func:`build_signer_from_env` so a signed chain can be verified and an
    UNSIGNED one (no key configured) still produces a verifier with no key.

    Fail-loud on a bad key (Requirements 6.3/6.4): if ``build_signer_from_env``
    raises because a key is configured but cannot be loaded, we print the cause
    to stderr and return :data:`EXIT_ERROR` -- we do NOT treat it as a passing or
    an UNSIGNED result.
    """
    # Imported here (not at module top) to avoid a circular import: chain.py
    # imports VerificationResult from this module, and AuditChain pulls in the
    # signer. Importing inside main keeps the model import graph clean.
    import sys

    from .chain import AuditChain
    from .signer import build_signer_from_env

    parser = argparse.ArgumentParser(
        prog="python -m fraud_detect.audit.verify",
        description=(
            "Verify the append-only, hash-chained, signed audit trail and report "
            "whether it is intact and signed, intact but UNSIGNED, broken, or "
            "carrying an invalid signature."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        default="data/audit_chain.jsonl",
        help="path to the audit chain JSONL file (default: data/audit_chain.jsonl)",
    )
    args = parser.parse_args(argv)

    # Build the signer so signed chains can be verified. UNSIGNED env -> None.
    # A configured-but-unloadable key fails loudly here (ValueError); surface it
    # as an error and exit non-zero -- never a silent pass / UNSIGNED fallback.
    try:
        signer = build_signer_from_env()
    except ValueError as exc:
        print(f"! could not build signer: {exc}", file=sys.stderr)
        return EXIT_ERROR

    chain = AuditChain(args.path, signer=signer)
    result = chain.verify()

    summary = _format_result(result, args.path)
    # Compare by status VALUE, not enum identity. When this module is run via
    # ``python -m fraud_detect.audit.verify``, runpy executes it a second time as
    # ``__main__`` while the package __init__ has already imported it under its
    # real name, giving two distinct VerificationStatus class objects. An ``is``
    # identity check against the __main__ copy's enum would then never match the
    # result built by AuditChain.verify (which uses the package copy), silently
    # falling through to the wrong exit code. ``str``-enum value equality is
    # stable across both module instances.
    status = result.status.value
    if status == VerificationStatus.INTACT_SIGNED.value:
        print(summary)
        return EXIT_INTACT_SIGNED
    if status == VerificationStatus.INTACT_UNSIGNED.value:
        # Surface the UNSIGNED state distinctly and loudly (Requirement 8.4):
        # an operator must never be left believing an unsigned chain was signed.
        print(summary)
        print(
            "WARNING: chain is INTACT but UNSIGNED -- events carry no signature. "
            "This is NOT a signed-OK result.",
            file=sys.stderr,
        )
        return EXIT_INTACT_UNSIGNED
    # BROKEN_CHAIN or SIGNATURE_INVALID -> integrity failure (Requirement 8.3).
    print(summary, file=sys.stderr)
    return EXIT_INTEGRITY_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
