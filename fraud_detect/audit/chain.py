"""The append-only, hash-chained ``AuditChain`` store.

``AuditChain`` owns the dedicated append-only JSONL file
(``data/audit_chain.jsonl``) that holds the immutable audit events. The mutable
case snapshot stays in ``data/cases.json`` and is not touched here. The file is
never rewritten or truncated in place: appends add exactly one newline-delimited
JSON object and nothing else is modified.

This module (task 4.1) implements construction and the **read** helpers:

* :meth:`AuditChain.tail` -- the last event in the chain, or ``None`` when the
  chain is empty or absent.
* :meth:`AuditChain.events_for` -- the events for a single ``case_id`` in chain
  order.

The **write** path (:meth:`append`, task 4.2) and the **verification** path
(:meth:`verify`, task 6.2) are added by later tasks. A ``threading.Lock`` is
created here so the append path can serialize writers (single-writer
discipline, Requirement 10.2) once it lands.

**How malformed lines are handled in the read helpers.** The private
:meth:`_read_all_events` parser RAISES :class:`MalformedChainLine` on the first
line that is not valid JSON / not a valid :class:`AuditEvent`, rather than
silently skipping it. ``tail`` and ``events_for`` are built on top of it and so
also surface corruption instead of hiding it -- silently dropping a bad line
here would make a later verification pass miss a real tamper/corruption event
(a malformed line is a chain break per the design, Requirement 7.4 / 9.3). The
verifier (task 6.2) intentionally does its own raw, line-by-line parse so it can
classify a malformed line as ``BROKEN_CHAIN`` at the right ``seq`` position; it
does not rely on these read helpers. A missing file is NOT corruption -- it is a
valid empty chain, so the parser yields nothing and the helpers return
``None`` / ``[]``.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from pydantic import ValidationError

from . import canonical
from .models import AuditEvent
from .signer import Signer
from .verify import VerificationResult, VerificationStatus

__all__ = ["AuditChain", "MalformedChainLine"]


class MalformedChainLine(ValueError):
    """A non-empty line of the chain file could not be parsed as an ``AuditEvent``.

    Raised by the read helpers (:meth:`AuditChain._read_all_events` and its
    callers) so that corruption is surfaced rather than silently skipped. The
    1-based ``line_number`` locates the offending line in the file.
    """

    def __init__(self, line_number: int, reason: str) -> None:
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"malformed audit chain line {line_number}: {reason}")


class AuditChain:
    """Append-only, hash-chained store of :class:`AuditEvent` records.

    Construction creates the parent directory of ``path`` but deliberately does
    NOT create the file itself: an absent or empty chain is a valid empty chain,
    and the file is created lazily by the first :meth:`append` (task 4.2). The
    optional ``signer`` is held for the append path; when it is ``None`` events
    are written UNSIGNED.
    """

    def __init__(
        self,
        path: str = "data/audit_chain.jsonl",
        signer: Optional[Signer] = None,
    ) -> None:
        """Prepare the chain at ``path`` with an optional ``signer``.

        Creates the parent directory (``mkdir(parents=True, exist_ok=True)``)
        but not the chain file -- an empty/absent chain is valid. ``signer``, if
        given, signs each appended event (task 4.2); ``None`` means UNSIGNED.
        """
        self.path = Path(path)
        # Ensure the parent directory exists, but never create the file itself:
        # an absent file is a valid empty chain.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.signer = signer

        # Serializes concurrent in-process appends so seq/prev_hash stay
        # consistent (single-writer discipline, Requirement 10.2). Used by the
        # append path (task 4.2); created here so it exists for the whole life
        # of the chain object.
        self._append_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Low-level line readers
    # ------------------------------------------------------------------
    def _iter_raw_lines(self) -> Iterator[str]:
        """Yield the raw, newline-stripped lines of the chain file, in file order.

        Tolerates a missing file (yields nothing). Blank/whitespace-only lines
        are skipped -- they carry no event. Lines are yielded verbatim otherwise
        (no JSON parsing), so a caller like the verifier can classify a malformed
        line itself.
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                yield stripped

    def _iter_events(self) -> Iterator[AuditEvent]:
        """Parse each non-empty line into an :class:`AuditEvent`, in file order.

        Raises :class:`MalformedChainLine` on the first line that is not valid
        JSON or not a valid ``AuditEvent``. Tolerates a missing file (yields
        nothing).
        """
        for line_number, raw in enumerate(self._iter_raw_lines(), start=1):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise MalformedChainLine(line_number, f"invalid JSON: {exc}") from exc
            try:
                yield AuditEvent.model_validate(obj)
            except ValidationError as exc:
                raise MalformedChainLine(
                    line_number, f"not a valid AuditEvent: {exc}"
                ) from exc

    def _read_all_events(self) -> List[AuditEvent]:
        """Read and parse every event in the chain, in file order.

        Returns an empty list for a missing or empty file. RAISES
        :class:`MalformedChainLine` on the first unparseable line rather than
        silently skipping it, so callers never mistake a corrupted chain for a
        shorter valid one. The read helpers (:meth:`tail`, :meth:`events_for`)
        build on this; the verifier (task 6.2) does its own raw parse instead so
        it can report a malformed line as a ``BROKEN_CHAIN`` at the right ``seq``.
        """
        return list(self._iter_events())

    # ------------------------------------------------------------------
    # Public read helpers
    # ------------------------------------------------------------------
    def tail(self) -> Optional[AuditEvent]:
        """Return the last event in the chain, or ``None`` if it is empty/absent.

        Raises :class:`MalformedChainLine` if a line cannot be parsed.
        """
        events = self._read_all_events()
        if not events:
            return None
        return events[-1]

    def events_for(self, case_id: str) -> List[AuditEvent]:
        """Return the events for ``case_id``, in chain order.

        Returns an empty list when the chain is empty/absent or no event matches.
        Raises :class:`MalformedChainLine` if a line cannot be parsed.
        """
        return [event for event in self._read_all_events() if event.case_id == case_id]

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def append(
        self,
        *,
        case_id: str,
        event: str,
        detail: str,
        ts: Optional[str] = None,
    ) -> AuditEvent:
        """Append one event to the chain, linked to the current tail, and return it.

        The whole read-tail -> compute -> write sequence runs under
        ``self._append_lock`` so concurrent in-process appends cannot interleave
        and corrupt the ``seq``/``prev_hash`` linkage (single-writer discipline,
        Requirement 10.2).

        The chain structure is computed exactly as the hashing primitives dictate
        (Requirement 2.1/2.2): the hash payload is ``seq, prev_hash, ts, case_id,
        event, detail`` and ``entry_hash`` is the SHA-256 over its canonical
        bytes. Signing (Requirement 4.x), when a ``signer`` is configured, is
        computed over the hash payload PLUS ``entry_hash`` and recorded as a
        base64 string alongside the signer's ``key_id``; with no signer the event
        is written UNSIGNED (``signature``/``key_id`` are ``None``).

        The line is appended (never rewrites/truncates prior content), then
        flushed and ``os.fsync``-ed so a committed event survives a crash
        (durability, Requirement 10.1).

        :param ts: Optional UTC ISO-8601 timestamp. When omitted, one is
            generated with ``datetime.utcnow().isoformat()`` to match the
            existing convention in ``store.py`` (the project has a known
            ``utcnow`` deprecation; we follow the existing format for
            consistency rather than introduce a different one).
        """
        with self._append_lock:
            if ts is None:
                ts = datetime.utcnow().isoformat()

            tail = self.tail()
            if tail is None:
                seq = 0
                prev_hash = canonical.GENESIS_HASH
            else:
                seq = tail.seq + 1
                prev_hash = tail.entry_hash

            # 1. Hash the payload (chain structure is independent of signing).
            hash_payload = canonical.hash_payload(
                {
                    "seq": seq,
                    "prev_hash": prev_hash,
                    "ts": ts,
                    "case_id": case_id,
                    "event": event,
                    "detail": detail,
                }
            )
            entry_hash = canonical.compute_entry_hash(hash_payload)

            # 2. Sign the payload INCLUDING entry_hash, if a signer is configured.
            if self.signer is not None:
                signed = canonical.signed_payload({**hash_payload, "entry_hash": entry_hash})
                signature = base64.b64encode(
                    self.signer.sign(canonical.canonical_bytes(signed))
                ).decode("ascii")
                key_id: Optional[str] = self.signer.key_id
            else:
                signature = None
                key_id = None

            audit_event = AuditEvent(
                seq=seq,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
                ts=ts,
                case_id=case_id,
                event=event,
                detail=detail,
                signature=signature,
                key_id=key_id,
            )

            # 3. Append exactly one newline-delimited JSON object; never rewrite
            #    or truncate existing content. Flush + fsync for durability.
            line = audit_event.model_dump_json() + "\n"
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

            return audit_event

    # ------------------------------------------------------------------
    # Verification path
    # ------------------------------------------------------------------
    def verify(self) -> VerificationResult:
        """Re-walk the append-only chain and report its integrity (read-only).

        Implements the two-pass ``verifyChain`` algorithm from the design. The
        chain file is opened **read-only** and is never modified, truncated, or
        auto-repaired (Requirement 9.1): a broken or invalid chain is surfaced as
        a result for a human to act on, never silently fixed.

        **Pass 0 — raw read.** Reads the raw non-empty lines in file order and
        parses each into an :class:`AuditEvent`. Unlike :meth:`_read_all_events`
        (which raises on the first bad line), this pass CATCHES a malformed line
        and reports it as ``BROKEN_CHAIN`` at its 0-based position -- which is the
        ``seq`` that *should* have been at that spot, since a malformed line has
        no trustworthy ``seq`` of its own. A partial/malformed trailing line left
        by a crash mid-append is therefore reported as a break at the tail and is
        **not** auto-truncated (Requirement 9.3).

        **Pass 1 — hash chain.** Walks the parsed events with
        ``expected_prev = GENESIS_HASH`` and ``expected_seq = 0``, checking the
        ``seq`` increment, that ``prev_hash`` matches the prior ``entry_hash``,
        and that the recomputed ``entry_hash`` matches the stored one
        (Requirement 7.1). Any mismatch is ``BROKEN_CHAIN`` at that ``seq``
        (Requirement 7.4). Signatures are only trusted once the hash chain is
        fully intact.

        **Pass 2 — signatures.** Walks the (now hash-sound) events again, counting
        UNSIGNED events. For each SIGNED event it resolves the public key for the
        event's ``key_id`` via ``self.signer`` and verifies the Ed25519 signature
        over the canonical signed bytes (hash payload PLUS ``entry_hash``). An
        unknown ``key_id``, no configured signer, or a signature that does not
        validate is ``SIGNATURE_INVALID`` at that ``seq`` (Requirements 7.5, 7.7).
        A signed chain with no configured verifier cannot be trusted as signed, so
        it is reported ``SIGNATURE_INVALID`` rather than silently accepted. After
        the pass, a chain with any UNSIGNED events is ``INTACT_UNSIGNED`` with the
        count (Requirement 7.3); otherwise ``INTACT_SIGNED`` (Requirement 7.2).

        An empty or absent chain is vacuously ``INTACT_SIGNED`` with
        ``total_events == 0`` (Requirement 7.6).
        """
        # --- Pass 0: raw read, catching malformed lines ---------------------
        events: List[AuditEvent] = []
        for position, raw in enumerate(self._iter_raw_lines()):
            try:
                obj = json.loads(raw)
                event = AuditEvent.model_validate(obj)
            except (json.JSONDecodeError, ValidationError) as exc:
                # A malformed line has no trustworthy seq of its own; report the
                # 0-based position (the seq that should have been here) and the
                # count of events successfully parsed before it.
                return VerificationResult.broken_chain(
                    seq=position,
                    reason=f"malformed line at position {position}: {exc}",
                    total=len(events),
                )
            events.append(event)

        total = len(events)
        if total == 0:
            # Empty/absent chain is vacuously signed (Requirement 7.6).
            return VerificationResult.intact_signed(total=0)

        # --- Pass 1: hash chain must walk cleanly ---------------------------
        expected_prev = canonical.GENESIS_HASH
        expected_seq = 0
        for position, event in enumerate(events):
            if event.seq != expected_seq:
                return VerificationResult.broken_chain(
                    seq=expected_seq,
                    reason=(
                        f"sequence gap/reorder at position {position}: "
                        f"expected seq {expected_seq}, found {event.seq}"
                    ),
                    total=total,
                )
            if event.prev_hash != expected_prev:
                return VerificationResult.broken_chain(
                    seq=event.seq,
                    reason="prev_hash does not match prior entry_hash",
                    total=total,
                )
            recomputed = canonical.compute_entry_hash(canonical.hash_payload(event))
            if recomputed != event.entry_hash:
                return VerificationResult.broken_chain(
                    seq=event.seq,
                    reason="entry_hash mismatch (content altered)",
                    total=total,
                )
            expected_prev = event.entry_hash
            expected_seq += 1

        # --- Pass 2: signatures (hash chain is intact) ----------------------
        unsigned_count = 0
        for event in events:
            if event.signature is None:
                unsigned_count += 1
                continue

            # A signed event claims authenticity; without a verifier/key we
            # cannot trust that claim, so treat it as invalid (design Scenario 8).
            if self.signer is None:
                return VerificationResult.signature_invalid(
                    seq=event.seq,
                    reason="no signer configured to verify signature",
                    total=total,
                )
            public_key = self.signer.public_key_for(event.key_id)
            if public_key is None:
                return VerificationResult.signature_invalid(
                    seq=event.seq,
                    reason=f"unknown key_id: {event.key_id}",
                    total=total,
                )

            signed_bytes = canonical.canonical_bytes(canonical.signed_payload(event))
            signature_bytes = base64.b64decode(event.signature)
            if not self.signer.verify(
                signed_bytes, signature_bytes, key_id=event.key_id
            ):
                return VerificationResult.signature_invalid(
                    seq=event.seq,
                    reason="signature does not validate",
                    total=total,
                )

        if unsigned_count > 0:
            return VerificationResult.intact_unsigned(
                total=total, unsigned=unsigned_count
            )
        return VerificationResult.intact_signed(total=total)
