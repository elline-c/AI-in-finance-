# Implementation Plan: Immutable Audit Trail

## Overview

Build the signed, hash-chained, append-only audit trail as a new `fraud_detect/audit/`
subpackage, test-driven and bottom-up: data model + canonical serialization + hashing
first, then the `Signer`, then `AuditChain.append`, then verification (all four
outcomes), then the CLI, then `CaseStore` integration, then the honest
security-boundary docs. Each step builds on the previous and ends by wiring the new
primitive into the existing `CaseStore`. Implementation language is Python 3
(pydantic v2 + stdlib + the new `cryptography` dependency), matching the design and the
existing codebase. Property-based tests use Hypothesis; unit/integration tests use
pytest. All tests live in `tests/`.

Tasks marked with `*` are optional test sub-tasks and may be skipped for a faster MVP.
Scope is strictly the prototype: migration backfill, absorbing `access_audit.jsonl`,
cross-process concurrency, and production key-rotation operations are out of scope.

## Tasks

- [x] 1. Scaffold the `fraud_detect/audit/` subpackage and add the crypto dependency
  - Create `fraud_detect/audit/__init__.py` exporting the package's public names
    (filled in as later tasks land).
  - Add `cryptography>=43.0.0` to `requirements.txt` as the feature's first non-stdlib
    runtime dependency (everything else stays stdlib + pydantic v2).
  - _Requirements: 1.2_

- [x] 2. Canonical serialization, hashing, and the `AuditEvent` model
  - [x] 2.1 Implement canonical serialization + hash/signed payload helpers
    - In `fraud_detect/audit/canonical.py`, add `GENESIS_HASH = "0" * 64`, a
      `canonical_bytes(payload: dict) -> bytes` helper using
      `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`,
      a `hash_payload(event)` builder (`seq, prev_hash, ts, case_id, event, detail`),
      a `signed_payload(event)` builder (hash payload + `entry_hash`, excluding
      `signature`/`key_id`), and `compute_entry_hash(hash_payload) -> str`
      (hex SHA-256 over the canonical hash-payload bytes).
    - _Requirements: 2.3, 2.4_
  - [x] 2.2 Implement the `AuditEvent` pydantic v2 model
    - In `fraud_detect/audit/models.py`, define `AuditEvent` with chain fields
      (`seq: int`, `prev_hash: str`, `entry_hash: str`), payload fields
      (`ts: str`, `case_id: str`, `event: str`, `detail: str`), and signature fields
      (`signature: str | None = None`, `key_id: str | None = None`), with validators
      enforcing non-negative `seq` and 64-lowercase-hex `prev_hash`/`entry_hash`.
    - _Requirements: 2.5, 3.3, 4.4_
  - [x]* 2.3 Property test — deterministic canonical hashing
    - **Property 3: Deterministic canonical hashing** — for any payload, canonical
      serialization and the recomputed `entry_hash` are identical across repeated
      computations and equal the stored hash.
    - **Validates: Requirements 2.3, 2.4**
  - [x]* 2.4 Property test — field-format invariant on `AuditEvent`
    - **Property 4: Field-format invariant** — every event has a non-negative integer
      `seq` and 64-lowercase-hex `prev_hash`/`entry_hash`.
    - **Validates: Requirements 2.5**

- [x] 3. Signing backend: `Signer` interface + `LocalFileSigner` + env builder
  - [x] 3.1 Define the `Signer` protocol and `LocalFileSigner`
    - In `fraud_detect/audit/signer.py`, define the `Signer` Protocol
      (`key_id` property, `sign(data) -> bytes`, `verify(data, signature, *, key_id) -> bool`,
      `public_key_for(key_id) -> Ed25519PublicKey | None`). Implement `LocalFileSigner`
      loading an Ed25519 private key from a file path, deriving a deterministic `key_id`
      (fingerprint of the public key, e.g. `ed25519:<short-hash>`), and resolving its
      matching public key.
    - _Requirements: 5.1, 5.2, 5.4_
  - [x] 3.2 Implement `build_signer_from_env` with fail-loud key handling
    - In `fraud_detect/audit/signer.py`, add `build_signer_from_env()` that returns a
      `LocalFileSigner` when `AUDIT_SIGNING_KEY_PATH` is set and `None` when it is unset.
      When the var is set but the file is missing, unreadable, or not a valid Ed25519
      private key, raise a clear error identifying the cause (no silent UNSIGNED
      fallback). Document that the key path must live outside `data/`.
    - _Requirements: 5.3, 6.3, 6.4_
  - [x]* 3.3 Property test — signing round-trip and key resolution
    - **Property 6: Signatures resolve by key_id and wrong keys fail** — for any payload
      and any generated keypair, a signature from key A validates under A and fails under
      a different key B; a mismatched/unknown `key_id` yields a verification failure.
    - **Validates: Requirements 4.4, 5.4**
  - [x]* 3.4 Unit tests — fail-loud construction and local-file semantics
    - Key configured but missing/invalid/unparseable → construction raises (not silent
      UNSIGNED). Unset env → `build_signer_from_env()` returns `None`. `key_id` is
      derived from the public key and stable across loads.
    - _Requirements: 5.2, 5.3, 6.3, 6.4_

- [x] 4. `AuditChain.append` — append-only, hash-chained, signed, durable
  - [x] 4.1 Implement `AuditChain` construction and read helpers
    - In `fraud_detect/audit/chain.py`, implement `AuditChain(path="data/audit_chain.jsonl", signer=None)`,
      creating the parent dir, plus `tail()`, `events_for(case_id)`, and an internal
      line reader/parser that yields `AuditEvent`s in file order.
    - _Requirements: 1.4_
  - [x] 4.2 Implement `append` with chaining, signing, fsync, and lock
    - Compute `seq`/`prev_hash` from the tail (`seq 0` + `GENESIS_HASH` when empty),
      compute `entry_hash` over the hash payload, sign the signed payload when a signer
      is present (store base64 `signature` + `key_id`) or write `null`/`null` when
      UNSIGNED, then append one newline-delimited JSON line and flush+fsync, all under a
      `threading.Lock`. Never rewrite or truncate existing lines. Return the written
      `AuditEvent`.
    - _Requirements: 1.1, 2.1, 2.2, 4.1, 4.2, 4.3, 6.1, 10.1, 10.2_
  - [x]* 4.3 Property test — append preserves prior content as an exact prefix
    - **Property 1: Append preserves the existing chain as a prefix** — appending leaves
      every prior line byte-identical and adds exactly one line.
    - **Validates: Requirements 1.1**
  - [x]* 4.4 Property test — genesis and chaining invariant
    - **Property 2: Genesis and chaining invariant** — first event has `seq == 0` and
      `prev_hash == GENESIS_HASH`; each later event increments `seq` by 1 and links
      `prev_hash` to the prior `entry_hash`.
    - **Validates: Requirements 2.1, 2.2**
  - [x]* 4.5 Property test — concurrent in-process appends serialize cleanly
    - **Property 12: Concurrent in-process appends serialize to a contiguous chain** —
      for N concurrent appends, `seq` 0..N-1 each appear exactly once and the chain
      verifies as intact.
    - **Validates: Requirements 10.2**
  - [x]* 4.6 Property test — every event carries its case_id
    - **Property 13: Every event carries its case_id** — every recorded event has a
      non-empty `case_id` equal to the case it was recorded for.
    - **Validates: Requirements 3.3**

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. `VerificationResult` model and the verifier
  - [x] 6.1 Implement the `VerificationResult` model
    - In `fraud_detect/audit/verify.py`, define the pydantic v2 `VerificationResult`
      (`status`, `intact`, `signed`, `total_events`, `unsigned_events`,
      `first_broken_seq`, `first_bad_signature_seq`, `reason`) covering the four
      statuses `INTACT_SIGNED | INTACT_UNSIGNED | BROKEN_CHAIN | SIGNATURE_INVALID`.
    - _Requirements: 7.2, 7.3, 7.4, 7.5_
  - [x] 6.2 Implement `AuditChain.verify` (two-pass re-walk)
    - Add `AuditChain.verify()` delegating to a verifier that (pass 1) re-walks from
      genesis checking `seq` increments by 1, `prev_hash` equals the prior `entry_hash`,
      and the recomputed `entry_hash` matches — returning `BROKEN_CHAIN` with
      `first_broken_seq`+`reason` on any failure or malformed line; (pass 2) verifies
      each signed event's signature via the signer's public key for its `key_id`,
      returning `SIGNATURE_INVALID` with `first_bad_signature_seq`+`reason` on a bad or
      unresolvable signature; counts unsigned events and returns `INTACT_UNSIGNED` when
      any are unsigned, `INTACT_SIGNED` otherwise. An empty chain returns `INTACT_SIGNED`
      with `total_events == 0`. Verification must not modify the chain file.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 9.1, 9.3_
  - [x]* 6.3 Property test — signed chain round-trip verifies as INTACT_SIGNED
    - **Property 5: Signed chain round-trip verifies as INTACT_SIGNED** — a chain built
      with a configured signer verifies as `INTACT_SIGNED` with `total_events` equal to
      the number appended and every event carrying a non-null `signature`/`key_id`.
    - **Validates: Requirements 4.1, 4.3, 7.2**
  - [x]* 6.4 Property test — unsigned chains reported as UNSIGNED, never signed
    - **Property 7: Unsigned chains are reported as UNSIGNED, never signed** — a chain
      written with no signer verifies as `INTACT_UNSIGNED`, `signed == False`, with
      `unsigned_events` equal to the number of unsigned events and hash links intact.
    - **Validates: Requirements 6.1, 6.2, 7.3**
  - [x]* 6.5 Property test — any single structural corruption is BROKEN_CHAIN
    - **Property 8: Any single structural corruption is caught as BROKEN_CHAIN** —
      editing one field, deleting a line, reordering two lines, inserting a line, or a
      malformed line yields `BROKEN_CHAIN` with `first_broken_seq` at the first affected
      position.
    - **Validates: Requirements 7.4, 9.3**
  - [x]* 6.6 Property test — signature tampering is SIGNATURE_INVALID
    - **Property 9: Signature tampering is caught as SIGNATURE_INVALID** — invalidating
      one event's `signature` or changing its `key_id` to an unresolvable value (hash
      chain intact) yields `SIGNATURE_INVALID` at that event's `seq`.
    - **Validates: Requirements 7.5**
  - [x]* 6.7 Property test — rewrite-with-rehash forgery still caught
    - **Property 10: Rewrite-with-rehash forgery is still caught** — editing one event's
      content and recomputing its and all downstream `entry_hash` values still yields
      `SIGNATURE_INVALID` because the attacker cannot re-sign.
    - **Validates: Requirements 7.7**
  - [x]* 6.8 Property test — verification has no side effects
    - **Property 11: Verification has no side effects on the chain** — for any chain,
      intact or corrupted, verification leaves the file bytes unchanged.
    - **Validates: Requirements 9.1**
  - [x]* 6.9 Unit test — empty chain verifies as INTACT_SIGNED
    - Empty/absent chain file verifies as `INTACT_SIGNED` with `total_events == 0`.
    - _Requirements: 7.6_

- [x] 7. Verification CLI with exit-code contract
  - [x] 7.1 Implement `python -m fraud_detect.audit.verify`
    - Add a `main(argv)` / `__main__` entry in `fraud_detect/audit/verify.py`
      (argparse, mirroring `statement_guard/cli.py` style) accepting
      `--path data/audit_chain.jsonl`, building the signer from env, running `verify`,
      printing the result, exiting 0 on `INTACT_SIGNED`, non-zero on `BROKEN_CHAIN`/
      `SIGNATURE_INVALID`, and surfacing `INTACT_UNSIGNED` distinctly (distinct message
      and exit code) so it is never presented as signed.
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x]* 7.2 Unit tests — CLI exit codes for each outcome
    - Intact signed fixture → exit 0; tampered/broken fixture and bad-signature fixture
      → non-zero; unsigned fixture → distinct non-"signed" messaging/exit.
    - _Requirements: 8.2, 8.3, 8.4_

- [x] 8. Integrate `AuditChain` into `CaseStore`
  - [x] 8.1 Construct the chain + signer in `CaseStore.__init__`
    - Update `fraud_detect/store.py`: add an `audit_path="data/audit_chain.jsonl"`
      parameter, build a signer via `build_signer_from_env()` (None → UNSIGNED;
      fail-loud if a key is configured but unreadable), and hold an `AuditChain`. Keep
      the existing `cases.json` snapshot write unchanged. Preserve the public signatures
      of `upsert_report` and `record_decision`.
    - _Requirements: 1.3, 3.4_
  - [x] 8.2 Append `SCREENED` / `HUMAN_*` events from the existing call sites
    - In `upsert_report`, append a `SCREENED` event; in `record_decision`, append
      `HUMAN_APPROVED`/`HUMAN_REJECTED` — each using the same `ts` and `detail` string the
      current `audit_log` records, and an explicit `case_id`. Keep `Case.audit_log` as a
      derived/read-only mirror for UI convenience (start-fresh migration stance; no
      backfill).
    - _Requirements: 3.1, 3.2, 3.3_
  - [x]* 8.3 Integration test — end-to-end screen + decision chain verifies
    - Run a screening then a human decision through `CaseStore`; assert the chain
      contains the two expected events (matching `event`/`detail`) and `verify` reports
      `INTACT_SIGNED` (with a signer) or `INTACT_UNSIGNED` (without).
    - _Requirements: 3.1, 3.2, 9.2_

- [x] 9. Honest security-boundary documentation (R11)
  - Update the README/docs (e.g. `FRAUD_DETECT_README.md`) to state plainly that the
    prototype is tamper-evident and forgery-resistant while the signing key is safe, but
    is not proof against signing-key theft and not proof against a wipe-and-restart;
    that production MUST move key custody to a KMS/HSM so the private key never leaves
    the security boundary; and that detecting a wipe-and-restart requires external
    anchoring/checkpointing (a production requirement, out of scope for the prototype).
    Also note the never-auto-reject, never-auto-repair governance stance for integrity
    failures.
  - _Requirements: 9.1, 9.2, 11.1, 11.2, 11.3_

- [x] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP.
- Each task references the specific requirement clauses it implements; property-test
  sub-tasks also name the Correctness Property they validate.
- Checkpoints (tasks 5 and 10) provide incremental validation points.
- Property tests run a minimum of 100 iterations and are tagged
  `Feature: immutable-audit-trail, Property {number}: {property_text}`.
- Scope is the prototype only. Deferred/out-of-scope items — migration backfill of the
  existing `cases.json` audit_log, absorbing `access_audit.jsonl`, cross-process
  concurrency (flock / dedicated writer), and production key-rotation/revocation
  operations — are intentionally NOT implementation tasks (see requirements.md
  "Assumptions and Deferred Scope").

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3", "3.4"] },
    { "id": 5, "tasks": ["4.1"] },
    { "id": 6, "tasks": ["4.2"] },
    { "id": 7, "tasks": ["4.3", "4.4", "4.5", "4.6"] },
    { "id": 8, "tasks": ["6.1"] },
    { "id": 9, "tasks": ["6.2"] },
    { "id": 10, "tasks": ["6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9"] },
    { "id": 11, "tasks": ["7.1"] },
    { "id": 12, "tasks": ["7.2"] },
    { "id": 13, "tasks": ["8.1"] },
    { "id": 14, "tasks": ["8.2"] },
    { "id": 15, "tasks": ["8.3", "9"] }
  ]
}
```
