# Requirements Document

## Introduction

The `fraud_detect` AML/fraud-screening system currently records its audit trail as
a plain `audit_log` list embedded inside each `Case` object in `data/cases.json`,
a file that is fully rewritten on every change. That makes the audit record as
editable as any other field: a past `SCREENED`, `HUMAN_APPROVED`, or
`HUMAN_REJECTED` event can be edited, deleted, or reordered with no trace. For a
regulated AML/fraud product (PDPA + MAS TRM governance regime), audit integrity is
table-stakes.

This feature makes the audit trail **append-only**, **hash-chained**, and
**digitally signed**. Audit events move to a dedicated append-only JSONL store
(`data/audit_chain.jsonl`) that is never rewritten in place, while the mutable case
snapshot stays in `cases.json`. Each event is linked to its predecessor by a hash
chain so edits, deletions, reorders, and insertions break the chain and are
detectable. Each event is additionally signed with Ed25519 over a payload that
commits to the chained `entry_hash`, so a full-file rewrite with recomputed hashes
cannot forge a verifiable chain without the private key. A verification routine and
CLI re-walk the chain, check signatures, and report one of four outcomes. Signing
is pluggable behind a `Signer` interface so a production KMS/HSM backend can replace
the prototype's local-file key. The system stays fail-safe and never auto-rejects:
a broken, invalid, missing-signature, or unexpectedly-unsigned chain is surfaced to
a human as an incident and never auto-repaired.

The requirements below are derived from the approved design and are scoped to the
prototype. Production hardening (KMS/HSM key custody and external anchoring) and
several design open questions are captured as constraints, assumptions, and
out-of-scope notes rather than prototype acceptance criteria.

## Glossary

- **Audit_Chain**: The append-only, hash-chained, optionally-signed store of audit
  events, persisted to `data/audit_chain.jsonl`. Owns appending and verification.
- **Audit_Event**: The immutable unit written to the Audit_Chain; one JSON object
  per line, carrying chain fields (`seq`, `prev_hash`, `entry_hash`), payload fields
  (`ts`, `case_id`, `event`, `detail`), and signature fields (`signature`, `key_id`).
- **Case_Store**: The existing component (`store.py`) that persists the mutable case
  snapshot to `data/cases.json` and routes audit events to the Audit_Chain.
- **Signer**: The pluggable signing backend interface (sign, verify, resolve public
  key by `key_id`). Prototype implementation is `Local_File_Signer`; production
  target is a KMS/HSM-backed implementation.
- **Local_File_Signer**: Prototype Signer that loads an Ed25519 private key from a
  file whose path comes from an environment variable, located outside the data
  directory.
- **Verifier**: The verification capability (function + CLI) that re-walks the chain
  and checks signatures, producing a Verification_Result.
- **Verification_Result**: The structured outcome of verification: one of
  `INTACT_SIGNED`, `INTACT_UNSIGNED`, `BROKEN_CHAIN`, or `SIGNATURE_INVALID`, with
  supporting fields.
- **entry_hash**: The hex SHA-256 over the canonical hash payload
  (`seq`, `prev_hash`, `ts`, `case_id`, `event`, `detail`) of an event.
- **prev_hash**: The `entry_hash` of the preceding event; `GENESIS_HASH` (64 zeros)
  for the first event (`seq == 0`).
- **signed payload**: The canonical bytes the Ed25519 signature is computed over —
  the hash payload plus `entry_hash`, excluding `signature` and `key_id`.
- **UNSIGNED**: An event with `signature == null` and `key_id == null`, written when
  no signing key was configured.
- **GENESIS_HASH**: The constant `"0" * 64`, used as `prev_hash` for `seq == 0`.
- **System**: The `fraud_detect` audit-trail subsystem comprising the above
  components, unless a more specific component is named.

## Requirements

### Requirement 1: Append-only audit event store separated from the mutable snapshot

**User Story:** As a compliance operator, I want audit events written to a dedicated
append-only store that is never rewritten in place, so that the history of what the
system saw and decided cannot be silently edited the way a rewritten file can.

#### Acceptance Criteria

1. WHEN an audit event is recorded, THE Audit_Chain SHALL append the event as one
   newline-delimited JSON object to `data/audit_chain.jsonl` without rewriting or
   truncating any existing line.
2. THE Audit_Chain SHALL persist audit events to a store separate from the mutable
   case snapshot in `data/cases.json`.
3. WHEN a case snapshot changes, THE Case_Store SHALL continue to persist the
   mutable case state to `data/cases.json`.
4. WHEN the audit chain file does not yet exist at append time, THE Audit_Chain
   SHALL create it and write the first event as the genesis event.

### Requirement 2: Hash-chained events for tamper evidence

**User Story:** As a compliance operator, I want each audit event cryptographically
linked to its predecessor, so that any edit, deletion, reorder, or insertion of a
past event breaks the chain and is detectable.

#### Acceptance Criteria

1. WHEN the first event is appended to an empty chain, THE Audit_Chain SHALL assign
   it `seq == 0` and `prev_hash == GENESIS_HASH`.
2. WHEN an event is appended to a non-empty chain, THE Audit_Chain SHALL assign it a
   `seq` equal to the previous event's `seq` plus 1 and a `prev_hash` equal to the
   previous event's `entry_hash`.
3. WHEN an event is appended, THE Audit_Chain SHALL compute `entry_hash` as the hex
   SHA-256 over the canonical hash payload containing `seq`, `prev_hash`, `ts`,
   `case_id`, `event`, and `detail`, excluding `entry_hash` itself.
4. THE Audit_Chain SHALL serialize the hash payload canonically (sorted keys,
   compact separators, UTF-8) so that `entry_hash` is reproducible across repeated
   computations and across process boundaries.
5. THE Audit_Event SHALL store `seq` as a non-negative integer, `prev_hash` as 64
   lowercase hex characters, and `entry_hash` as 64 lowercase hex characters.

### Requirement 3: Preserve existing audit event content and call sites

**User Story:** As a maintainer, I want the existing `SCREENED`, `HUMAN_APPROVED`,
and `HUMAN_REJECTED` events recorded with the same information and the same
`CaseStore` call signatures, so that integrating the chain does not change observed
behavior or break existing callers.

#### Acceptance Criteria

1. WHEN `CaseStore.upsert_report` is called, THE Case_Store SHALL append a
   `SCREENED` audit event to the Audit_Chain with the same `ts` and `detail` content
   recorded by the current `audit_log`.
2. WHEN `CaseStore.record_decision` is called, THE Case_Store SHALL append a
   `HUMAN_APPROVED` or `HUMAN_REJECTED` audit event to the Audit_Chain with the same
   `ts` and `detail` content recorded by the current `audit_log`.
3. THE Audit_Event SHALL include an explicit `case_id` field identifying the case
   the event belongs to.
4. THE Case_Store SHALL preserve the existing public signatures of `upsert_report`
   and `record_decision`.

### Requirement 4: Per-event Ed25519 signing committing to the chained hash

**User Story:** As a compliance operator, I want each event signed over a payload
that includes its `entry_hash`, so that an attacker who rewrites the file and
recomputes every hash still cannot forge a verifiable chain without the private key.

#### Acceptance Criteria

1. WHERE a Signer is configured, THE Audit_Chain SHALL sign each appended event with
   Ed25519 over the canonical signed payload containing `seq`, `prev_hash`, `ts`,
   `case_id`, `event`, `detail`, and `entry_hash`.
2. WHEN computing the signed payload, THE Audit_Chain SHALL include `entry_hash` and
   SHALL exclude the `signature` and `key_id` fields.
3. WHERE a Signer is configured, THE Audit_Chain SHALL store the base64-encoded
   signature in the event's `signature` field and the signer's identifier in the
   event's `key_id` field.
4. THE Audit_Chain SHALL store `key_id` per event so that verification of an event
   resolves its public key by that event's `key_id` independently of other events.

### Requirement 5: Pluggable signing backend

**User Story:** As a maintainer, I want signing encapsulated behind a `Signer`
interface, so that the prototype's local-file key can be swapped for a production
KMS/HSM backend without changing the Audit_Chain or the event model.

#### Acceptance Criteria

1. THE System SHALL expose a Signer interface providing a stable `key_id`, a `sign`
   operation, a signature-verify operation that selects the public key by `key_id`,
   and a public-key resolution operation by `key_id`.
2. THE Local_File_Signer SHALL load an Ed25519 private key from a file whose path is
   provided by an environment variable.
3. THE Local_File_Signer SHALL read its key file from a path outside the `data/`
   directory.
4. THE Local_File_Signer SHALL derive a `key_id` from its public key and resolve the
   matching public key for verification.
5. WHEN the signing backend is changed, THE System SHALL require changes only to the
   Signer construction and not to the Audit_Chain or the Audit_Event model.

### Requirement 6: Fail-safe but loud signing behavior

**User Story:** As a compliance operator, I want the system to keep working but be
explicit about signing state, so that an unsigned trail is never silently presented
as signed and a misconfigured key is never silently ignored.

#### Acceptance Criteria

1. IF no signing key is configured, THEN THE Audit_Chain SHALL still write events
   with a valid hash chain and SHALL mark those events UNSIGNED by setting
   `signature` and `key_id` to null.
2. IF no signing key is configured, THEN THE Verifier SHALL report the chain as
   `INTACT_UNSIGNED` and SHALL NOT report it as signed.
3. IF a signing key is configured but the key file is missing, has invalid
   permissions, is not a valid Ed25519 private key, or cannot be parsed, THEN THE
   Signer SHALL fail loudly at construction with an error identifying the cause.
4. IF a signing key is configured but cannot be loaded, THEN THE System SHALL NOT
   fall back to writing UNSIGNED events.

### Requirement 7: Verification outcomes

**User Story:** As a compliance operator, I want a verification routine that
re-walks the chain and checks signatures, so that I can prove whether the audit
trail is intact and signed, or pinpoint the first broken link or invalid signature.

#### Acceptance Criteria

1. WHEN verification runs, THE Verifier SHALL re-walk the chain from genesis and
   SHALL confirm, for each event, that `seq` increments by exactly 1, that
   `prev_hash` equals the previous event's `entry_hash`, and that the recomputed
   `entry_hash` equals the stored `entry_hash`.
2. WHEN every event's hash links are valid and every event carries a signature that
   validates under the public key resolved for its `key_id`, THE Verifier SHALL
   return `INTACT_SIGNED`.
3. WHEN every event's hash links are valid but at least one event carries no
   signature, THE Verifier SHALL return `INTACT_UNSIGNED` and SHALL report the count
   of unsigned events.
4. IF a hash-link failure, sequence gap, or malformed line is encountered, THEN THE
   Verifier SHALL return `BROKEN_CHAIN` with the `first_broken_seq` and a
   human-readable `reason`.
5. IF a signed event's signature does not validate, or its `key_id` resolves to no
   known public key, THEN THE Verifier SHALL return `SIGNATURE_INVALID` with the
   `first_bad_signature_seq` and a human-readable `reason`.
6. WHEN the chain file is empty, THE Verifier SHALL return `INTACT_SIGNED` with
   `total_events == 0`.
7. WHEN verification detects a signed event whose content was altered and whose
   `entry_hash` was recomputed to pass the hash chain, THE Verifier SHALL return
   `SIGNATURE_INVALID` at that event's `seq`.

### Requirement 8: Verification CLI and exit-code contract

**User Story:** As an operator, I want a command-line verification entry point with
a clear exit-code contract, so that I can check audit integrity in scripts and
pipelines and distinguish an unsigned chain from a signed one.

#### Acceptance Criteria

1. THE System SHALL provide a CLI entry point invoked as
   `python -m fraud_detect.audit.verify` that verifies the chain and reports the
   Verification_Result.
2. WHEN the verification result is `INTACT_SIGNED`, THE CLI SHALL exit with code 0.
3. WHEN the verification result is `BROKEN_CHAIN` or `SIGNATURE_INVALID`, THE CLI
   SHALL exit with a non-zero code.
4. WHEN the verification result is `INTACT_UNSIGNED`, THE CLI SHALL surface the
   unsigned state distinctly and SHALL NOT present it as signed.

### Requirement 9: Governance preserved — never auto-reject, never auto-repair

**User Story:** As a compliance officer, I want a broken, invalid, missing-signature,
or unexpectedly-unsigned chain surfaced to a human as an incident, so that the
system's existing never-auto-reject, human-in-the-loop governance stance is kept.

#### Acceptance Criteria

1. IF verification reports `BROKEN_CHAIN`, `SIGNATURE_INVALID`, or an unexpected
   `INTACT_UNSIGNED`, THEN THE System SHALL surface the result to a human and SHALL
   NOT modify, repair, or rewrite the chain.
2. WHEN an audit integrity failure is detected, THE System SHALL preserve the
   existing never-auto-reject behavior and SHALL NOT make a reject decision on its
   own.
3. IF an append fails mid-write and leaves a malformed trailing line, THEN THE
   Verifier SHALL report the malformed line as a break at the tail and THE System
   SHALL NOT auto-truncate the chain.

### Requirement 10: Durability and single-writer concurrency

**User Story:** As an operator, I want each append made durable and serialized, so
that a crash mid-append cannot silently corrupt the chain and concurrent in-process
writers cannot interleave `seq`/`prev_hash` values.

#### Acceptance Criteria

1. WHEN an event is appended, THE Audit_Chain SHALL flush and fsync the chain file
   before returning.
2. WHILE multiple in-process writers attempt to append concurrently, THE Audit_Chain
   SHALL serialize appends using an in-process lock so that `seq` and `prev_hash`
   remain consistent.

### Requirement 11: Honest security boundary (constraints)

**User Story:** As a compliance officer, I want the prototype's security limits
documented as explicit constraints, so that no one overstates what the prototype
guarantees and the production hardening path is clear.

#### Acceptance Criteria

1. THE System documentation SHALL state that the prototype is tamper-evident and
   forgery-resistant while the signing key is secure, but is not proof against
   signing-key theft and not proof against a wipe-and-restart of the chain.
2. THE System documentation SHALL state that production MUST move signing-key custody
   to a KMS/HSM so the private key never leaves the security boundary.
3. THE System documentation SHALL state that detecting a wipe-and-restart requires
   external anchoring or checkpointing, which is a production requirement and out of
   scope for the prototype.

## Assumptions and Deferred Scope

The following items derive from the design's Open Questions. They are represented as
assumptions or deferred/out-of-scope notes and do not constrain the prototype
acceptance criteria above.

- **Migration of existing `cases.json` audit_log (assumption):** The prototype starts
  fresh — the new chain begins at a dated genesis event and existing embedded
  `audit_log` entries are kept read-only for history. A one-time backfill into a
  single continuous chain is deferred.
- **Scope of logged events (deferred):** The chain covers decision events
  (`SCREENED`, `HUMAN_APPROVED`, `HUMAN_REJECTED`) for v1. Absorbing the access/view
  events currently in `data/access_audit.jsonl` into the same chained log is a
  follow-up using the same Audit_Chain primitive.
- **Cross-process concurrency (deferred):** The prototype is single-process and uses
  an in-process lock plus per-append fsync. Cross-process file locking (e.g. `flock`)
  or a dedicated writer process is deferred to the production database stage.
- **Production key management and rotation (out of scope for prototype):** Key
  custody location, rotation cadence, distribution of multiple active/retired public
  keys to verifiers, and key revocation (including how `verify` should report events
  signed by a revoked key) are a production key-management design that rides on the
  same Signer interface. The prototype uses a single local-file key and stores
  `key_id` per event so a future rotation does not break verification of older events.
