"""Property-based test for signature-level tampering detection.

Feature: immutable-audit-trail, Property 9: Signature tampering is caught as
SIGNATURE_INVALID.

A chain built intact with a configured signer, after EXACTLY ONE signature-level
tamper that leaves the hash chain intact, verifies as ``SIGNATURE_INVALID`` at
the tampered event's ``seq`` -- and because the hash chain (pass 1) still walks
cleanly, ``first_broken_seq`` is ``None``.

The tampers deliberately touch ONLY ``signature`` or ``key_id``: neither field is
part of the hashed payload (``seq, prev_hash, ts, case_id, event, detail``) nor
of ``entry_hash``, so editing them in the on-disk JSON line keeps every hash link
valid. The failure is therefore isolated to the signature pass (pass 2):

* **bad-signature** -- replace one event's ``signature`` with a
  different-but-valid-base64 string (the decoded signature bytes are rotated and
  re-encoded, or another event's signature is substituted). The bytes decode
  fine but do not validate under the signing key.
* **unknown-key_id** -- change one event's ``key_id`` to a value the signer does
  not resolve (e.g. ``"ed25519:deadbeef0000"``). An unresolvable ``key_id`` on a
  signed event is a SIGNATURE_INVALID failure.

Verification uses the SAME signer that built the chain, so a genuine ``key_id``
would normally resolve -- the failure comes purely from the tamper.

Validates: Requirements 7.5

Strategy + isolation/performance notes: Ed25519 key generation is comparatively
expensive, so a single signer is generated once per module (``module_signer``)
and reused across examples -- the property does not depend on which key is used.
Each example still builds its chain on its OWN fresh temp file created inside the
test body, so no signed-chain state leaks between examples.
"""
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import LocalFileSigner
from fraud_detect.audit.verify import VerificationStatus

# A non-empty list of (case_id, event, detail) tuples. Unicode text exercises the
# canonical serializer (ensure_ascii=False) and the signed payload.
_event_specs = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=40),   # case_id (non-empty to be meaningful)
        st.text(min_size=0, max_size=40),   # event
        st.text(min_size=0, max_size=200),  # detail
    ),
    min_size=1,
    max_size=12,
)

# Which signature-level tamper to apply (Hypothesis-chosen per example).
_tamper_kinds = st.sampled_from(["rotate_signature", "substitute_signature", "unknown_key_id"])


@pytest.fixture(scope="module")
def module_signer() -> LocalFileSigner:
    """A single Ed25519 signer generated once for the whole module.

    Generating a keypair per Hypothesis example would dominate the runtime, so
    the property reuses this one signer across all examples. Which specific key
    is used does not matter to the property under test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield LocalFileSigner.generate(str(Path(tmp) / "key.pem"))


def _rotate_signature_b64(signature_b64: str) -> str:
    """Return a different-but-valid-base64 signature.

    Decodes the base64 signature, rotates its bytes by one position (so it stays
    64 bytes and decodes cleanly), and re-encodes. The result is a syntactically
    valid Ed25519-length signature that will not validate under the real key.
    """
    raw = base64.b64decode(signature_b64)
    rotated = raw[1:] + raw[:1]
    if rotated == raw:  # pathological all-equal bytes -> flip one bit instead
        rotated = bytes([raw[0] ^ 0x01]) + raw[1:]
    return base64.b64encode(rotated).decode("ascii")


def _tamper_line(line: str, tamper_kind: str, donor_signature_b64: str | None) -> str:
    """Apply EXACTLY ONE signature-level tamper to one JSON chain line.

    Only ``signature`` / ``key_id`` are touched -- the hashed fields and
    ``entry_hash`` are left byte-for-byte intact, so the hash chain still walks.
    """
    obj = json.loads(line)
    if tamper_kind == "rotate_signature":
        obj["signature"] = _rotate_signature_b64(obj["signature"])
    elif tamper_kind == "substitute_signature":
        # Swap in a valid signature from a different event; it is well-formed
        # base64 but was produced over different signed bytes, so it won't match.
        obj["signature"] = donor_signature_b64
    elif tamper_kind == "unknown_key_id":
        obj["key_id"] = "ed25519:deadbeef0000"
    else:  # pragma: no cover - guarded by the strategy
        raise AssertionError(f"unknown tamper kind: {tamper_kind}")
    return json.dumps(obj)


@settings(max_examples=150, deadline=None)
@given(specs=_event_specs, tamper_kind=_tamper_kinds, idx_seed=st.integers(min_value=0))
def test_signature_tamper_is_signature_invalid(specs, tamper_kind, idx_seed, module_signer):
    """Property 9: one signature-level tamper (hash chain intact) yields
    SIGNATURE_INVALID at the tampered event's seq, with first_broken_seq None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"
        chain = AuditChain(path=str(chain_path), signer=module_signer)

        for case_id, event, detail in specs:
            chain.append(case_id=case_id, event=event, detail=detail)

        # Sanity: the freshly built chain is INTACT_SIGNED before tampering.
        assert chain.verify().status == VerificationStatus.INTACT_SIGNED

        # Split ONLY on the newline delimiters the chain writes. ``str.splitlines``
        # would also break on Unicode line separators (e.g. U+0085) that can appear
        # inside a case_id/detail string; the chain is strictly "\n"-delimited JSON,
        # and real newlines inside a field are JSON-escaped, so splitting on "\n" is
        # the faithful line split.
        raw_text = chain_path.read_text(encoding="utf-8")
        raw_lines = [ln for ln in raw_text.split("\n") if ln]
        assert len(raw_lines) == len(specs)

        # Pick exactly one line to tamper (its seq equals its index in the file).
        target_index = idx_seed % len(raw_lines)

        # For the "substitute" tamper we need a *different* event's signature as a
        # donor. With a single-event chain there is no other event, so substitute
        # falls back to the rotate tamper (still a signature-level change).
        effective_kind = tamper_kind
        donor_signature_b64 = None
        if tamper_kind == "substitute_signature":
            donor_index = next(
                (i for i in range(len(raw_lines)) if i != target_index), None
            )
            if donor_index is None:
                effective_kind = "rotate_signature"
            else:
                donor_signature_b64 = json.loads(raw_lines[donor_index])["signature"]

        tampered_obj_before = json.loads(raw_lines[target_index])
        raw_lines[target_index] = _tamper_line(
            raw_lines[target_index], effective_kind, donor_signature_b64
        )
        chain_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

        # The tamper must NOT have altered the hashed fields or entry_hash.
        tampered_obj_after = json.loads(raw_lines[target_index])
        for field in ("seq", "prev_hash", "ts", "case_id", "event", "detail", "entry_hash"):
            assert tampered_obj_after[field] == tampered_obj_before[field]

        result = chain.verify()

        # Hash chain (pass 1) stayed intact -> failure is isolated to signatures.
        assert result.status == VerificationStatus.SIGNATURE_INVALID
        assert result.first_broken_seq is None
        assert result.first_bad_signature_seq == tampered_obj_before["seq"]
