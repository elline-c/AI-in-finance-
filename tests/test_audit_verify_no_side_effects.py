"""Property-based test: verification is side-effect free on the chain file.

Feature: immutable-audit-trail, Property 11: Verification has no side effects on
the chain.

For any chain — intact OR corrupted, signed OR unsigned — calling
``AuditChain.verify()`` leaves the chain file's bytes exactly unchanged.
Verification is read-only: it never modifies, repairs, truncates, or rewrites
the chain, and it never auto-creates an absent chain file.

Each example:

* builds a fresh chain in a per-example temp directory, choosing per example
  whether events are SIGNED (via :class:`LocalFileSigner`) or UNSIGNED;
* sometimes applies a Hypothesis-chosen structural corruption (edit a field,
  delete an interior line, reorder adjacent lines, insert a duplicate, append a
  malformed trailing line) so corrupted chains are covered alongside intact ones;
* captures the on-disk bytes, calls ``verify()`` (possibly several times), and
  asserts the bytes are byte-for-byte identical afterwards.

The absent-file case is covered too: verifying a chain whose file does not exist
must not create it (the path must still not exist afterwards).

Validates: Requirements 9.1
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import LocalFileSigner
from fraud_detect.audit.verify import VerificationStatus

# Records to append. Text includes unicode to stress the canonical serializer
# and the JSON round-trip of the corruption step.
_records = st.lists(
    st.tuples(st.text(), st.text(), st.text()),
    min_size=0,
    max_size=8,
)

# Corruption kinds, plus "none" so intact chains are exercised too.
_KINDS = [
    "none",
    "edit_field",
    "delete_line",
    "reorder_adjacent",
    "insert_duplicate",
    "malformed_tail",
]


def _read_lines(path: Path) -> list[str]:
    """Return the chain file's JSONL records, in file order.

    Split on the real ``"\\n"`` record delimiter only (not ``str.splitlines``,
    which would also break on control chars that can legitimately appear inside
    a quoted JSON string). The trailing empty element after the final newline is
    dropped.
    """
    text = path.read_text(encoding="utf-8")
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _write_lines(path: Path, lines) -> None:
    """Rewrite the chain file from ``lines`` (newline-delimited). Tamperer only."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_corruption(path: Path, kind: str, rnd) -> None:
    """Apply one structural corruption of the given ``kind`` to the chain file.

    Falls back to a no-op when there are too few lines for the chosen kind; the
    property under test (no side effects) holds regardless of whether the chain
    ends up intact or broken, so a degenerate fallback is harmless.
    """
    lines = _read_lines(path)
    n = len(lines)

    if kind == "edit_field" and n >= 1:
        i = rnd.randrange(0, n)
        obj = json.loads(lines[i])
        obj["detail"] = (obj.get("detail") or "") + "\u2622tampered"
        lines[i] = json.dumps(obj, ensure_ascii=False)
    elif kind == "delete_line" and n >= 1:
        i = rnd.randrange(0, n)
        del lines[i]
    elif kind == "reorder_adjacent" and n >= 2:
        i = rnd.randrange(0, n - 1)
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
    elif kind == "insert_duplicate" and n >= 1:
        i = rnd.randrange(0, n)
        lines.insert(i + 1 if i + 1 <= n else n, lines[i])
    elif kind == "malformed_tail":
        lines.append("this is not json {{{")
    else:
        # Nothing to do for this (kind, length) combination.
        return

    _write_lines(path, lines)


@settings(max_examples=150)
@given(
    records=_records,
    signed=st.booleans(),
    kind=st.sampled_from(_KINDS),
    rnd=st.randoms(use_true_random=True),
    verify_times=st.integers(min_value=1, max_value=3),
)
def test_verify_has_no_side_effects(records, signed, kind, rnd, verify_times):
    """Property 11: ``verify()`` leaves the chain file bytes unchanged for any
    chain — intact or corrupted, signed or unsigned — and never creates an
    absent file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        chain_path = tmp / "audit_chain.jsonl"

        # Optional signer for this example. The key lives OUTSIDE the chain
        # file's directory, matching the real deployment constraint.
        if signed:
            signer = LocalFileSigner.generate(str(tmp / "signing_key.pem"))
        else:
            signer = None

        chain = AuditChain(path=str(chain_path), signer=signer)

        # --- absent-file case -------------------------------------------------
        # With no records the file is never created by append; verifying it must
        # not create it either.
        if not records:
            assert not chain_path.exists()
            result = chain.verify()
            # An absent chain is a valid empty chain.
            assert result.status == VerificationStatus.INTACT_SIGNED
            assert result.total_events == 0
            assert not chain_path.exists(), (
                "verify() must not create an absent chain file"
            )
            return

        # --- populated chain --------------------------------------------------
        for case_id, event, detail in records:
            chain.append(case_id=case_id, event=event, detail=detail)

        assert chain_path.exists()

        # Possibly corrupt the on-disk chain (the test stands in for a tamperer).
        _apply_corruption(chain_path, kind, rnd)

        # Capture the exact bytes AFTER any corruption, then verify repeatedly.
        before = chain_path.read_bytes()

        for _ in range(verify_times):
            chain.verify()

        after = chain_path.read_bytes()
        assert after == before, "verify() must not modify the chain file"
        assert chain_path.exists(), "verify() must not delete the chain file"
