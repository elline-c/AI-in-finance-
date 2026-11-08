"""Property-based test: any single structural corruption is BROKEN_CHAIN.

Feature: immutable-audit-trail, Property 8: Any single structural corruption is
caught as BROKEN_CHAIN.

For any intact chain of >= 2 events, applying EXACTLY ONE structural corruption
to the on-disk JSONL file and then re-verifying must return
``status == BROKEN_CHAIN`` with ``first_broken_seq`` pointing at the first
affected position. The corruption kinds exercised are:

* edit one field of a middle line (change ``detail`` in the JSON) WITHOUT
  recomputing ``entry_hash`` -> entry_hash mismatch at that line's seq;
* delete an interior line -> sequence gap / prev_hash mismatch (deleting the
  tail is excluded: a truncated hash chain is still a valid shorter chain and
  is undetectable structurally);
* reorder two adjacent lines -> seq out of order / prev_hash mismatch;
* insert a duplicate line -> break;
* append a malformed (non-JSON) trailing line -> break at the tail.

The corruption is applied by the TEST reading the JSONL lines, mutating them,
and rewriting the file -- the test is standing in for a tamperer. Production
code (``AuditChain.verify``) never rewrites the file; we also assert the file
bytes are unchanged by the ``verify()`` call itself.

Validates: Requirements 7.4, 9.3
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.verify import VerificationStatus

# A chain of >= 3 events. Unsigned is fine (faster) and still fully exercises
# the hash-chain structural checks. (>= 3 rather than >= 2 so the "delete an
# interior line" kind always has a non-first, non-last line to remove.) Text
# includes unicode to stress the canonical serializer and the JSON round-trip of
# the corruption step.
_records = st.lists(
    st.tuples(st.text(), st.text(), st.text()),
    min_size=3,
    max_size=10,
)

# The structural corruption kinds Property 8 enumerates.
_KINDS = ["edit_field", "delete_line", "reorder_adjacent", "insert_duplicate", "malformed_tail"]


def _build_chain(path: Path, records) -> None:
    """Append ``records`` to a fresh UNSIGNED chain at ``path``."""
    chain = AuditChain(path=str(path))  # no signer -> UNSIGNED, faster
    for case_id, event, detail in records:
        chain.append(case_id=case_id, event=event, detail=detail)


def _read_lines(path: Path) -> list[str]:
    """Return the chain file's JSONL records, in file order.

    Split on the actual ``"\\n"`` record delimiter only (not ``str.splitlines``,
    which would also break on control chars like NEL/VT/LS that can legitimately
    appear inside a quoted JSON string value and would falsely split one event
    across multiple "lines"). The trailing empty element after the final newline
    is dropped.
    """
    text = path.read_text(encoding="utf-8")
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _write_lines(path: Path, lines) -> None:
    """Rewrite the chain file from ``lines`` (newline-delimited). Tamperer only."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@settings(max_examples=150)
@given(records=_records, kind=st.sampled_from(_KINDS), rnd=st.randoms(use_true_random=True))
def test_single_structural_corruption_is_broken_chain(records, kind, rnd):
    """Property 8: exactly one structural corruption yields BROKEN_CHAIN with
    ``first_broken_seq`` at the first affected position; ``verify()`` leaves the
    file bytes unchanged."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"
        _build_chain(chain_path, records)

        chain = AuditChain(path=str(chain_path))

        # Sanity: the freshly built chain verifies as intact before we corrupt it.
        assert chain.verify().intact is True

        lines = _read_lines(chain_path)
        n = len(lines)
        assert n == len(records) and n >= 2

        # expected_seq tracks where, deterministically, we expect the break.
        expected_seq = None

        if kind == "edit_field":
            # Edit the "detail" field of a chosen line WITHOUT recomputing the
            # entry_hash -> stored entry_hash no longer matches the content, so
            # the break surfaces at exactly that line's seq (== its index).
            i = rnd.randrange(0, n)
            obj = json.loads(lines[i])
            obj["detail"] = (obj["detail"] or "") + "\u2622tampered"
            lines[i] = json.dumps(obj, ensure_ascii=False)
            expected_seq = i

        elif kind == "delete_line":
            # Delete an INTERIOR line (not the first, not the last). The line
            # that slides into position i now carries seq i+1 while the walk
            # expects seq i -> sequence gap reported at expected_seq == i.
            #
            # Deleting the LAST line is deliberately excluded: truncating a hash
            # chain's tail leaves a shorter but still-valid chain, so it is NOT
            # detectable by the structural walk (it needs an external length
            # anchor). With records min_size == 3 an interior line always exists.
            i = rnd.randrange(1, n - 1)
            del lines[i]
            expected_seq = i

        elif kind == "reorder_adjacent":
            # Swap two adjacent lines at i, i+1. At position i the walk now finds
            # seq i+1 but expects i -> break reported at expected_seq == i.
            i = rnd.randrange(0, n - 1)
            lines[i], lines[i + 1] = lines[i + 1], lines[i]
            expected_seq = i

        elif kind == "insert_duplicate":
            # Insert a duplicate of line i right after it. The duplicate carries
            # seq i, but after processing the original the walk expects i+1 ->
            # break reported at expected_seq == i + 1.
            i = rnd.randrange(0, n)
            lines.insert(i + 1, lines[i])
            expected_seq = i + 1

        elif kind == "malformed_tail":
            # Append a non-JSON trailing line. The malformed line sits at the
            # 0-based position equal to the original length -> break at the tail
            # reported at expected_seq == n.
            lines.append("this is not json {{{")
            expected_seq = n

        else:  # pragma: no cover - guarded by sampled_from
            raise AssertionError(f"unknown kind {kind!r}")

        _write_lines(chain_path, lines)

        # Capture the exact on-disk bytes AFTER corruption so we can prove the
        # verify() call itself is side-effect free (Requirement 9.1).
        before = chain_path.read_bytes()

        result = chain.verify()

        after = chain_path.read_bytes()
        assert after == before, "verify() must not modify the chain file"

        # Core assertion of Property 8: a single structural corruption is caught.
        assert result.status == VerificationStatus.BROKEN_CHAIN
        assert result.first_broken_seq is not None

        # Deterministic position check for every kind exercised here.
        assert result.first_broken_seq == expected_seq, (
            f"kind={kind}: expected first_broken_seq={expected_seq}, "
            f"got {result.first_broken_seq} (reason={result.reason})"
        )
