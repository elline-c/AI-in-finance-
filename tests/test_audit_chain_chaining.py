"""Property-based tests for the audit chain's genesis and chaining invariant.

Feature: immutable-audit-trail, Property 2: Genesis and chaining invariant.

For any non-empty sequence of appended events, the first event has ``seq == 0``
and ``prev_hash == GENESIS_HASH``, and every subsequent event has ``seq`` equal
to the previous event's ``seq`` plus 1 and ``prev_hash`` equal to the previous
event's ``entry_hash``. In addition, every event's recomputed ``entry_hash``
(over its canonical hash payload) equals its stored ``entry_hash``, so the
chain links are reproducible.

Validates: Requirements 2.1, 2.2

The strategy generates a non-empty list of ``(case_id, event, detail)`` tuples.
Each Hypothesis example appends them to a FRESH chain on its own isolated temp
file, created inside the test body, so examples never collide on a shared file.
String fields include unicode to exercise the canonical serializer.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit import canonical
from fraud_detect.audit.chain import AuditChain

# A non-empty list of (case_id, event, detail) tuples. Text includes unicode to
# exercise ensure_ascii=False in the canonical serializer.
_events = st.lists(
    st.tuples(st.text(), st.text(), st.text()),
    min_size=1,
    max_size=12,
)


@settings(max_examples=150)
@given(records=_events)
def test_genesis_and_chaining_invariant(records):
    """Property 2: the first appended event is the genesis (``seq == 0`` and
    ``prev_hash == GENESIS_HASH``); each later event increments ``seq`` by 1 and
    links ``prev_hash`` to the prior event's ``entry_hash``; and every stored
    ``entry_hash`` matches its recomputed value."""
    # Fresh, isolated temp file INSIDE the test body so Hypothesis examples do
    # not collide on a shared path.
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "audit_chain.jsonl"
        chain = AuditChain(path=str(chain_path))  # no signer -> UNSIGNED

        for case_id, event, detail in records:
            chain.append(case_id=case_id, event=event, detail=detail)

        # Read back all events in chain order.
        events = chain._read_all_events()

        assert len(events) == len(records)

        # Genesis invariant: first event.
        assert events[0].seq == 0
        assert events[0].prev_hash == canonical.GENESIS_HASH

        # Chaining invariant for every consecutive pair.
        for earlier, later in zip(events, events[1:]):
            assert later.seq == earlier.seq + 1
            assert later.prev_hash == earlier.entry_hash

        # Every stored entry_hash reproduces from its canonical hash payload.
        for ev in events:
            recomputed = canonical.compute_entry_hash(canonical.hash_payload(ev))
            assert recomputed == ev.entry_hash
