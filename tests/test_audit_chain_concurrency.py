"""Property-based test for ``AuditChain.append`` under concurrent writers.

Feature: immutable-audit-trail, Property 12: Concurrent in-process appends
serialize to a contiguous chain.

``AuditChain.append`` serializes concurrent in-process writers with an internal
``threading.Lock`` (single-writer discipline, Requirement 10.2). This test fires
N appends at a single ``AuditChain`` instance from N threads that start as
simultaneously as possible (a ``threading.Barrier`` releases them together so
the lock is actually contended) and asserts the result is a clean, contiguous
chain: every ``seq`` from 0..N-1 appears exactly once and the hash links verify
end to end.

Because ``verify`` (task 6.2) may not be merged in this working tree yet, this
test does NOT call ``AuditChain.verify``. It asserts the intactness invariants
directly by reading the chain back:

* the file has exactly N lines and exactly N events parse out of it;
* the multiset of ``seq`` values is exactly ``{0, 1, ..., N-1}`` (each once);
* sorted by ``seq``, event 0 is genesis (``prev_hash == GENESIS_HASH``) and each
  later event links ``prev_hash`` to the prior event's ``entry_hash`` with
  ``seq`` incrementing by exactly 1; and
* each event's recomputed ``entry_hash`` matches its stored ``entry_hash``.

Validates: Requirements 10.2

Isolation note: Hypothesis reuses the pytest ``tmp_path`` across examples, so
each example creates its OWN fresh temporary directory inside the test body and
tears it down before returning -- no chain state leaks between examples.

Note: thread-based concurrency property tests are necessarily lighter-weight
than pure-logic ones (each example spins up N OS threads), so ``max_examples``
is kept modest and ``deadline`` is disabled; it still runs at least 50 examples.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit import canonical
from fraud_detect.audit.chain import AuditChain


def _run_concurrent_appends(n: int) -> None:
    """Fire ``n`` concurrent appends at one chain and assert a clean chain.

    A fresh temp directory is created per call so Hypothesis examples never
    share a chain file. UNSIGNED mode (no signer) keeps each append fast.
    """
    with tempfile.TemporaryDirectory() as tmp:
        chain_path = Path(tmp) / "audit_chain.jsonl"
        chain = AuditChain(str(chain_path))  # UNSIGNED: no signer

        # Release all worker threads together so they genuinely race for the
        # append lock rather than finishing one-by-one as they are spawned.
        barrier = threading.Barrier(n)
        errors: List[BaseException] = []
        errors_lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                barrier.wait()
                chain.append(
                    case_id=f"case-{i}",
                    event="SCREENED",
                    detail=f"concurrent append #{i}",
                )
            except BaseException as exc:  # noqa: BLE001 - surface any thread error
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No worker thread may have raised -- an exception would mean the lock
        # failed to serialize writers cleanly.
        assert not errors, f"worker thread(s) raised: {errors!r}"

        # (a) The file has exactly N physical lines.
        raw_lines = chain_path.read_text(encoding="utf-8").splitlines()
        assert len(raw_lines) == n, (
            f"expected {n} lines in the chain file, got {len(raw_lines)}"
        )

        # (b) Exactly N events parse out, in file order.
        events = chain._read_all_events()  # noqa: SLF001 - reading back the chain
        assert len(events) == n, f"expected {n} parsed events, got {len(events)}"

        # (c) The multiset of seq values is exactly {0, 1, ..., N-1}: each once.
        seqs = sorted(e.seq for e in events)
        assert seqs == list(range(n)), (
            f"seq values are not exactly 0..{n - 1} each once: {seqs}"
        )

        # (d) Sorted by seq, the chain links contiguously and every entry_hash
        #     recomputes to its stored value.
        ordered = sorted(events, key=lambda e: e.seq)
        prev_hash = canonical.GENESIS_HASH
        for expected_seq, event in enumerate(ordered):
            assert event.seq == expected_seq
            assert event.prev_hash == prev_hash, (
                f"seq {event.seq} prev_hash does not link to the prior entry_hash"
            )
            recomputed = canonical.compute_entry_hash(canonical.hash_payload(event))
            assert event.entry_hash == recomputed, (
                f"seq {event.seq} entry_hash does not match its content"
            )
            prev_hash = event.entry_hash


# N is a small Hypothesis-chosen integer so the thread count stays manageable
# while still exercising real contention on the append lock.
@settings(max_examples=60, deadline=None)
@given(n=st.integers(min_value=2, max_value=12))
def test_concurrent_appends_serialize_to_contiguous_chain(n):
    """Property 12: N concurrent in-process appends serialize into a single
    contiguous, intact chain with seq 0..N-1 each appearing exactly once."""
    _run_concurrent_appends(n)
