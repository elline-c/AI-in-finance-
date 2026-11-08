"""Property-based test for the audit chain: every event carries its case_id.

Feature: immutable-audit-trail, Property 13: Every event carries its case_id.

Every recorded event has a non-empty ``case_id`` equal to the case it was
recorded for, and :meth:`AuditChain.events_for` returns exactly the events
appended for a given case, in chain order.

Validates: Requirements 3.3

The chain writes to a dedicated append-only JSONL file, so each generated
example uses a FRESH temporary file created inside the test body (one isolated
file per example) to avoid any cross-example state leaking through the chain.
"""
from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain

# Non-empty case_id strings. A small sampled set keeps multiple events landing
# on the same case (so events_for grouping is exercised), while the text branch
# broadens coverage with arbitrary non-empty case ids.
_case_ids = st.one_of(
    st.sampled_from(["CASE-1", "CASE-2", "CASE-3"]),
    st.text(min_size=1, max_size=12),
)

# One (case_id, event, detail) record to append.
_record = st.tuples(
    _case_ids,
    st.text(min_size=0, max_size=20),  # event name
    st.text(min_size=0, max_size=40),  # detail
)

_records = st.lists(_record, min_size=0, max_size=25)


@settings(max_examples=150)
@given(records=_records)
def test_every_event_carries_its_case_id(records):
    """Property 13: each appended event carries the exact case_id it was
    recorded for, and ``events_for`` returns that case's events in order."""
    with tempfile.TemporaryDirectory() as tmp:
        chain_path = Path(tmp) / "audit_chain.jsonl"
        chain = AuditChain(path=str(chain_path))

        # Append each record and capture the event it produced, in order.
        appended = []
        for case_id, event, detail in records:
            produced = chain.append(case_id=case_id, event=event, detail=detail)
            # Every appended event carries a non-empty case_id equal to the one
            # it was called with (matched by append order).
            assert produced.case_id == case_id
            assert produced.case_id != ""
            appended.append(produced)

        # Build the expected per-case event list, preserving append order.
        expected_by_case: "OrderedDict[str, list]" = OrderedDict()
        for ev in appended:
            expected_by_case.setdefault(ev.case_id, []).append(ev)

        # For each distinct case_id used, events_for returns exactly the events
        # appended for that case, in chain order.
        for case_id, expected_events in expected_by_case.items():
            got = chain.events_for(case_id)
            assert [e.seq for e in got] == [e.seq for e in expected_events]
            assert [e.case_id for e in got] == [case_id] * len(expected_events)
            assert [e.entry_hash for e in got] == [
                e.entry_hash for e in expected_events
            ]
