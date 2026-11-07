"""Task 20.3: property test for the implausible-measurement monitor.

Feature: ml-retraining, Property 37: The implausible-measurement monitor

**Validates: Requirements 15.1, 15.2, 15.4**
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining.monitor import (
    BLOCKED_CHANGES,
    Measurement,
    MeasurementStop,
    Monitor,
    SeedLedger,
    SeedSubstitutionRefused,
    StopLockActive,
    fired_rules,
)

UNSEEN = "test_unseen_layout"
SPLITS = ["train", "validation", "test", "test_unseen_layout", "real_style"]
SOURCES = ["corpus", "real_style", "forgery"]

auc_st = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.sampled_from([0.0, 0.4999, 0.5, 0.9899, 0.99, 1.0]),
)
count_st = st.one_of(st.integers(0, 60), st.sampled_from([0, 29, 30, 31]))
opt_bit = st.one_of(st.none(), st.integers(0, 1))
opt_count = st.one_of(st.none(), st.integers(0, 5))

measurement_st = st.builds(
    Measurement,
    name=st.sampled_from(["roc_auc", "flag", "doc_count"]),
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    source=st.sampled_from(SOURCES),
    split=st.sampled_from(SPLITS),
    master_seed=st.integers(0, 10**9),
    n_pos=count_st,
    n_neg=count_st,
    model=st.sampled_from(["candidate", "current"]),
    roc_auc=auc_st,
    flag=opt_bit,
    label=opt_bit,
    doc_count=opt_count,
    manifest_count=opt_count,
)


def _expected(m: Measurement, scope: str) -> list:
    out = []
    if m.model == "candidate" and m.split == UNSEEN and m.roc_auc is not None and m.roc_auc >= 0.99:
        out.append("a")
    if m.roc_auc is not None and m.roc_auc < 0.5 and m.n_pos >= 30 and m.n_neg >= 30:
        out.append("b")
    if (m.source == "real_style" and m.model == "candidate" and m.flag is not None
            and m.label is not None):
        if m.flag != m.label and (scope == "all" or (m.flag == 1 and m.label == 0)):
            out.append("c")
    if m.doc_count is not None and m.manifest_count is not None and m.doc_count != m.manifest_count:
        out.append("d")
    return out


def _clock():
    state = {"n": 0}
    start = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)

    def now():
        state["n"] += 1
        return start + timedelta(seconds=state["n"])

    return now


# Feature: ml-retraining, Property 37: The implausible-measurement monitor
@settings(max_examples=150, deadline=None)
@given(
    ms=st.lists(measurement_st, min_size=1, max_size=8),
    scope=st.sampled_from(["all", "false_alarms_only"]),
    change=st.sampled_from(BLOCKED_CHANGES),
    seed_pair=st.tuples(st.integers(0, 10**6), st.integers(0, 10**6)).filter(lambda t: t[0] != t[1]),
    bad_outcome=st.sampled_from(["failed", "stopped"]),
)
def test_property_37_monitor(ms, scope, change, seed_pair, bad_outcome):
    # Rules fire if and only if their condition holds.
    for m in ms:
        assert fired_rules(m, scope) == _expected(m, scope)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        mon = Monitor(run_dir, is_ignored=lambda p: True, clock=_clock(), rule_c_scope=scope)
        accepted = []
        stopping = None
        for m in ms:
            if _expected(m, scope):
                with pytest.raises(MeasurementStop) as ei:
                    mon.observe(m)
                stopping = (m, ei.value)
                break
            mon.observe(m)  # clean measurement accepted silently
            accepted.append(m)
            assert not mon.is_stopped
            mon.guard(change)  # no lock, no refusal

        if stopping is None:
            assert not mon.is_stopped
            return

        m, exc = stopping
        rules = _expected(m, scope)
        # The record carries the measurement fields.
        assert exc.exit_code == 4
        rec = exc.record
        assert rec["rule"] == rules[0]
        assert rec["measurement"] == m.name
        assert rec["value"] == m.value
        assert rec["source"] == m.source
        assert rec["split"] == m.split
        assert rec["master_seed"] == m.master_seed
        assert rec["n_pos"] == m.n_pos
        assert rec["n_neg"] == m.n_neg
        on_disk = json.loads((run_dir / "investigation.json").read_text(encoding="utf-8"))
        assert on_disk == rec

        # Locked: a new instance sees the lock, no measurement is accepted, changes are refused by name.
        again = Monitor(run_dir, is_ignored=lambda p: True, clock=_clock(), rule_c_scope=scope)
        assert again.is_stopped
        before = (run_dir / "investigation.json").read_text(encoding="utf-8")
        for other in ms:
            with pytest.raises(StopLockActive):
                again.observe(other)
        for c in BLOCKED_CHANGES:
            with pytest.raises(StopLockActive) as ei:
                again.guard(c)
            assert c in str(ei.value)
        assert (run_dir / "investigation.json").read_text(encoding="utf-8") == before

        # A decision entry lifts the lock only when it is "continue"; then measurements flow again.
        again.decide("continue", "reviewed", "owner")
        assert not again.is_stopped
        again.guard(change)

        # Seeds that failed or stopped are never replaced by another seed.
        ledger = SeedLedger(run_dir, is_ignored=lambda p: True, clock=_clock())
        bad, new = seed_pair
        ledger.attempt(bad)
        ledger.record(bad, bad_outcome, "x")
        with pytest.raises(SeedSubstitutionRefused):
            ledger.substitute(bad, new)
        assert [e["seed"] for e in ledger.entries()] == [bad]
