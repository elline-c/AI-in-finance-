"""Task 17.3: property test for the Awkward_Set structure.

Feature: ml-retraining, Property 25: Awkward_Set structure

Building an Awkward_Set renders PDFs, which is slow, so each property runs only
10 examples (no deadline) over a random seed and a random subset of case types.

**Validates: Requirements 8.1, 8.2, 8.7**
"""
from __future__ import annotations

import hashlib

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from fraud_detect.corpus.schema import SimulatorKnobs
from fraud_detect.ml.retraining import awkward as aw

CASES = aw.CASE_TYPES

_SETTINGS = settings(max_examples=10, deadline=None, derandomize=False,
                     suppress_health_check=list(HealthCheck))

seeds = st.one_of(st.just(aw.AWKWARD_MASTER_SEED), st.integers(min_value=1, max_value=2**31 - 1))
case_subsets = st.lists(st.sampled_from(CASES), min_size=1, max_size=2, unique=True).map(tuple)


def _credits(ledger):
    out = [0.0] * len(ledger.months)
    for r in ledger.rows:
        if r.amount > 0:
            out[r.month_index] += r.amount
    return out


def _fp(s):
    return [(d.doc_id, d.company_id, d.uen, d.layout, d.tamper_type,
             hashlib.sha256(d.pdf).hexdigest()) for d in s.docs]


def _check_figure(case, d):
    if case == "seasonal_growth":
        c = _credits(d.ledger)
        assert max(c) >= 2.0 * min(c)
        assert sum(c[-3:]) >= 1.10 * sum(c[:3])
    elif case == "instalments":
        counts = {}
        for r in d.ledger.rows:
            if r.kind == "receipt" and r.invoice_ref:
                counts[r.invoice_ref] = counts.get(r.invoice_ref, 0) + 1
        assert any(n >= 2 for n in counts.values())
    elif case == "long_12_months":
        idx = [m.year * 12 + m.month for m in d.ledger.months]
        assert idx == list(range(idx[0], idx[0] + 12))
    elif case == "round_payroll_rent":
        rows = [r for r in d.ledger.rows if r.kind in ("payroll", "rent")]
        assert rows
        whole = sum(round(abs(r.amount) * 100) % 10000 == 0 for r in rows)
        assert whole / len(rows) >= 0.80


@_SETTINGS
@given(seed=seeds, cases=case_subsets)
def test_awkward_set_structure(seed, cases):
    try:
        s1 = aw.build_awkward_set(seed, cases=cases)
    except aw.AwkwardBuildError as err:
        # a build may only fail by naming a case type that is below 3 in a class
        assert err.case_type in cases
        assert err.n_genuine < 3 or err.n_tampered < 3
        assert err.case_type in str(err)
        return

    for case in cases:
        docs = s1.by_case(case)
        assert sum(not d.is_tampered for d in docs) >= 3
        assert sum(d.is_tampered for d in docs) >= 3
        genuine = {}
        for d in docs:
            if not d.is_tampered:
                genuine[d.company_id] = d
                _check_figure(case, d)
        for t in (d for d in docs if d.is_tampered):
            g = genuine[t.company_id]
            assert t.company == g.company and t.layout == g.layout
            assert t.tamper_type in aw.TAMPER_ROTATION
    assert {d.case_type for d in s1.docs} == set(cases)

    s2 = aw.build_awkward_set(seed, cases=cases)
    assert _fp(s1) == _fp(s2)


@settings(max_examples=10, deadline=None, suppress_health_check=list(HealthCheck))
@given(max_attempts=st.integers(min_value=1, max_value=3))
def test_case_below_three_stops_pipeline_with_type_and_counts(max_attempts):
    flat = SimulatorKnobs(seasonality_strength=0.0, growth_range=(-0.01, -0.01))
    out = aw.prepare_awkward_set(cases=("seasonal_growth",),
                                 knobs={"seasonal_growth": flat},
                                 max_attempts=max_attempts)
    assert out.ok is False and out.exit_code == 1
    assert out.proceed_to_switch_rule is False and out.awkward_set is None
    assert "seasonal_growth" in out.message
    with pytest.raises(aw.AwkwardBuildError) as exc:
        aw.build_awkward_set(cases=("seasonal_growth",), knobs={"seasonal_growth": flat},
                             max_attempts=max_attempts)
    e = exc.value
    assert e.n_genuine < 3 or e.n_tampered < 3
    assert str(e.n_genuine) in out.message and str(e.n_tampered) in out.message
