"""Property 18: Company registration is temporary.

**Validates: Requirements 13.2, 13.3, 13.4**
"""
from __future__ import annotations

import copy

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect import reference_data
from fraud_detect import registry
from fraud_detect.nodes import financial_logic
from fraud_detect.corpus.scope import registered_companies


class _Boom(RuntimeError):
    pass


_uens = st.from_regex(r"\A9[0-9]{8}[A-Z]\Z", fullmatch=True).filter(
    lambda u: u not in reference_data.ACRA_REGISTRY)
_names = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ ", min_size=3, max_size=30).map(
    str.strip).filter(lambda n: len(n) >= 3)
_amount = st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False)
_company = st.fixed_dictionaries({
    "uen": _uens,
    "name": _names,
    "baseline": st.fixed_dictionaries(
        {"avg_monthly_credit": _amount, "avg_monthly_debit": _amount}),
})
_companies = st.lists(_company, min_size=1, max_size=5, unique_by=lambda c: c["uen"])


def _snapshot():
    return (copy.deepcopy(reference_data.ACRA_REGISTRY),
            copy.deepcopy(reference_data.CASHFLOW_BASELINE))


@settings(max_examples=50, deadline=None)
@given(companies=_companies, fail=st.booleans())
def test_registration_is_temporary(companies, fail):
    before = _snapshot()
    try:
        with registered_companies(companies):
            for c in companies:
                result = registry.lookup_entity(c["uen"])
                assert result.state == registry.FOUND
                assert result.record.name == c["name"]
                assert financial_logic.CASHFLOW_BASELINE[c["uen"]] == c["baseline"]
            if fail:
                raise _Boom()
    except _Boom:
        assert fail
    assert _snapshot() == before
