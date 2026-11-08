"""Property test for splits.

Feature: synthetic-corpus-generator, Property 16: Splits never leak
**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
"""
import random

from hypothesis import given, settings, strategies as st

from fraud_detect.corpus.splits import SPLITS, assign_split

_LAYOUTS = ["L1", "L2", "L3", "L4"]
_master = st.integers(min_value=0, max_value=2**63 - 1)
_docs = st.lists(
    st.tuples(st.text(min_size=1, max_size=8), st.sampled_from(_LAYOUTS)),
    min_size=1,
    max_size=40,
)
_hold_out = st.one_of(st.none(), st.sampled_from(_LAYOUTS))


@st.composite
def _fractions(draw):
    a = draw(st.integers(0, 100))
    b = draw(st.integers(0, 100 - a))
    return (a / 100, b / 100, (100 - a - b) / 100)


@settings(max_examples=100, deadline=None)
@given(master=_master, docs=_docs, hold_out=_hold_out, fractions=_fractions(), seed=st.integers(0, 10**6))
def test_splits_never_leak(master, docs, hold_out, fractions, seed):
    def run(plan):
        return [assign_split(master, c, lay, hold_out, fractions) for c, lay in plan]

    result = run(docs)

    # Exactly one split from the allowed set.
    assert all(s in SPLITS for s in result)

    # Held-out layout <=> test_unseen_layout.
    for (_, lay), s in zip(docs, result):
        assert (s == "test_unseen_layout") == (hold_out is not None and lay == hold_out)

    # One company, one split (among documents not diverted to the held-out layout).
    by_company = {}
    for (c, lay), s in zip(docs, result):
        if s != "test_unseen_layout":
            by_company.setdefault(c, set()).add(s)
    assert all(len(v) == 1 for v in by_company.values())

    # Independent of build order.
    shuffled = list(docs)
    random.Random(seed).shuffle(shuffled)
    expected = dict(zip(docs, result))
    assert dict(zip(shuffled, run(shuffled))) == expected
