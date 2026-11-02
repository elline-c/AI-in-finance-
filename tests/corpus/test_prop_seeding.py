"""Property test for stable seeding.

**Property 8: Same seed, same corpus (seeding part)**
**Validates: Requirements 10.1, 10.6**
"""
import random

from hypothesis import given, settings, strategies as st

from fraud_detect.corpus.seeding import rng, stable_seed

_parts = st.lists(
    st.one_of(st.text(max_size=12), st.integers(min_value=-10**6, max_value=10**6)),
    max_size=5,
)
_master = st.integers(min_value=0, max_value=2**63 - 1)


@settings(max_examples=100, deadline=None)
@given(master=_master, parts=_parts, noise_seed=st.integers(0, 10**6), draws=st.integers(0, 20))
def test_stable_seed_repeatable_and_independent_of_global_random(master, parts, noise_seed, draws):
    first = stable_seed(master, *parts)
    random.seed(noise_seed)
    for _ in range(draws):
        random.random()
    assert stable_seed(master, *parts) == first
    assert isinstance(first, int) and 0 <= first < 2**64


@settings(max_examples=100, deadline=None)
@given(master=_master, parts=_parts, other=_parts, other_master=_master)
def test_distinct_inputs_give_distinct_seeds_and_equal_rng_streams(master, parts, other, other_master):
    a, b = rng(master, *parts), rng(master, *parts)
    assert [a.random() for _ in range(4)] == [b.random() for _ in range(4)]
    # repr-based encoding keeps (int, str) variants apart, so differing inputs differ
    if (master, tuple(parts)) != (other_master, tuple(other)):
        assert stable_seed(master, *parts) != stable_seed(other_master, *other)


@settings(max_examples=50, deadline=None)
@given(master=_master, parts=st.lists(st.text(max_size=8), min_size=2, max_size=4, unique=True))
def test_seed_independent_of_planning_order(master, parts):
    forward = {p: stable_seed(master, "doc", p) for p in parts}
    backward = {p: stable_seed(master, "doc", p) for p in reversed(parts)}
    assert forward == backward
