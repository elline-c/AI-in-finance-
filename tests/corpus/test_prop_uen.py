"""Property test for the build-time UEN collision check.

Feature: synthetic-corpus-generator, Property 22: Generated UENs never clash and are repeatable
**Validates: Requirements 20.1, 20.2, 20.3**

Pure functions only (no PDFs). The reserved set is built from first draws that really
happen, so clashes occur in most examples, plus unrelated extra UENs.
"""
from __future__ import annotations

import dataclasses

from hypothesis import given, settings, strategies as st

from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.uen_check import UenChecker
from fraud_detect.reference_data import ACRA_REGISTRY

_seed = st.integers(min_value=0, max_value=2**63 - 1)
_extra_uen = st.from_regex(r"\A[0-9]{8,9}[A-Z]\Z", fullmatch=True)


@st.composite
def _scenario(draw):
    seed = draw(_seed)
    ids = [f"C{i:04d}" for i in range(draw(st.integers(1, 6)))]
    firsts = [make_company(seed, cid).uen for cid in ids]
    picks = draw(st.lists(st.sampled_from(range(len(ids))), max_size=len(ids), unique=True))
    reserved = {firsts[i] for i in picks}
    reserved |= set(draw(st.lists(_extra_uen, max_size=5)))
    if draw(st.booleans()):
        reserved |= set(ACRA_REGISTRY)
    return seed, ids, firsts, frozenset(reserved)


def _build_all(seed, ids, reserved):
    """What the builder's pre-pass does: companies in order, each UEN free of everything."""
    checker = UenChecker(reserved)
    claimed: set[str] = set()
    profiles = []
    for cid in ids:
        profile = make_company(seed, cid, is_taken=lambda u: checker.is_taken(u) or u in claimed)
        claimed.add(profile.uen)
        profiles.append(profile)
    return profiles


@settings(max_examples=100, deadline=None)
@given(_scenario())
def test_uens_never_clash_and_are_repeatable(scenario):
    seed, ids, firsts, reserved = scenario
    profiles = _build_all(seed, ids, reserved)
    uens = [p.uen for p in profiles]

    # no company UEN is reserved, and none repeats
    assert not set(uens) & reserved
    assert len(set(uens)) == len(uens)

    # the same inputs give the same UENs (and the same whole profiles)
    assert _build_all(seed, ids, reserved) == profiles

    # a company whose first draw did not clash keeps its profile; a clash changes only the UEN
    seen: set[str] = set()
    for cid, first, profile in zip(ids, firsts, profiles):
        base = make_company(seed, cid)
        assert base.uen == first
        if first not in reserved and first not in seen:
            assert profile == base
        else:
            assert profile.uen != first
            assert dataclasses.replace(profile, uen=first) == base
        seen.add(profile.uen)
