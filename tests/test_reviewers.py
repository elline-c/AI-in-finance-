"""Tests for the static reviewer / senior-officer rosters (ITEM 5 + ITEM 6).

Confirms the dropdown choice lists are non-empty, that a display string from the
roster validates, and that an off-roster value is rejected.
"""
from __future__ import annotations

from fraud_detect.reviewers import (
    REVIEWERS,
    SENIOR_OFFICERS,
    is_valid_reviewer,
    is_valid_senior,
    reviewer_choices,
    senior_choices,
)


def test_rosters_are_non_empty():
    assert REVIEWERS, "expected at least one reviewer in the roster"
    assert SENIOR_OFFICERS, "expected at least one senior officer in the roster"


def test_choices_use_plain_hyphen_and_match_count():
    choices = reviewer_choices()
    assert len(choices) == len(REVIEWERS)
    for c in choices:
        assert " - " in c
        assert "\u2014" not in c  # no em dash


def test_senior_choices_match_count():
    assert len(senior_choices()) == len(SENIOR_OFFICERS)


def test_known_reviewer_is_accepted():
    known = reviewer_choices()[0]
    assert is_valid_reviewer(known) is True


def test_unknown_reviewer_is_rejected():
    assert is_valid_reviewer("Nobody - Impostor") is False
    assert is_valid_reviewer("") is False
    # A senior officer's string is NOT a valid reviewer string.
    assert is_valid_reviewer(senior_choices()[0]) is False


def test_known_senior_is_accepted():
    known = senior_choices()[0]
    assert is_valid_senior(known) is True


def test_unknown_senior_is_rejected():
    assert is_valid_senior("Nobody - Impostor") is False
    assert is_valid_senior("") is False
    # A plain reviewer's string is NOT a valid senior string.
    assert is_valid_senior(reviewer_choices()[0]) is False
