"""Tests for ``fraud_detect.corpus.splits`` (design D7, Requirements 11.1-11.5, 11.7, 11.8).

Interface assumed (to be created in task 18.2):

* ``SPLITS``: tuple of the four allowed split names.
* ``assign_split(master_seed, company_id, layout, hold_out_layout, fractions) -> str``
* ``validate_fractions(fractions)``: raises ``ValueError`` with a clear message
  when the fractions are negative or do not sum to 1 within 0.001.
* ``find_gaps(tamper_types, counts_by_split) -> dict[str, list[str]]``: for each
  non-empty split, the sorted tamper types with zero documents.
"""
import pytest

from fraud_detect.corpus.splits import (
    SPLITS,
    assign_split,
    find_gaps,
    validate_fractions,
)

DEFAULT = (0.70, 0.15, 0.15)
LAYOUTS = ["two_col_slash", "two_col_dmon", "signed_amount_iso"]
COMPANIES = [f"co-{i:04d}" for i in range(300)]


def test_allowed_split_names():
    assert set(SPLITS) == {"train", "validation", "test", "test_unseen_layout"}


def test_every_document_gets_one_allowed_split_no_holdout():
    for cid in COMPANIES:
        s = assign_split(42, cid, "two_col_slash", None, DEFAULT)
        assert s in ("train", "validation", "test")


def test_default_fractions_roughly_respected():
    splits = [assign_split(42, c, "two_col_slash", None, DEFAULT) for c in COMPANIES]
    share = {k: splits.count(k) / len(splits) for k in ("train", "validation", "test")}
    assert 0.60 < share["train"] < 0.80
    assert 0.07 < share["validation"] < 0.25
    assert 0.07 < share["test"] < 0.25


def test_company_split_is_deterministic_and_layout_independent():
    for cid in COMPANIES[:50]:
        got = {assign_split(42, cid, lay, None, DEFAULT) for lay in LAYOUTS}
        assert len(got) == 1
        assert assign_split(42, cid, "two_col_slash", None, DEFAULT) == \
            assign_split(42, cid, "two_col_slash", None, DEFAULT)


def test_split_does_not_depend_on_call_order():
    forward = {c: assign_split(7, c, "two_col_dmon", None, DEFAULT) for c in COMPANIES}
    backward = {c: assign_split(7, c, "two_col_dmon", None, DEFAULT) for c in reversed(COMPANIES)}
    assert forward == backward


def test_master_seed_changes_assignment():
    a = [assign_split(1, c, "two_col_slash", None, DEFAULT) for c in COMPANIES]
    b = [assign_split(2, c, "two_col_slash", None, DEFAULT) for c in COMPANIES]
    assert a != b


def test_fractions_all_to_one_split():
    assert all(assign_split(3, c, "two_col_slash", None, (1.0, 0.0, 0.0)) == "train"
               for c in COMPANIES[:50])
    assert all(assign_split(3, c, "two_col_slash", None, (0.0, 1.0, 0.0)) == "validation"
               for c in COMPANIES[:50])
    assert all(assign_split(3, c, "two_col_slash", None, (0.0, 0.0, 1.0)) == "test"
               for c in COMPANIES[:50])


def test_held_out_layout_iff_unseen():
    hold = "two_col_dmon"
    for cid in COMPANIES[:100]:
        for lay in LAYOUTS:
            s = assign_split(42, cid, lay, hold, DEFAULT)
            assert (s == "test_unseen_layout") == (lay == hold)
            assert s in SPLITS


def test_non_held_out_layouts_keep_their_company_split():
    for cid in COMPANIES[:100]:
        base = assign_split(42, cid, "two_col_slash", None, DEFAULT)
        assert assign_split(42, cid, "two_col_slash", "two_col_dmon", DEFAULT) == base


def test_no_unseen_split_without_holdout():
    assert all(assign_split(9, c, lay, None, DEFAULT) != "test_unseen_layout"
               for c in COMPANIES[:100] for lay in LAYOUTS)


@pytest.mark.parametrize("fractions", [DEFAULT, (0.5, 0.25, 0.25), (0.8, 0.1, 0.1005)])
def test_valid_fractions_accepted(fractions):
    validate_fractions(fractions)


@pytest.mark.parametrize(
    "fractions",
    [(0.7, 0.2, 0.2), (0.5, 0.2, 0.2), (0.0, 0.0, 0.0), (1.2, -0.1, -0.1), (-0.1, 0.6, 0.5)],
)
def test_bad_fractions_refused_with_message(fractions):
    with pytest.raises(ValueError) as exc:
        validate_fractions(fractions)
    assert "fraction" in str(exc.value).lower()


def test_assign_split_refuses_bad_fractions():
    with pytest.raises(ValueError):
        assign_split(1, "co-1", "two_col_slash", None, (0.9, 0.9, 0.9))


def test_missing_tamper_type_in_non_empty_split_is_a_gap():
    types = ["font_swap", "balance_edit", "uen_mismatch"]
    counts = {
        "train": {"font_swap": 3, "balance_edit": 2, "uen_mismatch": 1},
        "validation": {"font_swap": 1},
        "test": {},
    }
    gaps = find_gaps(types, counts)
    assert gaps["validation"] == ["balance_edit", "uen_mismatch"]
    assert "train" not in gaps or gaps["train"] == []


def test_empty_split_is_not_reported_as_gap():
    gaps = find_gaps(["font_swap"], {"train": {"font_swap": 1}, "test": {}})
    assert not gaps.get("test")
    assert not gaps.get("train")
