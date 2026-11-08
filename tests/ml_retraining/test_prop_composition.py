"""Task 6.5: property tests for corpus composition gates and subtle-tamper classification.

Property 18: Corpus composition gates.   **Validates: Requirements 5.5, 6.1, 6.2, 6.3**
Property 19: Subtle-tamper classification. **Validates: Requirements 6.6**
"""
from __future__ import annotations

import os
import tempfile
from decimal import Decimal

from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import corpus_gates as cg

SETTINGS = settings(max_examples=100, deadline=None)

ALL_SPLITS = ("train", "validation", "test", "unseen_layout")
TYPES = ("edited_amount", "altered_date", "structuring", "font_swap", "row_insert")


# ------------------------------------------------------------------ Property 18 strategies
@st.composite
def genuine_tables(draw):
    rows = []
    for i in range(draw(st.integers(0, 60))):
        rows.append({"doc_id": f"G{i}", "split": draw(st.sampled_from(ALL_SPLITS)),
                     "is_tampered": draw(st.booleans()),
                     "genuine_resaved": draw(st.booleans()), "tamper_types": ()})
    return rows


@st.composite
def tamper_tables(draw):
    rows = []
    for i in range(draw(st.integers(0, 80))):
        tampered = draw(st.booleans())
        types = draw(st.lists(st.sampled_from(TYPES), max_size=3, unique=True)) if tampered else []
        rows.append({"doc_id": f"D{i}", "corpus_key": draw(st.sampled_from(("s1", "s2"))),
                     "split": draw(st.sampled_from(ALL_SPLITS)), "is_tampered": tampered,
                     "genuine_resaved": False, "tamper_types": tuple(types)})
    return rows


@SETTINGS
@given(rows=genuine_tables(), min_share=st.sampled_from([5.0, 15.0, 30.0]))
def test_resaved_share_fails_iff_a_split_is_below_minimum(rows, min_share):
    totals, resaved = {}, {}
    for r in rows:
        if r["is_tampered"]:
            continue
        totals[r["split"]] = totals.get(r["split"], 0) + 1
        resaved[r["split"]] = resaved.get(r["split"], 0) + int(r["genuine_resaved"])
    failing = {s for s, n in totals.items() if resaved[s] * 100 < min_share * n}
    result = cg.resaved_share_gate(rows, min_share=min_share)
    assert result.passed == (not failing)
    assert len(result.offenders) == len(failing)
    for split in failing:
        line = [o for o in result.offenders if o.startswith(f"{split}:")]
        assert len(line) == 1
        assert f"{100.0 * resaved[split] / totals[split]:.1f}%" in line[0]


@SETTINGS
@given(rows=tamper_tables(), min_docs=st.integers(1, 8))
def test_type_minimum_lists_exactly_train_types_below_minimum(rows, min_docs):
    counts = {}
    for r in rows:
        if r["split"] == "train" and r["is_tampered"]:
            for t in r["tamper_types"]:
                counts[t] = counts.get(t, 0) + 1
    expected = {t: n for t, n in counts.items() if n < min_docs}
    before = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            result = cg.tamper_type_minimum_gate(rows, min_docs=min_docs)
            written = os.listdir(tmp)
        finally:
            os.chdir(before)
    listed = {}
    for line in result.offenders:
        name, rest = line.split(":", 1)
        listed[name] = int(rest.split()[0])
    assert len(result.offenders) == len(listed)
    assert listed == expected
    assert result.passed == (not expected)
    assert written == []  # no completion marker, nothing written


@SETTINGS
@given(rows=tamper_tables())
def test_coverage_lists_exactly_missing_type_split_pairs(rows):
    checked = ("train", "validation", "test")
    seen, types = set(), set()
    for r in rows:
        if r["is_tampered"] and r["split"] in checked:
            for t in r["tamper_types"]:
                types.add(t)
                seen.add((t, r["split"]))
    expected = {(t, s) for t in types for s in checked if (t, s) not in seen}
    gaps = cg.coverage_gaps(rows)
    assert set(gaps) == expected and len(gaps) == len(expected)
    result = cg.coverage_gate(rows)
    assert result.passed == (not expected)
    assert len(result.offenders) == len(expected)
    for t, s in expected:
        assert f"{t}: no documents in {s}" in result.offenders


# ------------------------------------------------------------------ Property 19 strategies
def _money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


amount_entry = st.builds(
    lambda a, b: {"field": "amount", "original": _money(a), "new": _money(b)},
    st.integers(1, 10_000_00), st.integers(1, 10_000_00))
near_entry = st.builds(  # amounts clustered around the 5% boundary
    lambda a, d: {"field": "amount", "original": _money(a), "new": _money(max(1, a + d))},
    st.integers(100, 100_00), st.integers(-8_00, 8_00))
same_entry = st.builds(
    lambda a, f: {"field": f, "original": _money(a), "new": _money(a)} if f == "amount"
    else {"field": f, "original": "2026-01-05", "new": "2026-01-05"},
    st.integers(1, 10_000), st.sampled_from(("amount", "date")))
date_entry = st.builds(
    lambda d1, d2: {"field": "date", "original": f"2026-01-{d1:02d}", "new": f"2026-01-{d2:02d}"},
    st.integers(1, 28), st.integers(1, 28))
other_entry = st.just({"field": "row", "original": None, "new": "10.00"})
entries = st.lists(st.one_of(amount_entry, near_entry, same_entry, date_entry, other_entry), max_size=5)
limit = st.sampled_from([0.01, 0.05, 0.10])


def _oracle(rows, limit_):
    amounts = [e for e in rows if e["field"] == "amount" and Decimal(e["original"]) != Decimal(e["new"])]
    dates = [e for e in rows if e["field"] == "date" and e["original"] != e["new"]]
    cats = set()
    if len(amounts) == 1:
        cats.add("single_amount")
        a, b = Decimal(amounts[0]["original"]), Decimal(amounts[0]["new"])
        if abs(b - a) / abs(a) <= Decimal(str(limit_)):
            cats.add("amount_le_5pct")
    if len(dates) == 1:
        cats.add("single_date")
    return frozenset(cats)


@SETTINGS
@given(rows=entries, limit_=limit)
def test_classification_matches_independent_oracle(rows, limit_):
    assert cg.classify_subtle(rows, limit_) == _oracle(rows, limit_)


@SETTINGS
@given(rows=entries, limit_=limit)
def test_classification_ignores_order(rows, limit_):
    assert cg.classify_subtle(list(reversed(rows)), limit_) == cg.classify_subtle(rows, limit_)


@SETTINGS
@given(a=st.integers(100, 100_00), pct=st.integers(0, 20), up=st.booleans())
def test_boundary_five_percent_is_inclusive(a, pct, up):
    a = a * 100  # whole multiples so pct percent is an exact number of cents
    delta = a * pct // 100
    new = a + delta if up else max(1, a - delta)
    row = [{"field": "amount", "original": _money(a), "new": _money(new)}]
    cats = cg.classify_subtle(row, 0.05)
    if new == a:  # no change: no category at all
        assert cats == frozenset()
    else:
        assert "single_amount" in cats
        assert ("amount_le_5pct" in cats) == (abs(new - a) * 100 <= 5 * a)


@st.composite
def tampered_docs(draw):
    docs = []
    for i in range(draw(st.integers(0, 12))):
        docs.append({"doc_id": f"D{i}", "split": draw(st.sampled_from(ALL_SPLITS)),
                     "is_tampered": draw(st.booleans()), "affected_rows": draw(entries)})
    return docs


@SETTINGS
@given(docs=tampered_docs(), limit_=limit)
def test_subtle_gate_lists_exactly_missing_categories_in_tampered_splits(docs, limit_):
    present = {}
    for d in docs:
        if d["is_tampered"]:
            present.setdefault(d["split"], set()).update(_oracle(d["affected_rows"], limit_))
    expected = {f"{s}: no document with {c}" for s, cats in present.items()
                for c in cg.SUBTLE_CATEGORIES if c not in cats}
    result = cg.subtle_gate(docs, max_relative_change=limit_)
    assert set(result.offenders) == expected and len(result.offenders) == len(expected)
    assert result.passed == (not expected)
    assert result.waivable is True
