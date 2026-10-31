"""Task 4.3: property tests for Rule A, ``check_duplicate_invoice_refs``.

Uses hypothesis. Properties 2, 3, 4, 5 and 7 of the node3-fabrication-rules design.
"""
from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given, settings, strategies as st

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    FabricationConfig,
    check_duplicate_invoice_refs,
    extract_credit_refs,
    find_order_breaks,
)

CFG = DEFAULT_CONFIG
SETTINGS = settings(max_examples=100, deadline=None)
Y0 = date(2026, 1, 1)

_RESERVED = ("INV", "GIRO", "FAST", "FROM", "PAYNOW", "TT")
names = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=4, max_size=8).filter(
    lambda s: not any(w in s for w in _RESERVED)
)
refs = st.integers(min_value=100, max_value=99999)  # 3-5 digits, no leading zero
amounts = st.integers(min_value=1, max_value=50_000_000).map(lambda c: c / 100)
days = st.integers(min_value=0, max_value=364).map(lambda n: Y0 + timedelta(days=n))
pages = st.integers(min_value=1, max_value=6)

NOISE_DESCRIPTIONS = ("SALARY PAYROLL", "OFFICE RENT", "Customer Payment INV", "FAST FROM SOMEONE")


def _desc(name, ref):
    return f"FAST FROM {name} INV{ref}"


@st.composite
def scenario(draw, min_credits=1, max_credits=12, with_undated=False):
    """Unique (customer, ref) credits plus debit and no-reference noise.

    Returns (txns in row order, credit Transactions, free row numbers, names used).
    """
    pairs = draw(st.lists(st.tuples(names, refs), min_size=min_credits, max_size=max_credits, unique=True))
    n_noise = draw(st.integers(0, 5))
    # noise: debits reusing a real pair, or rows with no reference
    noise = []
    for _ in range(n_noise):
        kinds = ["plain_credit", "plain_debit"] + (["debit_ref"] if pairs else [])
        kind = draw(st.sampled_from(kinds))
        if kind == "debit_ref":
            noise.append(("debit", draw(st.sampled_from(pairs))))
        else:
            noise.append((kind, draw(st.sampled_from(NOISE_DESCRIPTIONS))))
    total = len(pairs) + len(noise)
    rows = sorted(draw(st.lists(st.integers(1, 300), min_size=total + 3, max_size=total + 3, unique=True)))
    free = rows[total:]  # spare row numbers for copies (not necessarily above the others)
    rows = rows[:total]
    order = draw(st.permutations(list(range(total))))
    items = [("credit", p) for p in pairs] + noise
    date_st = st.one_of(days, st.none()) if with_undated else days
    txns = []
    for slot, item_idx in enumerate(order):
        kind, payload = items[item_idx]
        row = rows[slot]
        d = draw(date_st)
        page = draw(pages)
        amt = draw(amounts)
        if kind == "credit":
            desc, signed = _desc(*payload), amt
        elif kind == "debit":
            desc, signed = _desc(*payload), -amt
        elif kind == "plain_credit":
            desc, signed = payload, amt
        else:
            desc, signed = payload, -amt
        txns.append(Transaction(index=row, date=d, description=desc, amount=signed, page=page))
    txns.sort(key=lambda t: t.index)
    credits = [t for t, in ((t,) for t in txns) if t.amount > 0 and "INV" in t.description and any(
        t.description == _desc(*p) for p in pairs)]
    return txns, credits, free, {n for n, _ in pairs}


def _copy_of(src, row, d, amt, page, name=None, credit=True):
    desc = src.description if name is None else _desc(name, src.description.rsplit("INV", 1)[1])
    return Transaction(index=row, date=d, description=desc, amount=amt if credit else -amt, page=page)


def _rows(field: str) -> list[int]:
    return [int(x) for x in field.removeprefix("rows ").split(", ")]


def _check(txns, cfg=CFG, likely=frozenset()):
    return check_duplicate_invoice_refs(txns, cfg, likely)


# Feature: node3-fabrication-rules, Property 2: Duplicate reference detection
# **Validates: Requirements 3.1, 3.2, 3.3**
@SETTINGS
@given(data=st.data(), sc=scenario(), amt=amounts, d=days, page=pages)
def test_property_2_copy_is_detected_and_lookalikes_are_not(data, sc, amt, d, page):
    txns, credits, free, used = sc
    src = data.draw(st.sampled_from(credits))
    new_row = free[0]

    # a copy: same customer & ref, any other amount/date
    out = _check(txns + [_copy_of(src, new_row, d, amt, page)])
    assert len(out) == 1
    f = out[0]
    assert f.code == "FL_DUP_INVOICE_REF"
    assert {src.index, new_row} <= set(_rows(f.field))

    # same ref under a different customer: nothing
    other = data.draw(names.filter(lambda n: n not in used))
    assert _check(txns + [_copy_of(src, new_row, d, amt, page, name=other)]) == []

    # same customer & ref but on a debit: nothing
    assert _check(txns + [_copy_of(src, new_row, d, amt, page, credit=False)]) == []


# Feature: node3-fabrication-rules, Property 3: Unique references are quiet
# **Validates: Requirements 3.4, 9.4**
@SETTINGS
@given(sc=scenario(min_credits=0))
def test_property_3_unique_pairs_give_no_findings(sc):
    txns, _, _, _ = sc
    assert _check(txns) == []


# Feature: node3-fabrication-rules, Property 3: Unique references are quiet
# **Validates: Requirements 3.4, 9.4**
@SETTINGS
@given(
    items=st.lists(st.tuples(st.sampled_from(NOISE_DESCRIPTIONS), st.booleans(), amounts, days), max_size=15)
)
def test_property_3_no_references_give_no_findings(items):
    txns = [
        Transaction(index=i + 1, date=d, description=desc, amount=a if credit else -a, page=1)
        for i, (desc, credit, a, d) in enumerate(items)
    ]
    assert _check(txns) == []


@st.composite
def grouped(draw, with_undated=False):
    """A scenario with k disjoint repeated-reference groups (each of size 2 or 3)."""
    txns, credits, free, _ = draw(scenario(min_credits=1, max_credits=10, with_undated=with_undated))
    k = draw(st.integers(1, len(credits)))
    sources = draw(st.lists(st.sampled_from(credits), min_size=k, max_size=k, unique_by=lambda t: t.index))
    used = {t.index for t in txns}
    next_row = max(used) + 1
    date_st = st.one_of(days, st.none()) if with_undated else days
    added = []
    for src in sources:
        for _ in range(draw(st.integers(1, 2))):
            added.append(Transaction(
                index=next_row, date=draw(date_st), description=src.description,
                amount=draw(amounts), page=draw(pages)))
            next_row += 1
    all_txns = sorted(txns + added, key=lambda t: t.index)
    return all_txns, k


def _group_rows(txns):
    by_desc = {}
    for t in txns:
        if t.amount > 0 and t.description.startswith("FAST FROM ") and "INV" in t.description:
            by_desc.setdefault(t.description, []).append(t.index)
    return [sorted(v) for v in by_desc.values() if len(v) >= 2]


# Feature: node3-fabrication-rules, Property 4: Severity follows the number of repeated references
# **Validates: Requirements 4.1, 4.2**
@SETTINGS
@given(g=grouped(), threshold=st.integers(1, 8))
def test_property_4_severity_follows_group_count(g, threshold):
    txns, k = g
    cfg = FabricationConfig(dup_escalate_min_refs=threshold)
    out = _check(txns, cfg)
    groups = _group_rows(txns)
    assert len(groups) == k
    if k < threshold:
        assert len(out) == k
        assert all(f.severity == Severity.MEDIUM and f.weight == cfg.dup_isolated_weight for f in out)
        assert sorted(_rows(f.field) for f in out) == sorted(groups)
    else:
        assert len(out) == 1
        f = out[0]
        assert f.severity == Severity.HIGH and f.weight == cfg.dup_escalated_weight
        assert set(_rows(f.field)) == {r for grp in groups for r in grp}


# Feature: node3-fabrication-rules, Property 5: Evidence names every occurrence
# **Validates: Requirements 5.2, 5.3, 5.4**
@SETTINGS
@given(g=grouped(with_undated=True), threshold=st.integers(1, 8), max_groups=st.integers(1, 8))
def test_property_5_evidence_names_every_occurrence(g, threshold, max_groups):
    txns, k = g
    cfg = FabricationConfig(dup_escalate_min_refs=threshold, evidence_max_groups=max_groups)
    out = _check(txns, cfg)
    by_row = {t.index: t for t in txns}
    groups = sorted(_group_rows(txns), key=lambda grp: grp[0])
    assert out
    for f in out:
        rows = _rows(f.field)
        assert rows == sorted(set(rows))
        assert f.page == by_row[rows[-1]].page
        if f.severity == Severity.MEDIUM:
            shown = [grp for grp in groups if set(grp) <= set(rows)]
        else:
            shown = groups[:max_groups]
            assert set(rows) == {r for grp in groups for r in grp}
        assert shown
        for grp in shown:
            for r in grp:
                t = by_row[r]
                when = t.date.strftime("%d %b %Y") if t.date is not None else "no date"
                assert when in f.evidence
                assert f"{t.amount:,.2f}" in f.evidence
                assert f"(page {t.page}, row {r})" in f.evidence


# Feature: node3-fabrication-rules, Property 7: Likelier copy is named only when unambiguous
# **Validates: Requirements 7.1, 7.2**
@SETTINGS
@given(data=st.data(), sc=scenario(), extra=st.integers(1, 2))
def test_property_7_likelier_copy_only_when_unambiguous(data, sc, extra):
    txns, credits, free, _ = sc
    src = data.draw(st.sampled_from(credits))
    copies = [
        Transaction(index=free[i], date=data.draw(days), description=src.description,
                    amount=data.draw(amounts), page=data.draw(pages))
        for i in range(extra)
    ]
    all_txns = sorted(txns + copies, key=lambda t: t.index)
    group = sorted([src.index] + [c.index for c in copies])
    others = [t.index for t in all_txns if t.index not in group]
    pool = group + others
    likely = frozenset(data.draw(st.lists(st.sampled_from(pool), unique=True, max_size=len(pool))))

    (f,) = _check(all_txns, likely=likely)
    flagged = [r for r in group if r in likely]
    named = [r for r in group if f"row {r} breaks the invoice number order, so it is the likelier copy" in f.evidence]
    if len(flagged) == 1:
        assert named == flagged
    else:
        assert named == []
        assert "likelier" not in f.evidence.lower()

    # same rule with the real Order_Break_Rows
    breaks = find_order_breaks(extract_credit_refs(all_txns, CFG), CFG)
    (g,) = _check(all_txns, likely=breaks)
    in_group = [r for r in group if r in breaks]
    got = [r for r in group if f"row {r} breaks the invoice number order" in g.evidence]
    assert got == (in_group if len(in_group) == 1 else [])
