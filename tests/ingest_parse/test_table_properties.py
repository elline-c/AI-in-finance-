"""Property test for the table reader (Requirement 5).

Feature: multi-layout-ingestion-and-fabrication-rules, Property 5: Non-transaction rows are excluded
"""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from fraud_detect.nodes.ingest_parse.table import find_table, read_rows
from fraud_detect.nodes.ingest_parse.types import Word

from .fixtures import ROW_STEP, _left, _right

# Words that cannot look like dates, money, currency prefixes or noise keywords.
_VOCAB = [
    "GIRO", "RENT", "NORTHBROOK", "HOLDINGS", "FAST", "TESSELATE", "RETAIL", "INV1001",
    "NETS/CARD", "QUILLFEATHER", "CAFE", "1234", "PAYNOW", "ORCHARDLEIGH", "TRADING",
    "LUMENWAVE", "MOBILE", "REF200001", "SALARY", "PAYROLL", "CPF", "BRINKWATER", "SUPPLIES",
]
_desc = st.lists(st.sampled_from(_VOCAB), min_size=1, max_size=4)
_cents = st.integers(min_value=1, max_value=99_999_999)
_NOISE_KINDS = ["bf", "bf_dated", "closing", "total", "page", "specimen", "fictional", "header"]

_txn = st.fixed_dictionaries({
    "kind": st.just("txn"),
    "day": st.integers(1, 28),
    "month": st.integers(1, 12),
    "desc": _desc,
    "out": st.booleans(),
    "amount": _cents,
    "balance": _cents,
    "wraps": st.lists(_desc, max_size=2),
})
_noise = st.fixed_dictionaries({
    "kind": st.sampled_from(_NOISE_KINDS), "amount": _cents, "balance": _cents,
})
_items = st.lists(st.one_of(_txn, _noise), min_size=1, max_size=15)

_TWO = {
    "header": [("Date", 40.0, 57.8), ("Description", 100.0, 145.1),
               ("Withdrawals", 351.7, 400.0), ("Deposits", 440.4, 475.0), ("Balance", 523.6, 555.0)],
    "date_x0": 40.0, "desc_x0": 100.0, "out_x1": 400.0, "in_x1": 475.0, "bal_x1": 555.0,
}
_SINGLE = {
    "header": [("Date", 40.0, 60.2), ("Details", 110.0, 140.0), ("Amount", 416.5, 450.0),
               ("Balance", 521.3, 555.0)],
    "date_x0": 40.0, "desc_x0": 110.0, "amount_x1": 450.0, "bal_x1": 555.0,
}


def _money(cents: int) -> str:
    return f"{cents / 100:,.2f}"


def _build(items, two_column: bool):
    g = _TWO if two_column else _SINGLE
    words: list[Word] = []
    expected = []
    top = 100.0

    def header():
        nonlocal top
        words.extend(Word(t, x0, x1, top) for t, x0, x1 in g["header"])
        top += ROW_STEP

    def line(*groups):
        nonlocal top
        words.extend(w for grp in groups for w in grp)
        top += ROW_STEP

    def date_tokens(day, month):
        return [f"{day:02d}/{month:02d}/2026"] if two_column else [f"2026-{month:02d}-{day:02d}"]

    def amount_group(cents, out, t):
        if two_column:
            return _right([_money(cents)], g["out_x1"] if out else g["in_x1"], t)
        return _right([("-" if out else "+") + _money(cents)], g["amount_x1"], t)

    header()
    for it in items:
        k, t = it["kind"], top
        if k == "txn":
            line(_left(date_tokens(it["day"], it["month"]), g["date_x0"], t),
                 _left(it["desc"], g["desc_x0"], t),
                 amount_group(it["amount"], it["out"], t),
                 _right([_money(it["balance"])], g["bal_x1"], t))
            desc = list(it["desc"])
            for wrap in it["wraps"]:
                line(_left(wrap, g["desc_x0"], top))
                desc += wrap
            expected.append((" ".join(desc), it["amount"], it["balance"], it["out"]))
        elif k == "bf":
            line(_left(["BALANCE", "BROUGHT", "FORWARD"], g["desc_x0"], t),
                 _right([_money(it["balance"])], g["bal_x1"], t))
        elif k == "bf_dated":
            line(_left(date_tokens(1, 1), g["date_x0"], t),
                 _left(["BALANCE", "BROUGHT", "FORWARD"], g["desc_x0"], t),
                 amount_group(it["amount"], False, t),
                 _right([_money(it["balance"])], g["bal_x1"], t))
        elif k == "closing":
            line(_left(["CLOSING", "BALANCE"], g["desc_x0"], t),
                 _right([_money(it["balance"])], g["bal_x1"], t))
        elif k == "total":
            line(_left(["TOTAL"], g["desc_x0"], t),
                 amount_group(it["amount"], True, t),
                 _right([_money(it["balance"])], g["bal_x1"], t))
        elif k == "page":
            line(_left(["Page", "1", "of", "1"], 40.0, t))
        elif k == "specimen":
            line(_left(["SPECIMEN", "Synthetic", "data", "for", "ML", "research"], 40.0, t))
        elif k == "fictional":
            line(_left(["Corvane", "Trust", "is", "a", "fictional", "entity", "Ref", "FIX-001"], 40.0, t))
        else:  # repeated table header
            header()
    return words, expected


@settings(max_examples=100, deadline=None)
@given(items=_items, two_column=st.booleans())
def test_non_transaction_rows_are_excluded(items, two_column):
    # Feature: multi-layout-ingestion-and-fabrication-rules, Property 5: Non-transaction rows are excluded
    # **Validates: Requirements 5.5, 5.9**
    words, expected = _build(items, two_column)
    layout = find_table(words, None)
    assert layout is not None
    assert layout.kind == ("two_column" if two_column else "single_signed")

    rows = read_rows(words, layout, None)

    # exactly the generated transaction rows, once each, in order
    assert len(rows) == len(expected)
    for row, (desc, amount, balance, out) in zip(rows, expected):
        # wrapped lines appended once; no noise text leaked into the description
        assert row.description == desc
        assert len(row.amount_cells) == 1
        assert row.amount_cells[0].money.value == amount / 100
        assert row.balance.value == balance / 100
        if two_column:
            assert row.amount_cells[0].column == ("money_out" if out else "money_in")
        low = (row.description + " " + row.line_text).lower()
        for bad in ("brought forward", "closing balance", "total", "page ", "specimen", "fictional"):
            assert bad not in low
