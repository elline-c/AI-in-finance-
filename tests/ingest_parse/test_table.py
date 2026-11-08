"""Tests for the table reader (task 5.1; Requirement 5).

Written BEFORE ``fraud_detect/nodes/ingest_parse/table.py`` exists (task 5.2),
so every test here is expected to fail with ``ModuleNotFoundError`` for now.

API contract assumed by these tests (compatible with design.md Component 4):

* ``find_table(page_words, profile=None) -> TableLayout | None``
* ``read_rows(page_words, layout, profile=None) -> list[RawRow]``
* ``profile`` is OPTIONAL and ignored by these tests (profiles arrive in
  task 7); we always pass ``None``. ``layout`` may be ``None`` when no table
  header was found, which selects the line-based fallback.
* ``TableLayout.columns`` maps ``"money_out"``, ``"money_in"``, ``"amount"``
  and ``"balance"`` to ``(x0, x1)``; two-word headers such as ``Money out`` are
  joined into one column. ``TableLayout.kind`` is ``"two_column"`` or
  ``"single_signed"``.
* ``RawRow`` has ``date_text``, ``value_date_text``, ``description``,
  ``amount_cells``, ``balance`` (a ``Money``) and ``line_text``.
* ``AmountCell`` has ``money`` (a ``Money``), ``x0``, ``x1`` and ``column``
  (the column key above, or ``None`` when there is no table layout).
"""
from __future__ import annotations

import pytest

from fraud_detect.nodes.ingest_parse.table import find_table, read_rows
from fraud_detect.nodes.ingest_parse.types import Word

from .fixtures import (
    ALL_FIXTURES,
    continuation_fixture,
    signed_amount_iso_fixture,
    six_page_page1of1_fixture,
    two_col_dmon_fixture,
    two_col_dual_date_fixture,
    two_col_slash_fixture,
    two_col_ddmmyy_cr_fixture,
)

NOISE = ("BROUGHT", "CLOSING", "SPECIMEN", "fictional", "Page ", "Ref FIX")


def _read_page(page):
    layout = find_table(page.words, None)
    assert layout is not None
    return layout, read_rows(page.words, layout, None)


def _read_fixture(fx):
    rows = []
    for page in fx.pages:
        rows.extend(_read_page(page)[1])
    return rows


def _expected(fx):
    return [t for s in fx.sections for t in s.txns]


# ----- header / column detection -------------------------------------------
@pytest.mark.parametrize(
    "build, out_key, in_key, out_x1, in_x1",
    [
        (two_col_slash_fixture, "money_out", "money_in", 400.0, 475.0),  # Withdrawals/Deposits
        (two_col_dual_date_fixture, "money_out", "money_in", 420.0, 490.0),  # Debit/Credit
        (two_col_ddmmyy_cr_fixture, "money_out", "money_in", 395.0, 470.0),  # DEBIT/CREDIT
        (two_col_dmon_fixture, "money_out", "money_in", 392.0, 472.0),  # Money out / Money in
    ],
)
def test_two_column_header_names_and_right_edges(build, out_key, in_key, out_x1, in_x1):
    layout = find_table(build().pages[0].words, None)
    assert layout is not None and layout.kind == "two_column"
    assert layout.columns[out_key][1] == out_x1
    assert layout.columns[in_key][1] == in_x1
    assert "balance" in layout.columns


def test_single_amount_column():
    layout = find_table(signed_amount_iso_fixture().pages[0].words, None)
    assert layout is not None and layout.kind == "single_signed"
    assert layout.columns["amount"][1] == 450.0


def test_no_header_means_no_layout():
    page = two_col_slash_fixture().pages[0]
    words = [w for w in page.words if w.top != 183]
    assert find_table(words, None) is None


# ----- right-edge column assignment ----------------------------------------
def test_amounts_assigned_by_right_edge_layout1():
    fx = two_col_slash_fixture()
    rows = _read_fixture(fx)
    for row, t in zip(rows, _expected(fx)):
        assert len(row.amount_cells) == 1
        cell = row.amount_cells[0]
        assert cell.x1 == (475.0 if t.amount > 0 else 400.0)
        assert cell.column == ("money_in" if t.amount > 0 else "money_out")
        assert cell.money.value == pytest.approx(abs(t.amount))


def test_narrow_amount_still_goes_by_right_edge():
    # Short debit "5.00" is far from the header's left edge but ends at 400.
    page = two_col_slash_fixture().pages[0]
    words = [w for w in page.words if w.top < 205]  # header block + table header + b/f
    words += [
        Word("01/07/2026", 40.0, 87.0, 212.4),
        Word("TINY", 100.0, 119.0, 212.4),
        Word("5.00", 380.0, 400.0, 212.4),
        Word("9,995.00", 520.0, 555.0, 212.4),
    ]
    layout = find_table(words, None)
    rows = read_rows(words, layout, None)
    assert len(rows) == 1
    assert rows[0].amount_cells[0].column == "money_out"


# ----- full-fixture reads: every row once, noise excluded ------------------
@pytest.mark.parametrize("name", list(ALL_FIXTURES))
def test_each_printed_row_read_exactly_once(name):
    fx = ALL_FIXTURES[name]()
    rows = _read_fixture(fx)
    exp = _expected(fx)
    assert len(rows) == len(exp)
    assert [r.description for r in rows] == [t.description for t in exp]
    assert [r.balance.value for r in rows] == pytest.approx([t.balance for t in exp])


@pytest.mark.parametrize("name", list(ALL_FIXTURES))
def test_summary_and_footer_lines_excluded(name):
    for row in _read_fixture(ALL_FIXTURES[name]()):
        text = row.description + " " + row.line_text
        assert not any(n in text for n in NOISE), text


def test_signed_single_amount_values():
    fx = signed_amount_iso_fixture()
    rows = _read_fixture(fx)
    for row, t in zip(rows, _expected(fx)):
        assert len(row.amount_cells) == 1
        assert row.amount_cells[0].column == "amount"
        assert row.amount_cells[0].money.value == pytest.approx(abs(t.amount))


def test_layout4_cr_suffix_balance():
    fx = two_col_ddmmyy_cr_fixture()
    rows = _read_fixture(fx)
    assert [r.balance.value for r in rows] == pytest.approx([t.balance for t in _expected(fx)])


def test_dmon_money_out_in_columns():
    fx = two_col_dmon_fixture()
    for row, t in zip(_read_fixture(fx), _expected(fx)):
        assert row.amount_cells[0].column == ("money_in" if t.amount > 0 else "money_out")
        assert row.date_text == f"{t.date.day} {t.date:%b}"


# ----- value date -----------------------------------------------------------
def test_dual_date_value_date_stored_separately():
    fx = two_col_dual_date_fixture()
    rows = _read_fixture(fx)
    assert all(r.value_date_text for r in rows)
    first = rows[0]
    assert first.date_text == "02 Jul 2026"
    assert first.value_date_text == "02 Jul 2026"
    assert rows[1].date_text == "04 Jul 2026"
    assert rows[3].date_text == "09 Jul 2026"
    assert rows[3].value_date_text == "10 Jul 2026"  # odd day: value date +1
    assert "Jul" not in rows[3].description


def test_single_date_layout_has_no_value_date():
    assert all(r.value_date_text is None for r in _read_fixture(two_col_slash_fixture()))


# ----- wrapped description --------------------------------------------------
def test_wrapped_description_appended_to_previous_row():
    fx = two_col_slash_fixture()
    page = fx.pages[0]
    words = list(page.words) + [Word("EXTRA", 100.0, 123.5, 230.0), Word("DETAIL", 126.0, 154.0, 230.0)]
    layout = find_table(words, None)
    rows = read_rows(words, layout, None)
    exp = _expected(fx)
    assert len(rows) == len(_read_page(page)[1])  # wrapped line is not a new row
    assert rows[1].description == exp[1].description + " EXTRA DETAIL"
    assert rows[2].description == exp[2].description


# ----- continuation page ----------------------------------------------------
def test_repeated_header_on_continuation_page():
    fx = continuation_fixture()
    assert len(fx.pages) == 2
    layout1, rows1 = _read_page(fx.pages[0])
    layout2, rows2 = _read_page(fx.pages[1])
    assert layout2.columns["money_out"][1] == layout1.columns["money_out"][1] == 400.0
    assert len(rows1) == 4 and len(rows2) == 3
    exp = _expected(fx)
    assert [r.description for r in rows1 + rows2] == [t.description for t in exp]


def test_six_pages_page1of1_each_read_once():
    fx = six_page_page1of1_fixture()
    for page, sec in zip(fx.pages, fx.sections):
        rows = _read_page(page)[1]
        assert len(rows) == len(sec.txns)
        assert not any("Page" in r.description for r in rows)


# ----- no header: line-based fallback --------------------------------------
def test_no_header_falls_back_to_line_reading():
    fx = two_col_slash_fixture()
    words = [w for w in fx.pages[0].words if w.top != 183]
    assert find_table(words, None) is None
    rows = read_rows(words, None, None)
    exp = _expected(fx)
    assert len(rows) == len(exp)
    assert [r.description for r in rows] == [t.description for t in exp]
    assert [r.balance.value for r in rows] == pytest.approx([t.balance for t in exp])
    assert all(len(r.amount_cells) == 1 for r in rows)
    assert [abs(r.amount_cells[0].money.value) for r in rows] == pytest.approx(
        [abs(t.amount) for t in exp]
    )
