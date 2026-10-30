"""PDF-render property tests for the multi-layout ingestion (task 9.3).

Small fictional bank statements are drawn with reportlab in several supported
layouts, written to a temporary directory inside the test body, and ingested
through ``fraud_detect.nodes.ingestion.ingest``.

Feature: multi-layout-ingestion-and-fabrication-rules
"""
from __future__ import annotations

import calendar
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from fraud_detect.nodes import financial_logic
from fraud_detect.nodes.ingestion import ingest

PAGE_H = A4[1]
FONT = ("Helvetica", 8)
SETTINGS = settings(
    max_examples=30, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# --------------------------------------------------------------------------
# layout definitions (geometry follows tests/ingest_parse/fixtures.py)
# --------------------------------------------------------------------------
def _num(cents: int) -> str:
    return f"{abs(cents) / 100:,.2f}"


@dataclass(frozen=True)
class Layout:
    name: str
    bank: str
    fmt_date: Callable[[date], str]
    row_date: Callable[[date], str]
    date_x0: float
    desc_x0: float
    header: tuple  # (text, x, "l"|"r")
    out_x1: Optional[float]
    in_x1: Optional[float]
    amount_x1: Optional[float]
    bal_x1: float
    money: Callable[[int], str]  # unsigned money text
    signed_amount: Optional[Callable[[int], str]] = None  # signed layout only


LAYOUTS = {
    # two money columns, "d Mon" row dates (statement_E / dataset layout 5)
    "two_col_dmon": Layout(
        "two_col_dmon", "Vellichor Bank", lambda d: f"{d:%d %b %Y}",
        lambda d: f"{d.day} {d:%b}", 38.0, 90.0,
        (("Date", 38.0, "l"), ("Description", 90.0, "l"), ("Money out", 392.0, "r"),
         ("Money in", 472.0, "r"), ("Balance", 557.0, "r")),
        392.0, 472.0, None, 557.0, lambda c: "S$" + _num(c)),
    # one signed amount column, ISO dates (dataset layout 3)
    "signed_amount_iso": Layout(
        "signed_amount_iso", "Aldermoor Savings", lambda d: d.isoformat(),
        lambda d: d.isoformat(), 40.0, 110.0,
        (("Date", 40.0, "l"), ("Details", 110.0, "l"), ("Amount", 450.0, "r"),
         ("Balance", 555.0, "r")),
        None, None, 450.0, 555.0, lambda c: "SGD " + _num(c),
        lambda c: ("+" if c > 0 else "-") + "SGD " + _num(c)),
    # two money columns, dd/mm/yyyy dates (dataset layout 1)
    "two_col_slash": Layout(
        "two_col_slash", "Corvane Trust", lambda d: d.strftime("%d/%m/%Y"),
        lambda d: d.strftime("%d/%m/%Y"), 40.0, 100.0,
        (("Date", 40.0, "l"), ("Description", 100.0, "l"), ("Withdrawals", 400.0, "r"),
         ("Deposits", 475.0, "r"), ("Balance", 555.0, "r")),
        400.0, 475.0, None, 555.0, lambda c: _num(c)),
}
TWO_LAYOUTS = ("two_col_dmon", "signed_amount_iso")


@dataclass(frozen=True)
class Txn:
    day: int
    description: str
    cents: int  # signed: credits +, debits -


@dataclass
class Month:
    year: int
    month: int
    opening: int  # cents
    txns: list  # list[Txn]
    # filled by _replay
    balances: Optional[list] = None
    closing: int = 0

    @property
    def start(self) -> date:
        return date(self.year, self.month, 1)

    @property
    def end(self) -> date:
        return date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])


def _replay(months: list) -> None:
    prev = None
    for m in months:
        if prev is not None:
            m.opening = prev.closing
        bal, out = m.opening, []
        for t in m.txns:
            bal += t.cents
            out.append(bal)
        m.balances, m.closing = out, bal
        prev = m


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
class _Canvas:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.c.setFont(*FONT)

    def left(self, x: float, top: float, text: str) -> None:
        self.c.drawString(x, PAGE_H - top, text)

    def right(self, x1: float, top: float, text: str) -> None:
        self.c.drawRightString(x1, PAGE_H - top, text)

    def table_header(self, lay: Layout, top: float) -> None:
        for text, x, side in lay.header:
            (self.left if side == "l" else self.right)(x, top, text)

    def next_page(self) -> None:
        self.c.showPage()
        self.c.setFont(*FONT)


def _bal_cell(cv: _Canvas, lay: Layout, top: float, cents: int) -> None:
    cv.right(lay.bal_x1, top, lay.money(cents))


def _txn_row(cv: _Canvas, lay: Layout, d: date, desc: str, printed: int, bal: int,
             top: float) -> None:
    cv.left(lay.date_x0, top, lay.row_date(d))
    cv.left(lay.desc_x0, top, desc)
    if lay.amount_x1 is not None:
        cv.right(lay.amount_x1, top, lay.signed_amount(printed))
    else:
        cv.right(lay.in_x1 if printed > 0 else lay.out_x1, top, lay.money(abs(printed)))
    _bal_cell(cv, lay, top, bal)


def _page_head(cv: _Canvas, lay: Layout, page_text: str) -> None:
    cv.left(40, 30, "STATEMENT OF ACCOUNT")
    cv.left(40, 44, lay.bank)
    cv.left(40, 56, page_text)
    cv.left(40, 66, "SPECIMEN")


def _footer(cv: _Canvas, lay: Layout) -> None:
    cv.left(40, 770, "SPECIMEN - Fictional institution - Synthetic data - Not a real bank document")
    cv.left(40, 782, f"{lay.bank} is a fictional entity created for testing. Ref FIX-001")


def render(path: Path, lay: Layout, months: list, *, holder: str, account: str, uen: str,
           rows_first_page: Optional[int] = None, page1of1: bool = False,
           tamper: Optional[tuple] = None) -> None:
    """Draw one page (or two, on spill) per month.

    ``tamper`` = (month_index, row_index, printed_cents): the printed amount of
    that row is replaced, the printed balance column is left as computed.
    """
    cv = _Canvas(path)
    first_page = True
    for mi, m in enumerate(months):
        printed = [t.cents for t in m.txns]
        if tamper and tamper[0] == mi:
            printed[tamper[1]] = tamper[2]
        n = len(m.txns)
        k = n if rows_first_page is None else min(n, rows_first_page)
        n_pages = 2 if k < n else 1
        if not first_page:
            cv.next_page()
        first_page = False
        _page_head(cv, lay, "Page 1 of 1" if page1of1 else f"Page 1 of {n_pages}")
        left = [holder, "3 FICTIONAL DRIVE", "#10-24", "SINGAPORE 990001", "BUSINESS CURRENT ACCOUNT"]
        right = [f"Account No. {account}", f"Company UEN {uen}",
                 f"Statement Period {lay.fmt_date(m.start)} to {lay.fmt_date(m.end)}",
                 f"Statement Date {lay.fmt_date(m.end + timedelta(days=1))}", "Currency SGD"]
        for i, (a, b) in enumerate(zip(left, right)):
            top = 78 + 12 * i
            cv.left(40, top, a)
            cv.left(300, top, b)
        # summary box
        deb = -sum(c for c in (t.cents for t in m.txns) if c < 0)
        cred = sum(c for c in (t.cents for t in m.txns) if c > 0)
        xs0, xs1 = (40.0, 170.8, 301.6, 432.5), (160.8, 291.6, 422.5, 553.3)
        for x, lab in zip(xs0, ("Opening Balance", "Total Withdrawals", "Total Deposits",
                                "Closing Balance")):
            cv.left(x, 142, lab)
        for x1, v in zip(xs1, (m.opening, deb, cred, m.closing)):
            cv.right(x1, 154, lay.money(v))
        cv.table_header(lay, 183)
        cv.left(lay.desc_x0, 199, "BALANCE BROUGHT FORWARD")
        _bal_cell(cv, lay, 199, m.opening)
        top = 212.4
        for i in range(k):
            t = m.txns[i]
            _txn_row(cv, lay, date(m.year, m.month, t.day), t.description, printed[i],
                     m.balances[i], round(top, 1))
            top += 12.4
        if n_pages == 1:
            cv.left(lay.desc_x0, round(top, 1), "CLOSING BALANCE")
            _bal_cell(cv, lay, round(top, 1), m.closing)
            _footer(cv, lay)
        else:
            _footer(cv, lay)
            cv.next_page()
            _page_head(cv, lay, f"Page 2 of {n_pages}")
            cv.table_header(lay, 73)
            top = 89.0
            for i in range(k, n):
                t = m.txns[i]
                _txn_row(cv, lay, date(m.year, m.month, t.day), t.description, printed[i],
                         m.balances[i], round(top, 1))
                top += 12.4
            cv.left(lay.desc_x0, round(top, 1), "CLOSING BALANCE")
            _bal_cell(cv, lay, round(top, 1), m.closing)
            _footer(cv, lay)
    cv.c.save()


def render_and_ingest(lay: Layout, months: list, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "statement.pdf"
        render(path, lay, months, holder="MERIDIAN QUAY TRADING PTE. LTD.",
               account="FX-1000-2000-3", uen="201234567K", **kw)
        return ingest(str(path))


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------
# Words are letters only and avoid month names, noise keywords (balance, total,
# page, ...) so a description never looks like a date, money or a header line.
WORDS = ("GIRO", "FAST", "PAYNOW", "SALARY", "PAYROLL", "NORTHBROOK", "TESSELATE",
         "QUILLFEATHER", "LUMENWAVE", "BRINKWATER", "ORCHARDLEIGH", "RETAIL", "SUPPLIES",
         "CAFE", "HOLDINGS", "TRADING", "MOBILE", "RENT", "CPF")
OPENING_CENTS = 100_000_000  # 1,000,000.00 keeps every balance positive

descriptions = st.lists(st.sampled_from(WORDS), min_size=2, max_size=4).map(" ".join)
amounts = st.integers(min_value=100, max_value=500_000)  # 1.00 .. 5,000.00 (2 decimals)
txn_st = st.builds(Txn, st.integers(1, 28), descriptions,
                   st.builds(lambda c, cr: c if cr else -c, amounts, st.booleans()))


def _sorted(txns: list) -> list:
    return sorted(txns, key=lambda t: t.day)  # stable: same-day rows keep order


def _month_st(max_rows: int, min_rows: int = 1):
    return st.lists(txn_st, min_size=min_rows, max_size=max_rows).map(_sorted)


def _consecutive(year: int, month: int, n: int) -> list:
    return [((year * 12 + month - 1 + i) // 12, (year * 12 + month - 1 + i) % 12 + 1)
            for i in range(n)]


# --------------------------------------------------------------------------
# Property 6
# --------------------------------------------------------------------------
# Feature: multi-layout-ingestion-and-fabrication-rules, Property 6: Layout independence
@SETTINGS
@given(txns=_month_st(30), year=st.integers(2024, 2026), month=st.integers(1, 12),
       first=st.one_of(st.none(), st.integers(1, 30)),
       layouts=st.permutations(list(LAYOUTS)).map(lambda p: p[:3]))
def test_layout_independence(txns, year, month, first, layouts):
    """**Validates: Requirements 7.3, 8.1, 8.2, 8.3, 10.1**"""
    seen = []
    for name in layouts:
        months = [Month(year, month, OPENING_CENTS, list(txns))]
        _replay(months)
        m = months[0]
        ing = render_and_ingest(LAYOUTS[name], months, rows_first_page=first)
        assert ing.layout_profile == name
        assert not [f for f in ing.data_quality_findings if f.code == "DQ_LAYOUT_UNRECOGNISED"]
        rows = [(t.date, t.description, round(t.amount, 2), round(t.balance, 2))
                for t in ing.transactions]
        expected = [(date(year, month, t.day), t.description, t.cents / 100, b / 100)
                    for t, b in zip(m.txns, m.balances)]
        assert rows == expected
        assert round(ing.opening_balance, 2) == m.opening / 100
        # replayed balance equals the printed closing balance
        assert round(ing.closing_balance, 2) == m.closing / 100
        assert round(m.opening / 100 + sum(r[2] for r in rows), 2) == round(ing.closing_balance, 2)
        seen.append(rows)
    assert all(r == seen[0] for r in seen)  # identical across layouts


# --------------------------------------------------------------------------
# Property 8
# --------------------------------------------------------------------------
# Feature: multi-layout-ingestion-and-fabrication-rules, Property 8: Tampered amount stays visible
@SETTINGS
@given(txns=_month_st(30), year=st.integers(2024, 2026), month=st.integers(1, 12),
       layout=st.sampled_from(TWO_LAYOUTS), pick=st.data(),
       delta=st.integers(100, 100_000), first=st.one_of(st.none(), st.integers(1, 30)))
def test_tampered_amount_stays_visible(txns, year, month, layout, pick, delta, first):
    """**Validates: Requirements 8.7, 8.8, 12.2**"""
    months = [Month(year, month, OPENING_CENTS, list(txns))]
    _replay(months)
    idx = pick.draw(st.integers(0, len(txns) - 1))
    orig = txns[idx].cents
    # same direction (column / sign), different magnitude (>= 1.00 apart)
    edited = orig + delta if orig > 0 else orig - delta
    ing = render_and_ingest(LAYOUTS[layout], months, rows_first_page=first,
                            tamper=(0, idx, edited))
    assert len(ing.transactions) == len(txns)
    # the edited amount is returned exactly as printed; the balance is unchanged
    row = ing.transactions[idx]
    assert round(row.amount, 2) == edited / 100
    assert round(row.balance, 2) == months[0].balances[idx] / 100
    breaks = [f for f in financial_logic.analyze(ing).findings if f.code == "FL_BALANCE_BREAK"]
    assert [f.field for f in breaks] == [f"row {row.index}"]


# --------------------------------------------------------------------------
# Property 9
# --------------------------------------------------------------------------
# Feature: multi-layout-ingestion-and-fabrication-rules, Property 9: Generated multi-page month files do not raise coverage false alarms
@SETTINGS
@given(data=st.data(), n_months=st.integers(2, 6), year=st.integers(2024, 2026),
       month=st.integers(1, 12), layout=st.sampled_from(TWO_LAYOUTS))
def test_multi_month_file_has_no_coverage_false_alarms(data, n_months, year, month, layout):
    """**Validates: Requirements 8.8, 10.2, 12.2**"""
    ym = _consecutive(year, month, n_months)
    months = [Month(y, m, OPENING_CENTS, data.draw(_month_st(6))) for y, m in ym]
    _replay(months)
    ing = render_and_ingest(LAYOUTS[layout], months, page1of1=True)
    periods = [(s.period_start, s.period_end) for s in ing.segments]
    assert periods == [(m.start, m.end) for m in months]
    assert len(set(periods)) == n_months
    codes = {f.code for f in financial_logic.analyze(ing).findings}
    codes |= {f.code for f in ing.data_quality_findings}
    assert not codes & {"FL_MONTH_DUPLICATE", "FL_MONTH_MISSING", "FL_MONTH_OUT_OF_ORDER",
                        "DQ_LAYOUT_UNRECOGNISED"}
    assert "FL_BALANCE_BREAK" not in codes


# Regression found by Property 6: a row whose description has no more words than
# its date has tokens (e.g. "1 Jan GIRO GIRO") was dropped from the candidate
# count, so layout coverage came out 0 and nothing was recognised.
def test_short_description_rows_are_recognised():
    months = [Month(2024, 1, OPENING_CENTS, [Txn(1, "GIRO GIRO", -100), Txn(3, "FAST", 2500)])]
    _replay(months)
    for name in ("two_col_dmon", "signed_amount_iso"):
        ing = render_and_ingest(LAYOUTS[name], months)
        assert ing.layout_profile == name
        assert [t.description for t in ing.transactions] == ["GIRO GIRO", "FAST"]
