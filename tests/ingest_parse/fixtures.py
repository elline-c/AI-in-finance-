"""In-code page fixtures for the six supported statement layouts.

Everything here is fictional (names, accounts, amounts). The word-box geometry
(x0, x1, top) follows what pdfplumber reported for the synthetic forgery
dataset layouts 1-5 and ``statement_E`` (read-only inspection): amounts are
right-aligned, so the right edge ``x1`` of an amount matches the right edge of
its column header. Layout 5 headers ``Money out`` / ``Money in`` are two words;
layout 4 prints ``CR`` as a separate word after the balance; layout 3 prints
``-SGD`` / ``+SGD`` as a separate word before the number.

Each builder returns a :class:`Fixture` carrying the pages plus the values a
correct parse must produce.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

from fraud_detect.nodes.ingest_parse.types import Line, PageContent, Word

CHAR_W = 4.7
GAP = 2.3
ROW_STEP = 12.4

PROFILE_NAMES = (
    "generator_v1",
    "two_col_dmon",
    "two_col_slash",
    "two_col_dual_date",
    "signed_amount_iso",
    "two_col_ddmmyy_cr",
)

# (day, description, signed amount). Credits positive, debits negative.
SAMPLE_TXNS: tuple[tuple[int, str, float], ...] = (
    (2, "GIRO RENT NORTHBROOK HOLDINGS", -1200.00),
    (4, "FAST FROM TESSELATE RETAIL INV1001", 3500.00),
    (4, "NETS/CARD QUILLFEATHER CAFE 1234", -18.40),
    (9, "PAYNOW FROM ORCHARDLEIGH TRADING DEP", 950.25),
    (15, "GIRO LUMENWAVE MOBILE REF200001", -64.90),
    (22, "SALARY PAYROLL CPF", -2100.00),
    (28, "FAST TO BRINKWATER SUPPLIES", -420.10),
)
OPENING = 10000.00


@dataclass(frozen=True)
class ExpectedTxn:
    date: date
    description: str
    amount: float  # signed: credits +, debits -
    balance: float


@dataclass(frozen=True)
class ExpectedSection:
    period_start: date
    period_end: date
    opening: float
    closing: float
    total_withdrawals: float
    total_deposits: float
    txns: tuple[ExpectedTxn, ...]


@dataclass
class Fixture:
    profile: str
    holder: str
    account: str
    uen: Optional[str]
    pages: list[PageContent]
    sections: list[ExpectedSection] = field(default_factory=list)

    @property
    def n_txns(self) -> int:
        return sum(len(s.txns) for s in self.sections)


# --------------------------------------------------------------------------
# low-level word / line helpers
# --------------------------------------------------------------------------
def _width(tok: str) -> float:
    return CHAR_W * len(tok)


def _left(tokens: list[str], x0: float, top: float) -> list[Word]:
    out, x = [], x0
    for t in tokens:
        out.append(Word(t, round(x, 1), round(x + _width(t), 1), top))
        x += _width(t) + GAP
    return out


def _right(tokens: list[str], x1: float, top: float) -> list[Word]:
    total = sum(_width(t) for t in tokens) + GAP * (len(tokens) - 1)
    return _left(tokens, x1 - total, top)


def _split(text: str) -> list[str]:
    return text.split()


class _Page:
    """Collects word groups per line and renders a PageContent."""

    def __init__(self, number: int):
        self.number = number
        self._lines: list[tuple[float, list[Word]]] = []

    def add(self, top: float, *groups: list[Word]) -> None:
        words = [w for g in groups for w in g]
        words.sort(key=lambda w: w.x0)
        self._lines.append((top, words))

    def build(self) -> PageContent:
        lines = [Line(" ".join(w.text for w in ws), top) for top, ws in self._lines]
        words = [w for _, ws in self._lines for w in ws]
        return PageContent(page=self.number, lines=lines, words=words)


def _fmt_num(v: float) -> str:
    return f"{abs(v):,.2f}"


def _last_day(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _replay(year: int, month: int, opening: float) -> ExpectedSection:
    bal, txns = opening, []
    for day, desc, amt in SAMPLE_TXNS:
        bal = round(bal + amt, 2)
        txns.append(ExpectedTxn(date(year, month, day), desc, amt, bal))
    debits = round(-sum(t.amount for t in txns if t.amount < 0), 2)
    credits = round(sum(t.amount for t in txns if t.amount > 0), 2)
    return ExpectedSection(
        date(year, month, 1), _last_day(year, month), opening, bal, debits, credits, tuple(txns)
    )


# --------------------------------------------------------------------------
# dataset-style layouts (1-5 and statement_E share one skeleton)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Spec:
    profile: str
    bank: str
    holder: str
    account: str
    uen: Optional[str]
    fmt_date: Callable[[date], str]  # period / statement date text
    row_date: Callable[[date], list[str]]  # tokens printed in the row date cell
    date_x0: float
    desc_x0: float
    header: tuple[tuple[str, float, float], ...]  # (text, x0, x1) for the table header
    out_x1: Optional[float]
    in_x1: Optional[float]
    amount_x1: Optional[float]  # single signed column (layout 3)
    bal_x1: float
    money: Callable[[float], list[str]]  # tokens for an unsigned money value
    value_date_x0: Optional[float] = None
    bal_suffix: bool = False  # layout 4 "CR"


_SUMMARY_X0 = (40.0, 170.8, 301.6, 432.5)
_SUMMARY_X1 = (160.8, 291.6, 422.5, 553.3)
_SUMMARY_LABELS = ("Opening Balance", "Total Withdrawals", "Total Deposits", "Closing Balance")
_FOOTER_1 = "SPECIMEN · Fictional institution · Synthetic data for ML research · Not a real bank document"


def _header_block(spec: _Spec, pg: _Page, sec: ExpectedSection, page_text: str) -> None:
    pg.add(30, _left(_split("STATEMENT OF ACCOUNT"), 40, 30))
    pg.add(44, _left(_split(spec.bank), 40, 44))
    pg.add(56, _left(_split(page_text), 40, 56))
    pg.add(66, _left(["SPECIMEN"], 40, 66))
    left = [spec.holder, "BLK 12 FICTIONAL GROVE 3", "#10-24", "SINGAPORE 990001"]
    right = [
        f"Account No. {spec.account}",
        f"Statement Period {spec.fmt_date(sec.period_start)} to {spec.fmt_date(sec.period_end)}",
        f"Statement Date {spec.fmt_date(sec.period_end + timedelta(days=1))}",
        "Currency SGD",
    ]
    if spec.uen:
        left = [spec.holder, "3 FICTIONAL DRIVE", "#10-24", "SINGAPORE 990001", "BUSINESS CURRENT ACCOUNT"]
        right = [
            f"Account No. {spec.account}",
            f"Company UEN {spec.uen}",
            right[1],
            right[2],
            "Currency SGD",
        ]
    for i, (l, r) in enumerate(zip(left, right)):
        top = 78 + 12 * i
        pg.add(top, _left(_split(l), 40, top), _left(_split(r), 300, top))
    # summary box: four labels, then four right-aligned values
    vals = [sec.opening, sec.total_withdrawals, sec.total_deposits, sec.closing]
    label_groups = [_left(_split(lab), x0, 142) for x0, lab in zip(_SUMMARY_X0, _SUMMARY_LABELS)]
    pg.add(142, *label_groups)
    value_groups = [_right(spec.money(v), x1, 154) for v, x1 in zip(vals, _SUMMARY_X1)]
    pg.add(154, *value_groups)


def _table_header(spec: _Spec, pg: _Page, top: float) -> None:
    words = []
    for text, x0, x1 in spec.header:
        words.append(Word(text, x0, x1, top))
    pg.add(top, words)


def _balance_cell(spec: _Spec, v: float, top: float) -> list[Word]:
    ws = _right(spec.money(v), spec.bal_x1, top)
    if spec.bal_suffix:
        ws.append(Word("CR", 550.6, 560.0, top))
    return ws


def _txn_row(spec: _Spec, pg: _Page, t: ExpectedTxn, top: float) -> None:
    groups = [_left(spec.row_date(t.date), spec.date_x0, top)]
    if spec.value_date_x0 is not None:
        vd = t.date + timedelta(days=1 if t.date.day % 2 else 0)
        groups.append(_left(spec.row_date(vd), spec.value_date_x0, top))
    groups.append(_left(_split(t.description), spec.desc_x0, top))
    if spec.amount_x1 is not None:
        sign = "+" if t.amount > 0 else "-"
        toks = spec.money(abs(t.amount))
        toks = [sign + toks[0]] + toks[1:] if toks[0].isalpha() else [sign + toks[0]]
        groups.append(_right(toks, spec.amount_x1, top))
    else:
        x1 = spec.in_x1 if t.amount > 0 else spec.out_x1
        groups.append(_right(spec.money(abs(t.amount)), x1, top))
    groups.append(_balance_cell(spec, t.balance, top))
    pg.add(top, *groups)


def _dataset_pages(
    spec: _Spec, sections: list[ExpectedSection], rows_first_page: Optional[int], page1of1: bool = False
) -> list[PageContent]:
    pages: list[PageContent] = []
    for sec in sections:
        first_rows = list(sec.txns if rows_first_page is None else sec.txns[:rows_first_page])
        rest = list(sec.txns[len(first_rows):])
        n_pages = 2 if rest else 1
        pg = _Page(len(pages) + 1)
        _header_block(spec, pg, sec, "Page 1 of 1" if page1of1 else f"Page 1 of {n_pages}")
        _table_header(spec, pg, 183)
        bf_top = 199.0
        pg.add(
            bf_top,
            _left(["BALANCE", "BROUGHT", "FORWARD"], spec.desc_x0, bf_top),
            _balance_cell(spec, sec.opening, bf_top),
        )
        top = 212.4
        for t in first_rows:
            _txn_row(spec, pg, t, round(top, 1))
            top += ROW_STEP
        if not rest:
            _closing_and_footer(spec, pg, sec, round(top, 1))
        else:
            _footer(spec, pg)
        pages.append(pg.build())
        if rest:
            pg2 = _Page(len(pages) + 1)
            pg2.add(30, _left(_split("STATEMENT OF ACCOUNT"), 40, 30))
            pg2.add(44, _left(_split(spec.bank), 40, 44))
            pg2.add(56, _left(_split(f"Page 2 of {n_pages}"), 40, 56))
            pg2.add(66, _left(["SPECIMEN"], 40, 66))
            _table_header(spec, pg2, 73)
            top = 89.0
            for t in rest:
                _txn_row(spec, pg2, t, round(top, 1))
                top += ROW_STEP
            _closing_and_footer(spec, pg2, sec, round(top, 1))
            pages.append(pg2.build())
    return pages


def _closing_and_footer(spec: _Spec, pg: _Page, sec: ExpectedSection, top: float) -> None:
    pg.add(
        top,
        _left(["CLOSING", "BALANCE"], spec.desc_x0, top),
        _balance_cell(spec, sec.closing, top),
    )
    _footer(spec, pg)


def _footer(spec: _Spec, pg: _Page) -> None:
    pg.add(770, _left(_split(_FOOTER_1), 40, 770))
    pg.add(
        782,
        _left(_split(f"{spec.bank} is a fictional entity created for synthetic dataset generation. Ref FIX-001"), 40, 782),
    )


def _iso(d: date) -> str:
    return d.isoformat()


def _spec_for(profile: str, holder: str, account: str, uen: Optional[str] = None) -> _Spec:
    if profile == "two_col_slash":  # dataset layout 1
        return _Spec(
            profile, "Corvane Trust", holder, account, uen,
            fmt_date=lambda d: d.strftime("%d/%m/%Y"),
            row_date=lambda d: [d.strftime("%d/%m/%Y")],
            date_x0=40.0, desc_x0=100.0,
            header=(("Date", 40.0, 57.8), ("Description", 100.0, 145.1),
                    ("Withdrawals", 351.7, 400.0), ("Deposits", 440.4, 475.0), ("Balance", 523.6, 555.0)),
            out_x1=400.0, in_x1=475.0, amount_x1=None, bal_x1=555.0,
            money=lambda v: [_fmt_num(v)],
        )
    if profile == "two_col_dual_date":  # dataset layout 2
        return _Spec(
            profile, "Halcyon Ledger Bank", holder, account, uen,
            fmt_date=lambda d: f"{d:%d %b %Y}",
            row_date=lambda d: [f"{d:%d}", f"{d:%b}", f"{d:%Y}"],
            date_x0=36.0, desc_x0=172.0, value_date_x0=104.0,
            header=(("Txn", 36.0, 51.2), ("Date", 53.4, 71.0), ("Value", 104.0, 126.0), ("Date", 128.2, 145.8),
                    ("Particulars", 172.0, 214.0), ("Debit", 399.5, 420.0), ("Credit", 465.6, 490.0),
                    ("Balance", 529.2, 559.0)),
            out_x1=420.0, in_x1=490.0, amount_x1=None, bal_x1=559.0,
            money=lambda v: [_fmt_num(v)],
        )
    if profile == "signed_amount_iso":  # dataset layout 3
        return _Spec(
            profile, "Aldermoor Savings", holder, account, uen,
            fmt_date=_iso, row_date=lambda d: [_iso(d)],
            date_x0=40.0, desc_x0=110.0,
            header=(("Date", 40.0, 60.2), ("Details", 110.0, 140.0), ("Amount", 416.5, 450.0),
                    ("Balance", 521.3, 555.0)),
            out_x1=None, in_x1=None, amount_x1=450.0, bal_x1=555.0,
            money=lambda v: ["SGD", _fmt_num(v)],
        )
    if profile == "two_col_ddmmyy_cr":  # dataset layout 4
        return _Spec(
            profile, "Kestrelmark Bank", holder, account, uen,
            fmt_date=lambda d: d.strftime("%d-%m-%y"),
            row_date=lambda d: [d.strftime("%d-%m-%y")],
            date_x0=36.0, desc_x0=92.0,
            header=(("DATE", 36.0, 54.7), ("DESCRIPTION", 92.0, 143.5), ("DEBIT", 371.6, 395.0),
                    ("CREDIT", 441.9, 470.0), ("BALANCE", 527.2, 560.0)),
            out_x1=395.0, in_x1=470.0, amount_x1=None, bal_x1=546.0,
            money=lambda v: [_fmt_num(v)], bal_suffix=True,
        )
    if profile == "two_col_dmon":  # dataset layout 5 and statement_E
        return _Spec(
            profile, "Vellichor Bank", holder, account, uen,
            fmt_date=lambda d: f"{d:%d %b %Y}",
            row_date=lambda d: [str(d.day), d.strftime("%b")],
            date_x0=38.0, desc_x0=90.0,
            header=(("Date", 38.0, 58.4), ("Description", 90.0, 140.8), ("Money", 345.8, 374.8),
                    ("out", 377.5, 392.0), ("Money", 431.6, 460.7), ("in", 463.4, 472.0),
                    ("Balance", 522.0, 557.0)),
            out_x1=392.0, in_x1=472.0, amount_x1=None, bal_x1=557.0,
            money=lambda v: ["S$" + _fmt_num(v)],
        )
    raise ValueError(profile)


# Layout 4's summary row has no CR and layout-4 "CR" is only on balance cells;
# the summary box uses spec.money directly, so no suffix is added there.


def _make(profile: str, rows_first_page: Optional[int] = None, uen: Optional[str] = None,
          holder: str = "TALIA ROWENHALE", account: str = "FX-1000-2000-3",
          months: tuple[tuple[int, int], ...] = ((2026, 7),), page1of1: bool = False) -> Fixture:
    spec = _spec_for(profile, holder, account, uen)
    sections, opening = [], OPENING
    for y, m in months:
        sec = _replay(y, m, opening)
        sections.append(sec)
        opening = sec.closing
    pages = _dataset_pages(spec, sections, rows_first_page, page1of1)
    return Fixture(profile, holder, account, uen, pages, sections)


# --------------------------------------------------------------------------
# public fixture builders (one per profile)
# --------------------------------------------------------------------------
def two_col_slash_fixture(rows_first_page: Optional[int] = None) -> Fixture:
    return _make("two_col_slash", rows_first_page)


def two_col_dual_date_fixture(rows_first_page: Optional[int] = None) -> Fixture:
    return _make("two_col_dual_date", rows_first_page)


def signed_amount_iso_fixture() -> Fixture:
    return _make("signed_amount_iso")


def two_col_ddmmyy_cr_fixture() -> Fixture:
    return _make("two_col_ddmmyy_cr")


def two_col_dmon_fixture(with_uen: bool = True) -> Fixture:
    """Layout 5 / statement_E style (``d Mon`` rows, ``S$`` values)."""
    return _make("two_col_dmon", uen="201234567K" if with_uen else None,
                 holder="MERIDIAN QUAY TRADING PTE. LTD." if with_uen else "TALIA ROWENHALE")


def continuation_fixture() -> Fixture:
    """Layout 1 month that runs onto page 2: repeated table header, no
    ``Statement Period`` line and no brought-forward row on the second page."""
    return _make("two_col_slash", rows_first_page=4)


def six_page_page1of1_fixture() -> Fixture:
    """Six monthly pages (Apr-Sep 2026), each printing 'Page 1 of 1'."""
    months = tuple((2026, m) for m in range(4, 10))
    return _make("two_col_dmon", uen="201234567K", holder="MERIDIAN QUAY TRADING PTE. LTD.",
                 months=months, page1of1=True)


# --------------------------------------------------------------------------
# generator_v1: our own synthetic generator (left-aligned, colon labels)
# --------------------------------------------------------------------------
def generator_v1_fixture(rows_first_page: Optional[int] = None) -> Fixture:
    holder, account, uen = "Aurelia Fenwick Pte Ltd", "123-456-789", "201234567K"
    sec = _replay(2026, 1, OPENING)
    first = list(sec.txns if rows_first_page is None else sec.txns[:rows_first_page])
    rest = list(sec.txns[len(first):])
    mm = 72 / 25.4

    def header(pg: _Page, page_no: int) -> float:
        top = 40.0
        pg.add(top, _left(_split("OCBC-STYLE BANK | Corporate Statement"), 20 * mm, top))
        top += 14
        if page_no == 1:
            lines = [
                f"Account Holder: {holder}", f"UEN: {uen}", f"Account No: {account}",
                f"Statement Period: {sec.period_start:%d/%m/%Y} - {sec.period_end:%d/%m/%Y}",
                "Currency: SGD", f"Opening Balance: {sec.opening:,.2f}", f"Page {page_no}",
            ]
        else:
            lines = [f"Account No: {account}  (continued)", f"Page {page_no}"]
        for text in lines:
            pg.add(top, _left(_split(text), 20 * mm, top))
            top += 14
        top += 8
        pg.add(top, _left(["Date"], 20 * mm, top), _left(["Description"], 55 * mm, top),
               _left(["Amount"], 120 * mm, top), _left(["Balance"], 160 * mm, top))
        return top + 17

    def row(pg: _Page, t: ExpectedTxn, top: float) -> None:
        sign = "-" if t.amount < 0 else ""
        pg.add(
            top,
            _left([t.date.strftime("%d/%m/%Y")], 20 * mm, top),
            _left(_split(t.description), 55 * mm, top),
            _left([f"SGD{sign}{_fmt_num(t.amount)}"], 120 * mm, top),
            _left([f"SGD{t.balance:,.2f}"], 160 * mm, top),
        )

    pages = []
    pg = _Page(1)
    top = header(pg, 1)
    for t in first:
        row(pg, t, top)
        top += 14
    pages.append(pg.build())
    if rest:
        pg2 = _Page(2)
        top = header(pg2, 2)
        for t in rest:
            row(pg2, t, top)
            top += 14
        pages.append(pg2.build())
    return Fixture("generator_v1", holder, account, uen, pages, [sec])


ALL_FIXTURES: dict[str, Callable[[], Fixture]] = {
    "generator_v1": generator_v1_fixture,
    "two_col_dmon": two_col_dmon_fixture,
    "two_col_slash": two_col_slash_fixture,
    "two_col_dual_date": two_col_dual_date_fixture,
    "signed_amount_iso": signed_amount_iso_fixture,
    "two_col_ddmmyy_cr": two_col_ddmmyy_cr_fixture,
}
