"""Fictional-value builders for the Node 3 fabrication-rule tests.

Public names (stable, imported by other test modules):

    business_file(...)                      -> (txns, ing)   six-month business file
    build_business_transactions(...)        -> list[Transaction]
    make_ingestion(txns, period=...)        -> IngestionResult
    insert_copy(txns, row, new_date, new_amount) -> list[Transaction]
    move_references_out_of_order(txns, rows)     -> list[Transaction]
    scale_month_credits(txns, month, factor)     -> list[Transaction]
    recompute_balances(txns, opening)            -> list[Transaction]
    credit_rows(txns) / extract_ref(desc) / customer_of(desc)
    PERIOD_START, PERIOD_END, OPENING_BALANCE, MONTHS

Everything is fictional and deterministic (no randomness). ``row`` arguments are
``Transaction.index`` values. Helpers never mutate their input.
"""
from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Optional, Sequence, Union

from fraud_detect.models import DocumentMetadata, IngestionResult, Transaction

PERIOD_START = date(2026, 4, 1)
PERIOD_END = date(2026, 9, 30)
OPENING_BALANCE = 150_000.00
MONTHS: tuple[tuple[int, int], ...] = tuple((2026, m) for m in range(4, 10))
ROWS_PER_PAGE = 25

PAYROLL_AMOUNT = 18_000.00
CPF_AMOUNT = 4_000.00
RENT_AMOUNT = 6_000.00
GST_AMOUNT = 9_500.00
GST_MONTH = (2026, 7)

# Target monthly credit turnover (a gentle upward trend). Over the full six months the
# two largest months (Aug, Sep) are ~1.10-1.11x the median of the other months, so they
# only report under a ratio override of 1.1; no month exceeds ~1.12x the median of the
# others in any window of 4-6 consecutive months, so the default 1.2 never reports.
MONTHLY_CREDIT_TARGETS = {4: 48_000.00, 5: 51_500.00, 6: 53_000.00,
                          7: 57_000.00, 8: 58_500.00, 9: 59_000.00}

# Fictional customers; none contains RENT/CPF/PAYROLL/LOAN/TRANSFER/INTEREST/CURRENT.
CUSTOMERS = (
    "CINDERPATH HOLDINGS", "MARLOWE TRADING", "BRIGHTWELL SUPPLIES",
    "ASHGROVE LOGISTICS", "QUILLON DESIGN", "HARROWGATE FOODS",
    "TALLOWMERE ENGINEERING", "OAKENSHAW RETAIL", "PELLUCID MEDIA",
    "WYNDHAM CRAFTS", "SORRELBY MARINE", "FENWICK AND SONS",
)
OTHER_DEBITS = (
    "SUPPLIER PAYMENT KESTREL PACKAGING", "UTILITIES SP SERVICES",
    "OFFICE SUPPLIES STATIONERS", "FAST TO LANTERN LOGISTICS",
    "INSURANCE PREMIUM ALDERMERE", "BANK CHARGES", "TELECOM SINGTEL",
    "SUPPLIER PAYMENT NORTHGATE PARTS", "FAST TO HOLLOWAY FREIGHT",
    "MARKETING SPEND BRAMBLE ADS",
)

_INV_RE = re.compile(r"\bINV\s*0*(\d+)\b", re.IGNORECASE)
MonthLike = Union[tuple, str, date]


def extract_ref(description: Optional[str]) -> Optional[str]:
    m = _INV_RE.search(description or "")
    return f"INV{int(m.group(1))}" if m else None


def customer_of(description: str) -> str:
    d = _INV_RE.sub("", description).replace("FAST FROM", "")
    return " ".join(d.split())


def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def recompute_balances(txns: Sequence[Transaction], opening: float) -> list[Transaction]:
    """Return copies re-indexed 1..n (in the given order) with running balances."""
    out: list[Transaction] = []
    bal = round(opening, 2)
    for i, t in enumerate(txns, start=1):
        bal = round(bal + (t.amount or 0.0), 2)
        out.append(t.model_copy(update={
            "index": i, "balance": bal, "page": (i - 1) // ROWS_PER_PAGE + 1,
        }))
    return out


def opening_of(txns: Sequence[Transaction]) -> float:
    first = txns[0]
    return round((first.balance or 0.0) - (first.amount or 0.0), 2)


def build_business_transactions(opening: float = OPENING_BALANCE) -> list[Transaction]:
    """Six months (2026-04..2026-09) of a fictional trading business."""
    # (date, order, description, amount); order keeps credits before debits on a day.
    spec: list[tuple[date, int, str, float]] = []
    inv = 3262
    k = 0
    for (y, m) in MONTHS:
        last = _last_day(y, m)
        spec.append((date(y, m, 1), 1, f"RENT {calendar.month_abbr[m].upper()} {y} UNIT 12-04", -RENT_AMOUNT))
        spec.append((date(y, m, 25), 1, f"PAYROLL {calendar.month_abbr[m].upper()} {y}", -PAYROLL_AMOUNT))
        spec.append((date(y, m, 26), 1, "CPF CONTRIBUTION", -CPF_AMOUNT))
        if (y, m) == GST_MONTH:
            spec.append((date(y, m, 15), 1, "GST PAYMENT IRAS", -GST_AMOUNT))
        n_credits = 10 + (m * 3) % 5          # 10..14, fixed per month
        for j in range(n_credits):
            k += 1
            inv += 1 + (k % 3)                 # strictly rising
            day = 2 + (j * 27) // n_credits    # spread over days 2..28
            amt = 2_000 + (k * 7919) % 6_000 + ((k * 37) % 100) / 100
            cust = CUSTOMERS[k % len(CUSTOMERS)]
            spec.append((date(y, m, day), 0, f"FAST FROM {cust} INV{inv}", round(amt, 2)))
        for j in range(8 + (m % 3)):
            k += 1
            day = 3 + (j * 25) // 8
            amt = 400 + (k * 613) % 2_600 + ((k * 53) % 100) / 100
            spec.append((date(y, m, day), 2, OTHER_DEBITS[k % len(OTHER_DEBITS)], -round(amt, 2)))
    # Rescale each month's credits to its target turnover (last credit absorbs rounding).
    for (y, m) in MONTHS:
        idx = [i for i, r in enumerate(spec) if r[0].month == m and r[3] > 0]
        factor = MONTHLY_CREDIT_TARGETS[m] / sum(spec[i][3] for i in idx)
        scaled = [round(spec[i][3] * factor, 2) for i in idx]
        scaled[-1] = round(MONTHLY_CREDIT_TARGETS[m] - sum(scaled[:-1]), 2)
        for i, a in zip(idx, scaled):
            spec[i] = (spec[i][0], spec[i][1], spec[i][2], a)
    spec.sort(key=lambda s: (s[0], s[1]))      # stable: ties keep insertion order
    raw = [Transaction(index=0, date=d, date_raw=d.strftime("%d %b %Y"),
                       description=desc, amount=a) for d, _, desc, a in spec]
    return recompute_balances(raw, opening)


def make_ingestion(
    txns: Sequence[Transaction],
    period: Optional[tuple[Optional[date], Optional[date]]] = (PERIOD_START, PERIOD_END),
) -> IngestionResult:
    """Minimal IngestionResult around ``txns``; ``period=None`` leaves the period unknown."""
    start, end = period if period else (None, None)
    txns = list(txns)
    pages = max((t.page or 1 for t in txns), default=1)
    return IngestionResult(
        source_path="fabrication_fixture.pdf",
        page_count=pages,
        raw_text="",
        metadata=DocumentMetadata(),
        account_holder="TESTCO PTE LTD",
        statement_period_start=start,
        statement_period_end=end,
        opening_balance=opening_of(txns) if txns else None,
        closing_balance=txns[-1].balance if txns else None,
        declared_currency="SGD",
        transactions=txns,
    )


def business_file(period=(PERIOD_START, PERIOD_END)):
    """The main builder: returns ``(txns, ing)``."""
    txns = build_business_transactions()
    return txns, make_ingestion(txns, period=period)


def credit_rows(txns: Sequence[Transaction]) -> list[Transaction]:
    return [t for t in txns if (t.amount or 0) > 0]


def insert_copy(txns: Sequence[Transaction], row: int, new_date: date,
                new_amount: float) -> list[Transaction]:
    """Add a credit sharing customer and reference with ``row`` but a new date/amount.

    The copy goes after every row dated on or before ``new_date``; rows are
    re-indexed and balances recomputed from the original opening balance.
    """
    txns = list(txns)
    opening = opening_of(txns)
    src = next(t for t in txns if t.index == row)
    copy = src.model_copy(update={
        "date": new_date, "date_raw": new_date.strftime("%d %b %Y"),
        "amount": round(new_amount, 2),
    })
    pos = sum(1 for t in txns if t.date is not None and t.date <= new_date)
    txns.insert(pos, copy)
    return recompute_balances(txns, opening)


def move_references_out_of_order(txns: Sequence[Transaction], rows: Sequence[int]) -> list[Transaction]:
    """Rotate the invoice numbers among ``rows`` (customers stay put)."""
    rows = list(rows)
    by_index = {t.index: t for t in txns}
    refs = [_INV_RE.search(by_index[r].description or "").group(0) for r in rows]
    shifted = refs[1:] + refs[:1]
    new_desc = {
        r: _INV_RE.sub(new, by_index[r].description, count=1)
        for r, new in zip(rows, shifted)
    }
    return [t.model_copy(update={"description": new_desc[t.index]}) if t.index in new_desc else t.model_copy()
            for t in txns]


def _month_key(month: MonthLike) -> tuple[int, int]:
    if isinstance(month, date):
        return (month.year, month.month)
    if isinstance(month, str):
        y, m = month.split("-")
        return (int(y), int(m))
    return (int(month[0]), int(month[1]))


def scale_month_credits(txns: Sequence[Transaction], month: MonthLike, factor: float) -> list[Transaction]:
    """Multiply every credit dated in ``month`` by ``factor``; balances are recomputed."""
    key = _month_key(month)
    out = [
        t.model_copy(update={"amount": round(t.amount * factor, 2)})
        if (t.amount or 0) > 0 and t.date and (t.date.year, t.date.month) == key
        else t.model_copy()
        for t in txns
    ]
    return recompute_balances(out, opening_of(txns))
