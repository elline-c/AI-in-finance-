"""Table reader (Requirement 5): find the transaction table and read each row.

Amounts are right-aligned, so an amount belongs to the column whose header
RIGHT edge is nearest to the amount's right edge. Two-word headers such as
``Money out`` are joined before use. Summary rows, footers and wrapped
description lines never become rows. When no header is found the reader falls
back to line-based reading (date first, amounts last).

Safety: the work per line is linear in the number of words, money and date
checks reject over-long tokens up front (see ``money`` and ``dates``), and the
regexes used here are anchored and have no nested quantifiers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal, Optional

from .dates import parse_day_month, parse_full_date
from .money import Money, parse_money
from .types import Word

_LINE_TOL = 2.5          # words whose tops differ by less than this share a line
_MAX_TOKEN = 64          # no amount/prefix token is longer than this
_MAX_CELLS = 3           # at most: two amounts + balance

_DESC_HEADERS = {"description", "details", "particulars", "narrative", "transaction"}
_OUT_NAMES = {"debit", "debits", "withdrawal", "withdrawals"}
_IN_NAMES = {"credit", "credits", "deposit", "deposits"}
_JOIN_FIRST = {"money": {"out": "money_out", "in": "money_in"},
               "paid": {"out": "money_out", "in": "money_in"}}

_DEC_RE = re.compile(r"\.\d{2}")
_PREFIX_RE = re.compile(r"[+-]?(?:SGD|S\$|\$)?", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d{1,2}")
_PAGE_RE = re.compile(r"page\s{0,3}\d{1,4}\s{0,3}(?:of\s{0,3}\d{1,4})?", re.IGNORECASE)
_NOISE_STARTS = (
    "balance brought forward", "brought forward", "balance b/f", "closing balance",
    "balance c/f", "carried forward", "total",
)


@dataclass
class AmountCell:
    money: Money
    x0: float
    x1: float
    column: Optional[str] = None  # "money_out" | "money_in" | "amount" | None


@dataclass
class RawRow:
    date_text: str
    value_date_text: Optional[str]
    description: str
    amount_cells: list[AmountCell]
    balance: Money
    line_text: str
    top: float = 0.0
    page: int = 0  # filled in by the caller; the table reader sees one page's words


@dataclass
class TableLayout:
    header_y: float
    columns: dict[str, tuple[float, float]]
    kind: Literal["two_column", "single_signed", "unknown"]
    has_value_date: bool = False
    desc_x0: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- lines
def _group_lines(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    start_top = 0.0
    for w in sorted(words, key=lambda w: (w.top, w.x0)):
        if lines and w.top - start_top < _LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
            start_top = w.top
    for ln in lines:
        ln.sort(key=lambda w: w.x0)
    return lines


def _line_text(line: list[Word]) -> str:
    return " ".join(w.text for w in line)


# ---------------------------------------------------------------- header
def _header_columns(line: list[Word]) -> Optional[TableLayout]:
    lower = [w.text.lower() for w in line]
    if "date" not in lower or "balance" not in lower:
        return None
    desc = [w for w, t in zip(line, lower) if t in _DESC_HEADERS]
    if not desc:
        return None
    columns: dict[str, tuple[float, float]] = {}
    i = 0
    while i < len(line):
        w, t = line[i], lower[i]
        nxt = lower[i + 1] if i + 1 < len(line) else None
        key: Optional[str] = None
        x0, x1, step = w.x0, w.x1, 1
        if t in _JOIN_FIRST and nxt in _JOIN_FIRST[t]:
            key, x1, step = _JOIN_FIRST[t][nxt], line[i + 1].x1, 2
        elif t in _OUT_NAMES:
            key = "money_out"
        elif t in _IN_NAMES:
            key = "money_in"
        elif t == "amount":
            key = "amount"
        elif t == "balance":
            key = "balance"
        if key is not None and key not in columns:
            columns[key] = (x0, x1)
        i += step
    if "balance" not in columns:
        return None
    if "money_out" in columns and "money_in" in columns:
        kind = "two_column"
    elif "amount" in columns:
        kind = "single_signed"
    else:
        kind = "unknown"
    return TableLayout(
        header_y=line[0].top,
        columns=columns,
        kind=kind,
        has_value_date=lower.count("date") >= 2,
        desc_x0=min(w.x0 for w in desc),
    )


def find_table(page_words: list[Word], profile: Any = None) -> Optional[TableLayout]:
    """Return the first table header found on the page, or ``None``."""
    for line in _group_lines(page_words):
        layout = _header_columns(line)
        if layout is not None:
            return layout
    return None


# ---------------------------------------------------------------- rows
_DUMMY_START, _DUMMY_END = date(2000, 1, 1), date(2001, 12, 31)


def _read_date(tokens: list[str]) -> int:
    """Number of leading tokens (0, 1, 2 or 3) that form a date."""
    if not tokens:
        return 0
    t0 = tokens[0]
    if len(t0) <= 12 and parse_full_date(t0).value is not None:
        return 1
    if len(tokens) >= 2 and _DIGITS_RE.fullmatch(t0) and len(tokens[1]) <= 9:
        if len(tokens) >= 3 and len(tokens[2]) == 4 and tokens[2].isdigit():
            if parse_full_date(" ".join(tokens[:3])).value is not None:
                return 3
        if parse_day_month(f"{t0} {tokens[1]}", _DUMMY_START, _DUMMY_END).value is not None:
            return 2
    return 0


def _is_number_word(text: str) -> bool:
    return len(text) <= _MAX_TOKEN and _DEC_RE.search(text) is not None and parse_money(text) is not None


def _read_cells(words: list[Word]) -> tuple[int, list[tuple[Money, float, float]]]:
    """Scan from the right for money cells; return (index of first cell word, cells left-to-right)."""
    cells: list[tuple[Money, float, float]] = []
    i = len(words)
    while i > 0 and len(cells) < _MAX_CELLS:
        j = i - 1
        suffix = None
        if words[j].text.upper() in ("CR", "DR") and j > 0:
            suffix = words[j].text
            j -= 1
        if not _is_number_word(words[j].text):
            break
        num = words[j]
        first = j
        if j > 0 and len(words[j - 1].text) <= 8 and _PREFIX_RE.fullmatch(words[j - 1].text) \
                and words[j - 1].text:
            first = j - 1
        text = " ".join(w.text for w in words[first : j + 1])
        if suffix:
            text += " " + suffix
        money = parse_money(text)
        if money is None:
            break
        cells.append((money, words[first].x0, num.x1))
        i = first
    cells.reverse()
    return i, cells


def _is_noise(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return True
    if "specimen" in low or "fictional" in low:
        return True
    if _PAGE_RE.fullmatch(low):
        return True
    return low.startswith(_NOISE_STARTS)


def _assign_column(layout: TableLayout, x1: float) -> Optional[str]:
    best, best_d = None, 0.0
    for key in ("money_out", "money_in", "amount"):
        col = layout.columns.get(key)
        if col is None:
            continue
        d = abs(col[1] - x1)
        if best is None or d < best_d:
            best, best_d = key, d
    return best


def read_rows(page_words: list[Word], layout: Optional[TableLayout], profile: Any = None) -> list[RawRow]:
    """Read every transaction row on one page, each printed row exactly once."""
    rows: list[RawRow] = []
    prev: Optional[RawRow] = None
    for line in _group_lines(page_words):
        if layout is not None and line[0].top <= layout.header_y + _LINE_TOL / 2:
            continue  # above or on the table header
        if _header_columns(line) is not None:
            prev = None  # repeated header
            continue
        text = _line_text(line)
        if _is_noise(text):
            prev = None
            continue
        tokens = [w.text for w in line]
        n_date = _read_date(tokens)
        body = line[n_date:]
        date_text = " ".join(tokens[:n_date])
        value_text: Optional[str] = None
        if n_date and layout is not None and layout.has_value_date:
            n2 = _read_date([w.text for w in body])
            if n2:
                value_text = " ".join(w.text for w in body[:n2])
                body = body[n2:]
        first_cell, cells = _read_cells(body)
        if layout is not None:
            # amounts must sit right of the description column start
            cells = [c for c in cells if c[1] >= layout.desc_x0 - 1.0]
        desc_words = body[:first_cell] if cells else body
        desc = " ".join(w.text for w in desc_words)
        if n_date and len(cells) >= 2:
            if _is_noise(desc):  # e.g. "01 Jul BALANCE BROUGHT FORWARD ..."
                prev = None
                continue
            *amounts, bal = cells
            row = RawRow(
                date_text=date_text,
                value_date_text=value_text,
                description=desc,
                amount_cells=[
                    # value is the printed magnitude; the sign stays in explicit_sign/suffix
                    AmountCell(replace(m, value=abs(m.value)), x0, x1, _assign_column(layout, x1) if layout is not None else None)
                    for m, x0, x1 in amounts
                ],
                balance=bal[0],
                line_text=text,
                top=line[0].top,
            )
            rows.append(row)
            prev = row
        elif n_date == 0 and not cells and prev is not None:
            prev.description = (prev.description + " " + text).strip()
        else:
            prev = None
    return rows
