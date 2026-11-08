"""Node 1 date reader (Requirements 2.1 to 2.8).

Turns the date styles seen on bank statements into real dates without guessing.
Anything that cannot be read safely comes back with ``value=None`` and the raw
text kept, so callers can raise a data-quality finding.

All regexes are anchored or use bounded quantifiers, and over-long input is
rejected up front, so very long lines cannot cause catastrophic backtracking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Longest string that can be a single date ("30 September 2026" is 17 chars).
_MAX_DATE_LEN = 40

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass(frozen=True)
class ParsedDate:
    value: date | None   # None if it could not be read safely
    raw: str             # text exactly as printed
    year_inferred: bool  # True for "4 Apr" style where the year came from the period


# --- single-date patterns (fully anchored) --------------------------------
_SLASH_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_ISO_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DASH_RE = re.compile(r"(\d{1,2})-(\d{1,2})-(\d{2}|\d{4})")
_DMONY_RE = re.compile(r"(\d{1,2})[ \t]{1,5}([A-Za-z]{3,9})[ \t]{1,5}(\d{4})")
_DMON_RE = re.compile(r"(\d{1,2})[ \t]{1,5}([A-Za-z]{3,9})")

# Unanchored token used to find the two ends inside a period line.
_DATE_TOKEN = (
    r"(?:\d{1,2}/\d{1,2}/\d{4}"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}-\d{1,2}-(?:\d{4}|\d{2})"
    r"|\d{1,2}[ \t]{1,5}[A-Za-z]{3,9}[ \t]{1,5}\d{4})"
)
_PERIOD_RE = re.compile(
    r"statement[ \t]{1,10}period[ \t]{0,5}:?[ \t]{0,5}"
    rf"({_DATE_TOKEN})"
    r"[ \t]{0,5}(?:-|\u2013|\u2014|to)[ \t]{0,5}"
    rf"({_DATE_TOKEN})",
    re.IGNORECASE,
)


def _month_number(name: str) -> int | None:
    key = name.lower()
    num = _MONTHS.get(key[:3])
    if num is None:
        return None
    # Accept the abbreviation, "Sept", or the full month name only.
    full = (
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    )[num - 1]
    if key == key[:3] or key == "sept" or key == full:
        return num
    return None


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _read_full(text: str) -> date | None:
    """Read one full date; day-first for numeric d/m/y styles."""
    if len(text) > _MAX_DATE_LEN:
        return None
    s = text.strip()
    m = _SLASH_RE.fullmatch(s)
    if m:
        return _make_date(int(m[3]), int(m[2]), int(m[1]))
    m = _ISO_RE.fullmatch(s)
    if m:
        return _make_date(int(m[1]), int(m[2]), int(m[3]))
    m = _DASH_RE.fullmatch(s)
    if m:
        year = int(m[3])
        if len(m[3]) == 2:
            year += 2000  # yy is read as 20yy
        return _make_date(year, int(m[2]), int(m[1]))
    m = _DMONY_RE.fullmatch(s)
    if m:
        month = _month_number(m[2])
        if month is None:
            return None
        return _make_date(int(m[3]), month, int(m[1]))
    return None


def parse_full_date(raw: str) -> ParsedDate:
    """Parse dd/mm/yyyy, yyyy-mm-dd, d(d) Mon yyyy, dd-mm-yy and dd-mm-yyyy."""
    return ParsedDate(value=_read_full(raw), raw=raw, year_inferred=False)


def parse_period(text: str) -> tuple[ParsedDate, ParsedDate] | None:
    """Find ``Statement Period <start> to <end>`` (colon and separator optional)."""
    m = _PERIOD_RE.search(text)
    if not m:
        return None
    first, second = m.group(1), m.group(2)
    return (
        ParsedDate(_read_full(first), first, False),
        ParsedDate(_read_full(second), second, False),
    )


def parse_day_month(raw: str, period_start: date, period_end: date) -> ParsedDate:
    """Parse ``d Mon`` and take the year from the period.

    Tries each year the period touches (so a Dec-to-Jan period gives Dec the
    start year and Jan the end year). If no candidate falls inside the period
    the value stays ``None``; a year is never picked silently.
    """
    unread = ParsedDate(None, raw, True)
    if len(raw) > _MAX_DATE_LEN:
        return unread
    m = _DMON_RE.fullmatch(raw.strip())
    if not m:
        return unread
    month = _month_number(m[2])
    if month is None:
        return unread
    day = int(m[1])
    for year in range(period_start.year, period_end.year + 1):
        candidate = _make_date(year, month, day)
        if candidate is not None and period_start <= candidate <= period_end:
            return ParsedDate(candidate, raw, True)
    return unread
