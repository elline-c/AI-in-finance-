"""Header reader (Requirements 4.1-4.9).

Reads holder, UEN, account number, currency, period, statement date, opening,
closing and the printed totals from one section (one page's lines and words).
Every field that cannot be read safely is ``None``; nothing is guessed.

Strategies, in order: same-line ``Label: value``; left/right pair split at
``Account No.`` (the left half is the holder); summary row mapped by label
(plain text first, word-box x-position as a fallback).

Regexes are anchored or bounded and lines are truncated to ``_MAX_LINE``
characters before any pattern runs, so very long lines are safe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ...models import Finding, Severity
from .dates import _DATE_TOKEN, ParsedDate, parse_full_date, parse_period
from .money import parse_money
from .types import Line, Word

AGENT = "Node1:Ingestion"
_MAX_LINE = 2000


@dataclass
class SectionHeader:
    holder: Optional[str] = None
    uen: Optional[str] = None
    account_number: Optional[str] = None
    currency: Optional[str] = None
    period_start: Optional[ParsedDate] = None
    period_end: Optional[ParsedDate] = None
    statement_date: Optional[ParsedDate] = None
    opening: Optional[float] = None
    closing: Optional[float] = None
    printed_total_withdrawals: Optional[float] = None
    printed_total_deposits: Optional[float] = None
    page: int = 1


# ---- patterns ----------------------------------------------------------
_ACCOUNT_NO_RE = re.compile(r"\baccount[ \t]{1,3}(?:no\b\.?|number\b)", re.IGNORECASE)
_HOLDER_LABEL_RE = re.compile(
    r"\s{0,5}account[ \t]{1,3}(?:holder|name)\b[ \t]{0,3}:?[ \t]{0,3}(.{1,200})",
    re.IGNORECASE,
)
_UEN_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9-])(?:company[ \t]{1,3}uen|uen|co\.?[ \t]{0,2}reg\.?[ \t]{0,2}no\.?)"
    r"(?![A-Za-z])[ \t]{0,3}:?[ \t]{0,3}([^\s]{1,40})",
    re.IGNORECASE,
)
# Singapore UEN: 8-9 digits + letter (9-10 chars) or T/S/R + yy + 2 letters + 4 digits + letter.
_UEN_VALUE_RE = re.compile(r"(?:\d{8,9}[A-Z]|[TSR]\d{2}[A-Z]{2}\d{4}[A-Z])", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"\bcurrency\b[ \t]{0,3}:?[ \t]{0,3}([A-Za-z]{3})\b", re.IGNORECASE)
_STATEMENT_DATE_RE = re.compile(
    rf"statement[ \t]{{1,3}}date[ \t]{{0,3}}:?[ \t]{{0,3}}({_DATE_TOKEN})", re.IGNORECASE
)
_ACCOUNT_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 \-/]{2,39}")
_TRAILING_PAREN_RE = re.compile(r"\s{0,5}\([^)]{0,40}\)\s{0,5}$")

# not a holder: titles, product line, page marker, specimen marker
_NOT_HOLDER_RE = re.compile(
    r"statement[ \t]{1,3}of[ \t]{1,3}account|business[ \t]{1,3}current[ \t]{1,3}account|"
    r"specimen|page[ \t]{1,3}\d{1,4}[ \t]{1,3}of[ \t]{1,3}\d{1,4}",
    re.IGNORECASE,
)

# summary-row labels: key -> alternation
_SUMMARY_LABELS = {
    "opening": r"opening[ \t]{1,3}balance",
    "withdrawals": r"total[ \t]{1,3}(?:withdrawals|debits|money[ \t]{1,3}out)",
    "deposits": r"total[ \t]{1,3}(?:deposits|credits|money[ \t]{1,3}in)",
    "closing": r"closing[ \t]{1,3}balance",
}
_SUMMARY_RE = re.compile(
    "|".join(f"(?P<{k}>{v})" for k, v in _SUMMARY_LABELS.items()), re.IGNORECASE
)
_SAME_LINE_BALANCE_RE = {
    k: re.compile(rf"(?<![A-Za-z]){v}[ \t]{{0,3}}:[ \t]{{0,3}}([^\n]{{1,64}})", re.IGNORECASE)
    for k, v in _SUMMARY_LABELS.items()
}
_TWO_DP_RE = re.compile(r"\.\d{2}(?!\d)")
_CURRENCY_TOKENS = {"sgd", "s$", "$"}
_SUFFIX_TOKENS = {"cr", "dr"}
_CELL_GAP = 4.0       # words closer than this (points) belong to one amount cell
_ROW_TOLERANCE = 3.0  # words within this vertical distance are on the same line

_FIELD_OF = {
    "opening": "opening",
    "withdrawals": "printed_total_withdrawals",
    "deposits": "printed_total_deposits",
    "closing": "closing",
}


def _is_holder_text(text: str) -> bool:
    t = text.strip(" \t:-")
    return bool(t) and _NOT_HOLDER_RE.search(t) is None


def _split_at_account_no(text: str) -> Optional[tuple[str, str]]:
    m = _ACCOUNT_NO_RE.search(text)
    if m is None:
        return None
    return text[: m.start()].strip(), text[m.end():].strip()


def _clean_account(right: str) -> Optional[str]:
    value = _TRAILING_PAREN_RE.sub("", right.lstrip(" \t:")).strip()
    return value if _ACCOUNT_VALUE_RE.fullmatch(value) else None


# ---- holder / account number ------------------------------------------
def _read_holder_and_account(texts: list[str]) -> tuple[Optional[str], Optional[str]]:
    holder: Optional[str] = None
    account: Optional[str] = None
    labelled: Optional[str] = None
    for text in texts:
        if labelled is None:
            m = _HOLDER_LABEL_RE.fullmatch(text)
            if m:
                value = m.group(1)
                split = _split_at_account_no(value)
                if split is not None:
                    value = split[0]
                value = value.strip()
                if _is_holder_text(value):
                    labelled = value
        split = _split_at_account_no(text)
        if split is None:
            continue
        left, right = split
        if account is None:
            account = _clean_account(right)
        if holder is None and left and not _HOLDER_LABEL_RE.fullmatch(text) and _is_holder_text(left):
            holder = left
    return (labelled or holder), account


# ---- UEN / currency / dates -------------------------------------------
def _read_uen(texts: list[str]) -> Optional[str]:
    for text in texts:
        for m in _UEN_LABEL_RE.finditer(text):
            value = m.group(1).strip(".,;")
            if _UEN_VALUE_RE.fullmatch(value):
                return value.upper()
    return None


def _read_currency(texts: list[str]) -> Optional[str]:
    for text in texts:
        m = _CURRENCY_RE.search(text)
        if m:
            return m.group(1).upper()
    return None


def _read_period(texts: list[str]) -> tuple[Optional[ParsedDate], Optional[ParsedDate]]:
    for text in texts:
        found = parse_period(text)
        if found is not None:
            return found
    return None, None


def _read_statement_date(texts: list[str]) -> Optional[ParsedDate]:
    for text in texts:
        m = _STATEMENT_DATE_RE.search(text)
        if m:
            return parse_full_date(m.group(1))
    return None


# ---- summary row -------------------------------------------------------
def _labels_in(text: str) -> list[tuple[str, str]]:
    """(key, printed label text) in order of appearance."""
    return [(m.lastgroup, m.group(0)) for m in _SUMMARY_RE.finditer(text) if m.lastgroup]


def _amounts_from_text(values_text: str, count: int) -> Optional[list[float]]:
    tokens: list[str] = []
    for tok in values_text.split():
        low = tok.lower()
        if tokens and low in _SUFFIX_TOKENS:
            tokens[-1] += " " + tok
        elif tokens and tokens[-1].lower() in _CURRENCY_TOKENS:
            tokens[-1] += tok
        else:
            tokens.append(tok)
        if len(tokens) > count:
            return None
    if len(tokens) != count:
        return None
    out: list[float] = []
    for tok in tokens:
        if _TWO_DP_RE.search(tok) is None:
            return None
        money = parse_money(tok)
        if money is None:
            return None
        out.append(money.value)
    return out


def _label_x0(words: list[Word], label_text: str) -> Optional[float]:
    toks = [t.lower() for t in label_text.split()]
    for i in range(len(words) - len(toks) + 1):
        if all(words[i + j].text.lower() == toks[j] for j in range(len(toks))):
            return words[i].x0
    return None


def _amounts_from_words(
    words: list[Word], label_line: Line, values_line: Line, labels: list[tuple[str, str]]
) -> dict[str, float]:
    label_words = sorted(
        (w for w in words if abs(w.top - label_line.top) <= _ROW_TOLERANCE), key=lambda w: w.x0
    )
    value_words = sorted(
        (w for w in words if abs(w.top - values_line.top) <= _ROW_TOLERANCE), key=lambda w: w.x0
    )
    columns: list[tuple[float, str]] = []
    for key, text in labels:
        x0 = _label_x0(label_words, text)
        if x0 is None:
            return {}
        columns.append((x0, key))
    columns.sort()
    # group adjacent words into amount cells ("SGD" + "10,000.00")
    cells: list[tuple[float, float, str]] = []
    for w in value_words:
        if cells and w.x0 - cells[-1][1] <= _CELL_GAP:
            x0, _, txt = cells[-1]
            cells[-1] = (x0, w.x1, txt + w.text)
        else:
            cells.append((w.x0, w.x1, w.text))
    found: dict[str, float] = {}
    seen: set[str] = set()
    for x0, x1, txt in cells:
        centre = (x0 + x1) / 2
        key = columns[0][1]
        for col_x0, col_key in columns:
            if col_x0 <= centre:
                key = col_key
        if key in seen:  # two cells in one column: ambiguous
            found.pop(key, None)
            continue
        seen.add(key)
        money = parse_money(txt)
        if money is not None and _TWO_DP_RE.search(txt) is not None:
            found[key] = money.value
    return found


def _read_summary(lines: list[Line], words: list[Word]) -> dict[str, float]:
    for i, line in enumerate(lines[:-1]):
        labels = _labels_in(line.text[:_MAX_LINE])
        if len({k for k, _ in labels}) < 2:
            continue
        values_line = lines[i + 1]
        amounts = _amounts_from_text(values_line.text[:_MAX_LINE], len(labels))
        if amounts is not None:
            return {k: v for (k, _), v in zip(labels, amounts)}
        return _amounts_from_words(words, line, values_line, labels)
    return {}


def _same_line_balances(texts: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, rx in _SAME_LINE_BALANCE_RE.items():
        for text in texts:
            m = rx.search(text)
            if m:
                money = parse_money(m.group(1).strip())
                if money is not None:
                    out[key] = money.value
                    break
    return out


# ---- public API --------------------------------------------------------
def read_header(lines: list[Line], words: list[Word], profile: Any, page: int) -> SectionHeader:
    """Read one section's header. ``profile`` may be ``None`` and is not required."""
    texts = [ln.text[:_MAX_LINE] for ln in lines]
    holder, account = _read_holder_and_account(texts)
    start, end = _read_period(texts)
    balances = _same_line_balances(texts)
    balances.update(_read_summary(lines, words))
    header = SectionHeader(
        holder=holder,
        uen=_read_uen(texts),
        account_number=account,
        currency=_read_currency(texts),
        period_start=start,
        period_end=end,
        statement_date=_read_statement_date(texts),
        page=page,
    )
    for key, attr in _FIELD_OF.items():
        setattr(header, attr, balances.get(key))
    return header


def missing_field_findings(header: SectionHeader) -> list[Finding]:
    """Mandatory-field findings (holder, UEN, period, opening) in fixed order."""
    checks = [
        ("account holder", header.holder, "DQ_MISSING_HOLDER"),
        ("UEN", header.uen, "DQ_MISSING_UEN"),
        (
            "statement period",
            header.period_start.value if header.period_start is not None else None,
            "DQ_MISSING_PERIOD",
        ),
        ("opening balance", header.opening, "DQ_MISSING_OPENING"),
    ]
    findings: list[Finding] = []
    for label, value, code in checks:
        if value in (None, ""):
            findings.append(Finding(
                agent=AGENT, code=code, title=f"Missing mandatory field: {label}",
                severity=Severity.MEDIUM, weight=8, field=label,
                evidence=f"Could not read the {label} from the statement; flagged rather than inferred.",
            ))
    return findings
