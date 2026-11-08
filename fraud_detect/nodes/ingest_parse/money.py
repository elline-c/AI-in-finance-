"""Money reader: ``S$5,365.00``, ``SGD 28,249.88``, ``-SGD 13.47``, ``3,900.18 CR``, ``(1,200.00)``.

Returns ``None`` for anything that is not a single, well-formed amount; it never guesses.
The pattern has no nested or overlapping quantifiers, and very long input is rejected
up front, so there is no catastrophic backtracking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_MAX_LEN = 64  # no real amount cell is longer than this

_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MONEY_RE = re.compile(
    r"(?P<open>\()?\s*"
    r"(?P<s1>[+-])?\s*"
    r"(?:SGD|S\$|\$)?\s*"
    r"(?P<s2>[+-])?\s*"
    rf"(?P<num>{_NUMBER})"
    r"\s*(?P<close>\))?"
    r"(?:\s*(?P<suffix>CR|DR))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Money:
    value: float                 # rounded to 2dp; negative for '-', parentheses or DR
    explicit_sign: Optional[int]  # -1, +1, or None when no sign was printed
    suffix: Optional[str]        # "CR" or "DR" when printed


def parse_money(raw: Optional[str]) -> Optional[Money]:
    if raw is None:
        return None
    text = raw.strip()
    if not text or len(text) > _MAX_LEN:
        return None
    m = _MONEY_RE.fullmatch(text)
    if m is None:
        return None
    # parentheses must be balanced
    if (m.group("open") is None) != (m.group("close") is None):
        return None
    s1, s2 = m.group("s1"), m.group("s2")
    if s1 and s2:  # e.g. "--5.00", "+-5.00"
        return None
    sign_char = s1 or s2
    paren = m.group("open") is not None
    if paren and sign_char:
        return None

    magnitude = float(m.group("num").replace(",", ""))
    explicit: Optional[int] = None
    if paren or sign_char == "-":
        explicit = -1
    elif sign_char == "+":
        explicit = 1

    suffix = m.group("suffix").upper() if m.group("suffix") else None
    value = -magnitude if explicit == -1 else magnitude
    if suffix == "DR":
        value = -abs(value)
    return Money(value=round(value, 2), explicit_sign=explicit, suffix=suffix)
