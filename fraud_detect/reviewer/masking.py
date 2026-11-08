"""What the model is allowed to see: masked identifiers and defanged document text.

The model runs on this machine (see ``llm_client``), so masking is a second layer, not the
only protection. It still matters: it keeps a company's identity out of prompts, logs and
stored notes, and it keeps text from the document from steering the model.

Two jobs:

* ``mask``      replaces identifiers (account holder, registered name, UEN, account and ID
                numbers, e-mail addresses, phone numbers) with fixed placeholders.
* ``defang``    treats text taken from the uploaded document, or typed by a person, as
                untrusted data: control characters and chat-template tokens go, instruction-
                like phrases are redacted (using the security layer's own patterns), and the
                length is capped.

Amounts, dates, page numbers and counterparty descriptions are kept: they are the substance
of the explanation.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, Optional

from ..models import Finding
from ..nodes.security import _INSTRUCTION_PATTERNS

HOLDER = "[ACCOUNT HOLDER]"
OTHER_HOLDER = "[OTHER ACCOUNT HOLDER]"
REGISTERED = "[REGISTERED NAME]"
REDACTED = "[instruction-like text removed]"

MAX_EVIDENCE_CHARS = 400
MAX_TITLE_CHARS = 150
MAX_NOTE_CHARS = 300

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_TEMPLATE_TOKEN_RE = re.compile(r"<\|[^|>]{0,40}\|>|\[/?INST\]|<</?SYS>>|</?s>", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_NRIC_RE = re.compile(r"\b[STFGM]\d{7}[A-Z]\b")
_UEN_RE = re.compile(r"\b(?:\d{8,9}[A-Z]|[TSR]\d{2}[A-Z]{2}\d{4}[A-Z])\b")
# A comma or dot only counts as part of a number when a digit sits on its other side, so a
# phone number at the end of a sentence ("call 9123 4567, then") is still found while
# "9,123,456" and "9123.4567" are left alone.
_PHONE_RE = re.compile(r"(?<!\d)(?<!\d,)(?<!\d\.)(?:\+?65[\s-]?)?[689]\d{3}[\s-]?\d{4}(?!\d)(?!,\d)(?!\.\d)")
_LONG_DIGITS_RE = re.compile(r"(?<![\d,.])\d{9,}(?![\d,]*\d)")
_ENTITY_TAIL_RE = re.compile(
    r"[\s,.]*\b(?:pte\.?\s*ltd\.?|private\s+limited|limited|ltd\.?|llp|llc|inc\.?|corp\.?)\s*$",
    re.IGNORECASE)


def _name_variants(name: Optional[str]) -> list[str]:
    """The name as given and without its entity suffix ('Pte Ltd'), longest first."""
    if not name or len(name.strip()) < 3:
        return []
    full = " ".join(name.split())
    short = _ENTITY_TAIL_RE.sub("", full).strip(" ,.")
    out = {full}
    if len(short) >= 4 and " " in short:      # a single common word would over-match
        out.add(short)
    return sorted(out, key=len, reverse=True)


def _replace_names(text: str, names: Iterable[Optional[str]], placeholder: str) -> str:
    for name in names:
        for variant in _name_variants(name):
            text = re.sub(re.escape(variant).replace(r"\ ", r"\s+"), placeholder, text,
                          flags=re.IGNORECASE)
    return text


def mask(text: str, *, holder: Optional[str] = None, registered: Optional[str] = None,
         uen: Optional[str] = None, other_holders: Iterable[Optional[str]] = ()) -> str:
    """``text`` with identifiers replaced by placeholders."""
    t = text or ""
    t = _replace_names(t, [holder], HOLDER)
    t = _replace_names(t, [registered], REGISTERED)
    t = _replace_names(t, other_holders, OTHER_HOLDER)
    if uen:
        t = re.sub(re.escape(uen), "[UEN]", t, flags=re.IGNORECASE)
    t = _EMAIL_RE.sub("[EMAIL]", t)
    t = _NRIC_RE.sub("[ID NUMBER]", t)
    t = _UEN_RE.sub("[UEN]", t)
    t = _PHONE_RE.sub("[PHONE]", t)
    t = _LONG_DIGITS_RE.sub("[ACCOUNT NUMBER]", t)
    return t


def defang(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    """Make untrusted text safe to place inside a prompt as quoted data."""
    t = html.unescape(text or "")
    t = _CONTROL_RE.sub("", t)
    t = _TEMPLATE_TOKEN_RE.sub("", t)
    t = t.replace("\u201c", '"').replace("\u201d", '"')   # curly quotes are reserved for verified quotes
    for _cat, _label, rx in _INSTRUCTION_PATTERNS:
        t = rx.sub(REDACTED, t)
    t = " ".join(t.split())
    if len(t) > limit:
        t = t[: max(0, limit - 1)].rstrip() + "\u2026"
    return t


def clean_text(text: str, *, limit: int = MAX_EVIDENCE_CHARS, **mask_kwargs) -> str:
    """Mask identifiers first, then defang, so a placeholder can never be cut in half."""
    return defang(mask(html.unescape(text or ""), **mask_kwargs), limit)


def finding_evidence(f: Finding, **mask_kwargs) -> str:
    """Evidence text of a finding as the model may see it.

    A security finding quotes the hidden instruction it found; that quotation is the one
    thing that must never reach the model, so only the fact of the finding is passed on.
    """
    if f.code == "SEC_EMBEDDED_INSTRUCTION":
        where = f.field or "the document"
        return f"Instruction-like text was found in {where}. Its content is withheld."
    return clean_text(f.evidence, **mask_kwargs)
