"""Feature engineering for the statement-forgery model.

Features come straight from the PDF file so that TRAINING and INFERENCE use the
identical code path. They target the signals the dataset data card describes:
text-layer artifacts (hidden originals under overlays), PDF structure, metadata
(editor Producer/Creator, ModDate later than CreationDate), font inconsistency,
and arithmetic reconciliation of the transaction table.

Deliberately NOT used as features (they are label-derived — see data card):
    is_tampered, tamper_type, tamper_method, arithmetic_consistent,
    hidden_text_layer, metadata_altered, font_inconsistent, date_inconsistent,
    overlay_bg_matched, genuine_resaved, field_tampered, regions, original_value,
    new_value, bank_layout, bank_name
We re-derive equivalent evidence from the file itself instead.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import pdfplumber
from pypdf import PdfReader

# Inputs to the forgery model.
#
# ``n_distinct_date_formats`` and ``duplicate_amount_overlap`` are still computed by
# ``extract_features`` (the Node 6b evidence text uses them) but are NOT model
# inputs. In the training data genuine and forged documents have the same median
# for both (2 and 3); the model used them to recognise how the dataset's own
# generator prints a document (two date formats, a summary block that repeats
# numbers), so any statement printed differently scored as forged. Removing them
# cost nothing on held-out data (out-of-layout ROC-AUC 0.938 either way).
FEATURE_NAMES = [
    "n_pages",
    "n_text_objects",
    "n_distinct_fonts",
    "minor_font_share",          # glyph share of the least-common font
    "n_content_streams",         # extra content streams hint at overlay edits
    "has_editing_software",      # editor Producer/Creator present
    "moddate_after_creation",    # ModDate strictly later than CreationDate
    "n_money_tokens",
    "balance_reconcile_breaks",  # rows where running balance doesn't add up
    "balance_reconcile_rate",    # breaks / rows checked
    "n_currency_styles",
]

# Features that scale with how long a document is. Training documents are single
# months (median 28 transactions, at most 3 pages), so the model cannot judge a
# document outside the sizes it saw. Node 6b stays silent on such documents
# instead of guessing (see ``ml_statement``). Deliberately NOT the tamper-signal
# features: a large break count is evidence of forgery, not unfamiliarity.
SIZE_FEATURES = ["n_pages", "n_text_objects", "n_money_tokens"]

_EDITING_SOFTWARE = ["photoshop", "illustrator", "gimp", "inkscape", "acrobat",
                     "pdfescape", "pdf-xchange", "foxit", "libreoffice", "word"]

_DATE_PATTERNS = {
    "dmy_slash": r"\b\d{2}/\d{2}/\d{4}\b",
    "iso": r"\b\d{4}-\d{2}-\d{2}\b",
    "d_mon_y": r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\b",
    "dmy_dash2": r"\b\d{2}-\d{2}-\d{2}\b",
    "d_mon": r"\b\d{1,2}\s+[A-Za-z]{3}\b",
}

_MONEY_RE = re.compile(r"(?:SGD|S\$|\$)?\s?-?\d{1,3}(?:,\d{3})*\.\d{2}")


def _parse_pdf_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", str(raw))
    if not m:
        return None
    try:
        parts = [int(g) if g else 0 for g in m.groups()]
        y, mo, d, hh, mm, ss = parts
        return datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None


def _metadata_features(path: str) -> dict:
    prod = creator = ""
    created = mod = None
    n_streams = 0
    try:
        reader = PdfReader(path)
        meta = reader.metadata or {}
        prod = str(meta.get("/Producer", "") or "")
        creator = str(meta.get("/Creator", "") or "")
        created = _parse_pdf_date(meta.get("/CreationDate"))
        mod = _parse_pdf_date(meta.get("/ModDate"))
        for page in reader.pages:
            contents = page.get("/Contents")
            if contents is None:
                continue
            try:
                n_streams += len(contents) if isinstance(contents, list) else 1
            except TypeError:
                n_streams += 1
    except Exception:
        pass

    blob = (prod + " " + creator).lower()
    has_editor = 1.0 if any(s in blob for s in _EDITING_SOFTWARE) else 0.0
    mod_after = 1.0 if (created and mod and mod > created) else 0.0
    return {
        "has_editing_software": has_editor,
        "moddate_after_creation": mod_after,
        "n_content_streams": float(n_streams),
    }


def _money_to_float(tok: str) -> Optional[float]:
    cleaned = tok.replace("SGD", "").replace("S$", "").replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


_OPENING_LINE_RE = re.compile(r"BALANCE BROUGHT FORWARD|Opening Balance", re.IGNORECASE)
_ANY_DATE_RE = re.compile("|".join(_DATE_PATTERNS.values()))


def _reconcile_rows(lines: list[str]) -> tuple[int, float]:
    """Row-by-row balance reconciliation from the statement's text lines.

    A transaction row is a line that carries a date and at least two money values;
    its LAST two values are the amount and the running balance. The opening
    balance comes from a "BALANCE BROUGHT FORWARD" / "Opening Balance" line, so
    the first row is checked too. Header summary lines (several numbers, no date),
    page markers and closing totals are skipped. Both signs are tried, because
    some layouts print the amount unsigned in separate debit/credit columns.

    Returns ``(break_count, break_rate)``. Replaces the earlier approach of
    pairing every money value on the page as alternating amount/balance, which
    mis-paired the header summary and so reported many "breaks" on genuine
    documents.
    """
    prev: Optional[float] = None
    checked = breaks = 0
    for line in lines:
        vals = [v for v in (_money_to_float(t) for t in _MONEY_RE.findall(line)) if v is not None]
        if not vals:
            continue
        if _OPENING_LINE_RE.search(line):
            prev = vals[0]
            continue
        if len(vals) >= 2 and _ANY_DATE_RE.search(line):
            amount, balance = abs(vals[-2]), vals[-1]
            if prev is not None:
                checked += 1
                if not (abs(prev + amount - balance) <= 0.01 or abs(prev - amount - balance) <= 0.01):
                    breaks += 1
            prev = balance
    return breaks, (breaks / checked if checked else 0.0)


def _reconcile_rate(amounts: list[float], balances: list[float]) -> tuple[int, float]:
    """Count rows where balance[i] != balance[i-1] + amount[i] (within tolerance).

    Works on the aligned amount/balance sequences we can recover from the text.
    Returns (break_count, break_rate). Robust to sign conventions by testing both.
    """
    if len(balances) < 2 or len(amounts) < 2:
        return 0, 0.0
    n = min(len(amounts), len(balances))
    breaks = 0
    checked = 0
    for i in range(1, n):
        prev, cur, amt = balances[i - 1], balances[i], amounts[i]
        if prev is None or cur is None or amt is None:
            continue
        checked += 1
        ok = (abs((prev + amt) - cur) <= 0.01) or (abs((prev - amt) - cur) <= 0.01)
        if not ok:
            breaks += 1
    rate = breaks / checked if checked else 0.0
    return breaks, rate


def extract_features(path: str) -> Optional[dict[str, float]]:
    """Extract the forgery feature vector from a single PDF. Returns None if the
    file cannot be opened at all (caller treats that as a fail-safe skip)."""
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            font_counts: dict[str, int] = {}
            n_text_objects = 0
            text_parts: list[str] = []
            money_seq: list[float] = []
            for page in pdf.pages:
                for ch in page.chars:
                    name = (ch.get("fontname", "unknown") or "unknown").split("+")[-1]
                    font_counts[name] = font_counts.get(name, 0) + 1
                n_text_objects += len(page.chars)
                txt = page.extract_text() or ""
                text_parts.append(txt)
    except Exception:
        return None

    raw_text = "\n".join(text_parts)

    # Fonts
    total_glyphs = sum(font_counts.values()) or 1
    n_fonts = len(font_counts)
    minor_share = (min(font_counts.values()) / total_glyphs) if font_counts else 0.0

    # Dates
    fmt_hits = sum(1 for pat in _DATE_PATTERNS.values() if re.search(pat, raw_text))

    # Currency styles
    styles = set()
    if re.search(r"\bSGD\b", raw_text):
        styles.add("SGD")
    if re.search(r"S\$", raw_text):
        styles.add("S$")
    if re.search(r"(?<![A-Za-z])\$", raw_text):
        styles.add("$")

    # Money tokens + arithmetic reconciliation (best-effort from the text layer)
    money_tokens = _MONEY_RE.findall(raw_text)
    money_vals = [v for v in (_money_to_float(t) for t in money_tokens) if v is not None]
    # Heuristic: treat the sequence of money values as alternating amount/balance
    # is unreliable, so reconcile on the full ordered balance-like series.
    breaks, rate = _reconcile_rows(raw_text.splitlines())

    # Duplicate amount overlap: overlays often leave the original value in the
    # text layer AND the new value -> suspiciously repeated tokens.
    from collections import Counter
    counts = Counter(money_tokens)
    dup_overlap = float(sum(c - 1 for c in counts.values() if c > 1))

    meta = _metadata_features(path)

    feats = {
        "n_pages": float(n_pages),
        "n_text_objects": float(n_text_objects),
        "n_distinct_fonts": float(n_fonts),
        "minor_font_share": float(minor_share),
        "n_content_streams": meta["n_content_streams"],
        "has_editing_software": meta["has_editing_software"],
        "moddate_after_creation": meta["moddate_after_creation"],
        "n_distinct_date_formats": float(fmt_hits),
        "n_money_tokens": float(len(money_tokens)),
        "balance_reconcile_breaks": float(breaks),
        "balance_reconcile_rate": float(rate),
        "n_currency_styles": float(len(styles)),
        "duplicate_amount_overlap": dup_overlap,
    }
    return feats


def to_row(path: str) -> Optional[list[float]]:
    f = extract_features(path)
    if f is None:
        return None
    return [f[name] for name in FEATURE_NAMES]
