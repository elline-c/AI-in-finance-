"""Style profile read back from a PDF (design finding F6).

The corpus manifests hold no style values, so every value here is read from the PDF
itself, in memory. Nothing is written to disk. ``pypdf`` and ``pdfplumber`` are imported
inside the function so that importing this module stays light.
"""
from __future__ import annotations

import io
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

# Catalogues are plain tuples in the generator; importing them pulls reportlab, so the
# imports are done lazily inside the functions that need them.

_STREET_RE = re.compile(
    r"\b\d{1,4} [A-Z][A-Za-z]+ (?:Road|Street|Avenue|Lane|Drive|Way|Crescent|Close|Walk|Link)\b")
_POSTCODE_RE = re.compile(r"\bSINGAPORE \d{6}\b", re.IGNORECASE)
_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


@dataclass(frozen=True)
class StyleProfile:
    producer: str
    producer_family: str                      # "base", "export", "editor" or "other"
    creator: str
    fonts: Tuple[str, ...]                    # sorted, subset prefixes removed
    bank_name: Optional[str]
    street: Optional[str]                     # printed street line, None if the layout has none
    second_address_line: Optional[str]        # printed postcode line, None if absent
    row_step_bucket: Optional[float]          # median row spacing, to 0.1 pt
    creation_hour: Optional[int]
    page_size: Tuple[float, float]
    footer_text: Optional[str]


def _family(producer: str) -> str:
    from fraud_detect.corpus.finish import PRODUCERS
    from fraud_detect.corpus.tamper.pdf_ops import EDITOR_PRODUCERS, EXPORT_PRODUCERS
    if producer in PRODUCERS:
        return "base"
    if producer in EXPORT_PRODUCERS:
        return "export"
    if producer in EDITOR_PRODUCERS:
        return "editor"
    return "other"


def _lines(page) -> list:
    """Text lines of a page as (top, text), top to bottom."""
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    rows: dict = defaultdict(list)
    for w in words:
        rows[round(w["top"], 0)].append(w)
    out = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w["x0"])
        out.append((statistics.mean(w["top"] for w in ws), " ".join(w["text"] for w in ws)))
    return out


def _row_step(pages) -> Optional[float]:
    gaps = []
    for page in pages:
        tops = sorted({round(w["top"], 2) for w in page.extract_words()})
        # Keep only table-like gaps: consecutive distinct line tops 10 to 16 pt apart.
        gaps.extend(b - a for a, b in zip(tops, tops[1:]) if 10.0 <= b - a <= 16.0)
    if not gaps:
        return None
    return round(statistics.median(gaps) * 10.0) / 10.0


def style_profile(source: Union[bytes, bytearray, io.IOBase, str, Path]) -> StyleProfile:
    """Read the styling values from one PDF (bytes, file object or path)."""
    import pdfplumber
    from pypdf import PdfReader
    from fraud_detect.corpus.finish import BANK_NAMES

    if isinstance(source, (bytes, bytearray)):
        stream = io.BytesIO(bytes(source))
    elif isinstance(source, (str, Path)):
        stream = io.BytesIO(Path(source).read_bytes())
    else:
        stream = io.BytesIO(source.read())

    meta = PdfReader(io.BytesIO(stream.getvalue())).metadata or {}
    producer = str(meta.get("/Producer") or "")
    creator = str(meta.get("/Creator") or "")
    author = str(meta.get("/Author") or "")
    created = _PDF_DATE_RE.match(str(meta.get("/CreationDate") or ""))
    creation_hour = int(created.group(4)) if created else None

    with pdfplumber.open(io.BytesIO(stream.getvalue())) as pdf:
        pages = list(pdf.pages)
        first = pages[0]
        page_size = (round(float(first.width), 1), round(float(first.height), 1))
        fonts = sorted({_SUBSET_PREFIX.sub("", c.get("fontname", ""))
                        for p in pages for c in p.chars if c.get("fontname")})
        text = first.extract_text() or ""
        lines = _lines(first)
        row_step = _row_step(pages)

    bank = next((b for b in BANK_NAMES if b in text), None)
    if bank is None and author in BANK_NAMES:
        bank = author
    street_m = _STREET_RE.search(text)
    post_m = _POSTCODE_RE.search(text)
    footer = lines[-1][1] if lines else None

    return StyleProfile(
        producer=producer,
        producer_family=_family(producer),
        creator=creator,
        fonts=tuple(fonts),
        bank_name=bank,
        street=street_m.group(0) if street_m else None,
        second_address_line=post_m.group(0) if post_m else None,
        row_step_bucket=row_step,
        creation_hour=creation_hour,
        page_size=page_size,
        footer_text=footer,
    )
