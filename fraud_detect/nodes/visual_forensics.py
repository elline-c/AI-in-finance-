"""Node 2 — Visual Forensics Agent.

Inspects the document for signs of digital tampering that live in the *rendering*
rather than the numbers:

    * Font consistency  — a small cluster of glyphs in a different font family
      than the body text is a classic overwrite / copy-paste signature.
    * Format consistency — mixed date formats, or mixed currency notation in the
      transaction amounts, within one statement suggest content was edited or spliced.
    * Metadata inconsistency — editing-software producers (Photoshop, Illustrator,
      PDF editors), a file modified after it was created by something other than a
      print/export tool, and a creation date that predates the statement period.

In production this node would also run pixel-level ELA / copy-move detection on
rasterised pages; here we approximate with structural signals that are robust on
digital PDFs. Each signal is emitted as an explainable :class:`Finding`.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from ..models import AgentReport, Finding, IngestionResult, Severity

AGENT = "Node2:VisualForensics"

# Fonts we treat as the legitimate "body" of a bank statement.
_BODY_FONTS = {"Helvetica", "Helvetica-Bold", "Arial", "Arial-Bold", "Times-Roman"}

# Producer / creator substrings that indicate image-editing software.
_EDITING_SOFTWARE = ["photoshop", "illustrator", "gimp", "inkscape", "acdsee", "corel",
                     "pdfescape", "pdf-xchange", "pdf xchange", "foxit", "sejda",
                     "smallpdf", "ilovepdf", "pdfelement", "nitro pdf"]
# Whole word "editor" ("PDF Editor", "Foxit PDF Editor"). Adobe's PDF libraries are
# deliberately NOT listed: real banks generate statements with them.
_EDITOR_WORD = re.compile(r"\beditor\b")

# Producers that only re-save a file without changing its content: print-to-PDF
# drivers and "export / save as PDF" functions. A later ModDate from one of these is
# an ordinary re-save, not an edit.
_RESAVE_TOOLS = ["print", "export", "save as", "saveas", "quartz", "skia", "cups",
                 "driver", "browser", "chrome", "safari", "firefox"]
_MODIFIED_GAP_SECONDS = 60     # ModDate must be later than CreationDate by more than this


def _norm_font(name: str) -> str:
    # pdfplumber prefixes subset fonts like "ABCDEF+Helvetica"; strip that.
    return name.split("+")[-1]


def _parse_pdf_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})", raw)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _parse_pdf_datetime(raw: Optional[str]) -> Optional[datetime]:
    """PDF date with the time of day when present (``D:YYYYMMDDHHMMSS``); zone ignored."""
    if not raw:
        return None
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2})?)?", raw)
    if not m:
        return None
    y, mo, d, hh, mi, ss = (int(g) if g else 0 for g in m.groups())
    try:
        return datetime(y, mo, d, hh, mi, ss)
    except ValueError:
        return None


# A transaction row starts with a date in one of the layouts we read.
_ROW_START = re.compile(
    r"^\s*(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}-\d{1,2}-\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{4})?)\s"
)
# An amount: optional sign, optional currency marker, optional sign, number with 2 decimals.
_AMOUNT = re.compile(
    r"[+-]?\s*(?P<cur>US\$|USD|SGD|S\$|\$)?\s*[+-]?\s*\d[\d,]*\.\d{2}(?!\d)"
)


def _amount_styles(ing: IngestionResult) -> set[str]:
    """Currency notation used by the amounts in the TRANSACTION ROWS only.

    The account-header line ("Currency SGD") and the summary box are labels, not
    amounts, so they are ignored. An amount with no marker adds no style.
    """
    if ing.page_texts:
        text = "\n".join(p.text for p in ing.page_texts)
    else:
        text = ing.raw_text or ""
    styles: set[str] = set()
    for line in text.splitlines():
        if not _ROW_START.match(line):
            continue
        for m in _AMOUNT.finditer(line):
            cur = m.group("cur")
            if cur is None:
                continue
            styles.add("SGD" if cur == "SGD" else "$-symbol" if cur in ("S$", "$") else cur)
    return styles


def analyze(ing: IngestionResult) -> AgentReport:
    findings: list[Finding] = []

    # --- 1. Font-consistency (altered-text) check --------------------------
    aggregate: dict[str, int] = {}
    for fu in ing.fonts:
        for name, count in fu.fonts.items():
            aggregate[_norm_font(name)] = aggregate.get(_norm_font(name), 0) + count

    total_glyphs = sum(aggregate.values()) or 1
    for name, count in aggregate.items():
        share = count / total_glyphs
        # A non-body font that only appears in a tiny fraction of glyphs is the
        # tell-tale sign of a localised overwrite.
        if name not in _BODY_FONTS and share < 0.05 and count > 0:
            findings.append(Finding(
                agent=AGENT, code="VF_FONT_MISMATCH",
                title="Localised font inconsistency (possible altered text)",
                severity=Severity.HIGH, weight=22,
                evidence=(f"Font '{name}' appears in only {count} glyphs "
                          f"({share:.1%} of the document). Bank statements are typeset "
                          f"in a single family; an isolated foreign font is a common "
                          f"signature of digitally overwritten numbers/text."),
            ))

    # --- 2. Date-format consistency ---------------------------------------
    fmts = set()
    for t in ing.transactions:
        if not t.date_raw:
            continue
        if re.match(r"^\d{2}/\d{2}/\d{4}$", t.date_raw):
            fmts.add("dd/mm/yyyy")
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", t.date_raw):
            fmts.add("yyyy-mm-dd")
    if len(fmts) > 1:
        findings.append(Finding(
            agent=AGENT, code="VF_DATE_FORMAT_MIX",
            title="Inconsistent date formats within one statement",
            severity=Severity.MEDIUM, weight=14,
            evidence=(f"Multiple date formats detected in the transaction table: "
                      f"{sorted(fmts)}. A single bank system emits one date format; "
                      f"mixing suggests rows were edited or spliced in."),
        ))

    # --- 3. Currency-notation consistency (transaction amounts only) -------
    styles = _amount_styles(ing)
    if len(styles) > 1:
        findings.append(Finding(
            agent=AGENT, code="VF_CURRENCY_MIX",
            title="Inconsistent currency notation",
            severity=Severity.MEDIUM, weight=10,
            evidence=(f"Mixed currency notation found in the transaction amounts "
                      f"({sorted(styles)}). Amounts within one account should use a "
                      f"single notation. (The account-header currency label is not counted.)"),
        ))

    # --- 4. Metadata: editing-software trace ------------------------------
    prod = (ing.metadata.producer or "") + " " + (ing.metadata.creator or "")
    prod_l = prod.lower()
    hit = next((sw for sw in _EDITING_SOFTWARE if sw in prod_l), None)
    if hit is None and _EDITOR_WORD.search(prod_l):
        hit = "editor"
    edit_software_found = hit is not None
    if hit is not None:
        findings.append(Finding(
            agent=AGENT, code="VF_EDIT_SOFTWARE",
            title="Editing software trace in metadata",
            severity=Severity.HIGH, weight=24,
            evidence=(f"Document producer/creator metadata references '{hit}' "
                      f"({ing.metadata.producer!r}). Genuine bank statements are "
                      f"produced by banking/report systems, not editing tools."),
        ))

    # --- 4b. Metadata: modified after it was created, by something that edits ----
    created_dt = _parse_pdf_datetime(ing.metadata.creation_date)
    modified_dt = _parse_pdf_datetime(ing.metadata.mod_date)
    if (not edit_software_found and created_dt and modified_dt
            and (modified_dt - created_dt).total_seconds() > _MODIFIED_GAP_SECONDS):
        resave = next((t for t in _RESAVE_TOOLS if t in prod_l), None)
        if resave is None:
            findings.append(Finding(
                agent=AGENT, code="VF_MODIFIED_AFTER_CREATION",
                title="File changed after it was created",
                severity=Severity.MEDIUM, weight=14,
                evidence=(f"The file was last modified {modified_dt} but created "
                          f"{created_dt}, and its producer is {ing.metadata.producer!r}, "
                          f"which is not a print or export tool. A bank's statement system "
                          f"normally writes the file once; a later change by another program "
                          f"suggests it was opened and edited. (A plain re-save through a "
                          f"print or export tool is not flagged.)"),
            ))

    # --- 5. Metadata: creation date vs statement period -------------------
    created = _parse_pdf_date(ing.metadata.creation_date)
    if created and ing.statement_period_start:
        if created.date() < ing.statement_period_start:
            findings.append(Finding(
                agent=AGENT, code="VF_CREATION_BEFORE_PERIOD",
                title="Document created before the statement period",
                severity=Severity.HIGH, weight=18,
                evidence=(f"File creation date {created.date()} predates the statement "
                          f"period start {ing.statement_period_start}. The file cannot "
                          f"legitimately exist before the period it reports."),
            ))

    return AgentReport(agent=AGENT, findings=findings)
