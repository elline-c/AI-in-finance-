"""Node 2 — Visual Forensics Agent.

Inspects the document for signs of digital tampering that live in the *rendering*
rather than the numbers:

    * Font consistency  — a small cluster of glyphs in a different font family
      than the body text is a classic overwrite / copy-paste signature.
    * Format consistency — mixed date formats or mixed currency symbols within
      one statement suggest content was edited or spliced.
    * Metadata inconsistency — editing-software producers (Photoshop, Illustrator,
      GIMP) and a document creation date that predates the statement period.

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
_EDITING_SOFTWARE = ["photoshop", "illustrator", "gimp", "inkscape", "acdsee", "corel"]


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

    # --- 3. Currency-symbol consistency -----------------------------------
    symbols = set(re.findall(r"(SGD|S\$|\bUSD\b|US\$|\$)", ing.raw_text))
    # Normalise obvious equivalents for counting distinct "styles".
    styles = set()
    for s in symbols:
        if s in ("SGD",):
            styles.add("SGD")
        elif s in ("S$", "$"):
            styles.add("$-symbol")
        else:
            styles.add(s)
    if len(styles) > 1:
        findings.append(Finding(
            agent=AGENT, code="VF_CURRENCY_MIX",
            title="Inconsistent currency notation",
            severity=Severity.MEDIUM, weight=10,
            evidence=(f"Mixed currency notation found ({sorted(styles)}). "
                      f"Amounts within one account should use a single notation."),
        ))

    # --- 4. Metadata: editing-software trace ------------------------------
    prod = (ing.metadata.producer or "") + " " + (ing.metadata.creator or "")
    prod_l = prod.lower()
    for sw in _EDITING_SOFTWARE:
        if sw in prod_l:
            findings.append(Finding(
                agent=AGENT, code="VF_EDIT_SOFTWARE",
                title="Image-editing software trace in metadata",
                severity=Severity.HIGH, weight=24,
                evidence=(f"Document producer/creator metadata references '{sw}' "
                          f"({ing.metadata.producer!r}). Genuine bank statements are "
                          f"produced by banking/report systems, not image editors."),
            ))
            break

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
