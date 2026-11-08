"""What a narrative may talk about: the citation sources for one case.

Every claim in a reviewer narrative must cite sources by id. This module builds those
ids and keeps, for each, the text that supports claims about it. The validator checks
claims against it and the prompt shows the model the same ids, so a model cannot cite
something the case does not contain.

Id scheme
---------
``F<n>``   a finding (n = position in ``report.findings``, same as ``breakdown.finding_id``)
``A<n>``   a confidence adjustment (``A1`` is the starting point)
``BONUS`` ``FLOOR``   score components that are not findings
``ROUTE``  the routing decision and its cause
``K<n>``   curated knowledge (policy text, finding-code catalogue)
``P<n>``   a past case a human reviewer decided
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..models import Finding, RiskReport, Severity
from ..nodes import risk_synthesis as node4
from . import breakdown

MAX_PROMPT_FINDINGS = 20

KIND_FINDING = "finding"
KIND_ADJUSTMENT = "adjustment"
KIND_SCORE = "score"
KIND_ROUTE = "route"
KIND_KNOWLEDGE = "knowledge"
KIND_CASE = "case"

_ROW_RE = re.compile(r"\brow\s+(\d+)\b", re.IGNORECASE)
PAGE_RE = re.compile(r"\b(?:pages?|pp?\.?)\s*(\d+)(?:\s*(?:-|\u2013|to)\s*(\d+))?\b", re.IGNORECASE)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class KnowledgeDoc:
    """A curated passage retrieved for this case (policy text or a finding-code entry)."""
    doc_id: str               # stable id, e.g. "POL-ROUTING" or "CODE:FL_BALANCE_BREAK"
    title: str
    text: str


@dataclass(frozen=True)
class SimilarCase:
    """A past case decided by a human reviewer (APPROVED or REJECTED), already masked."""
    case_id: str
    decision: str             # "APPROVED" | "REJECTED"
    score: float              # similarity, 0..1
    shared_codes: tuple[str, ...] = ()
    shared_fingerprints: int = 0
    same_company: bool = False
    reason: str = ""          # reviewer's reason, masked and shortened
    feedback: str = ""


@dataclass
class Source:
    id: str
    kind: str
    label: str                                   # short human label shown next to a citation chip
    text: str = ""                               # everything a claim about this source may rely on
    pages: set[int] = field(default_factory=set)
    rows: set[int] = field(default_factory=set)
    codes: set[str] = field(default_factory=set)
    decision: Optional[str] = None               # past cases only
    finding_index: Optional[int] = None          # findings only
    weight: float = 0.0
    points: Optional[float] = None               # signed points of an adjustment / score component


@dataclass
class ExplainContext:
    report: RiskReport
    sources: dict[str, Source]
    knowledge: list[KnowledgeDoc]
    cases: list[SimilarCase]
    allowed_numbers: set[str]                    # numbers a claim may use without a source
    omitted_findings: int = 0                    # findings left out of the prompt (too many)
    forbidden_phrases: list[str] = field(default_factory=list)   # injected text that must not be echoed

    def ids(self, kind: Optional[str] = None) -> list[str]:
        return [s.id for s in self.sources.values() if kind is None or s.kind == kind]

    def finding_sources(self) -> list[Source]:
        return [s for s in self.sources.values() if s.kind == KIND_FINDING]


def norm_number(token: str) -> str:
    """'12,000' -> '12000', '26.0' -> '26', '5.50' -> '5.5'."""
    t = token.replace(",", "")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t


def numbers_in(text: str) -> set[str]:
    return {norm_number(m.group(0)) for m in NUM_RE.finditer(text)}


def page_numbers(text: str) -> set[int]:
    """Every page a text refers to ("page 3", "p.4", "pp. 2-5")."""
    out: set[int] = set()
    for m in PAGE_RE.finditer(text):
        first = int(m.group(1))
        out.add(first)
        if m.group(2):
            out |= set(range(first, min(int(m.group(2)), first + 50) + 1))
    return out


def _extracted_quote(evidence: str) -> Optional[str]:
    """The quoted document text inside a security finding's evidence, if any."""
    m = re.search(r'Extracted text:\s*"(.*)"\s*$', evidence, re.DOTALL)
    return m.group(1) if m else None


def _finding_source(i: int, f: Finding) -> Source:
    text = " ".join(str(x) for x in (f.code, f.title, f.severity.value, f.evidence,
                                     f.field or "", f"page {f.page}" if f.page else ""))
    rows = {int(m.group(1)) for m in _ROW_RE.finditer(" ".join([f.field or "", f.evidence]))}
    where = ", ".join(x for x in ((f"p.{f.page}" if f.page else ""), f.field or "") if x)
    return Source(id=breakdown.finding_id(i), kind=KIND_FINDING,
                  label=f"{f.code}" + (f" ({where})" if where else ""),
                  text=text, pages=({f.page} if f.page else set()) | page_numbers(text), rows=rows,
                  codes={f.code}, finding_index=i, weight=float(f.weight))


def _allowed_numbers(report: RiskReport) -> set[str]:
    nums: set[str] = {"1", "100", "25", "50", "75", str(node4.AUTO_CLEAR_MAX_RISK),
                      str(node4.MIN_CONFIDENCE_FOR_AUTO), str(breakdown.MEDIUM_FLOOR),
                      str(breakdown.CORROBORATION_BONUS), str(breakdown.CONFIDENCE_BASE),
                      str(report.risk_score), str(report.confidence_score),
                      str(len(report.findings))}
    codes: dict[str, int] = {}
    for f in report.findings:
        nums.add(norm_number(f"{f.weight:g}"))
        codes[f.code] = codes.get(f.code, 0) + 1
    nums.update(str(n) for n in codes.values())
    nums.add(str(sum(1 for f in report.findings if f.severity in (Severity.MEDIUM, Severity.HIGH))))
    ex = report.explanation
    if ex is not None:
        for l in ex.risk_lines:
            nums.add(norm_number(f"{abs(l.points):g}"))
        for l in ex.confidence_lines:
            nums.add(norm_number(f"{abs(l.points):g}"))
        nums.add(norm_number(f"{ex.risk_total:g}"))
        nums.add(norm_number(f"{ex.confidence_total:g}"))
        nums.add(str(len([l for l in ex.risk_lines if l.kind == "FINDING"])))
    return nums


def build_context(report: RiskReport, *, knowledge: Optional[list[KnowledgeDoc]] = None,
                  cases: Optional[list[SimilarCase]] = None) -> ExplainContext:
    """Sources, ids and allowed numbers for ``report``.

    The most heavily weighted findings come first when there are more than
    ``MAX_PROMPT_FINDINGS``; the rest are counted in ``omitted_findings`` and cannot be cited.
    """
    knowledge, cases = list(knowledge or []), list(cases or [])
    sources: dict[str, Source] = {}

    ranked = sorted(enumerate(report.findings), key=lambda p: (-p[1].weight, p[0]))
    kept = sorted(ranked[:MAX_PROMPT_FINDINGS], key=lambda p: p[0])
    for i, f in kept:
        s = _finding_source(i, f)
        sources[s.id] = s

    forbidden: list[str] = []
    for f in report.findings:
        if f.code == "SEC_EMBEDDED_INSTRUCTION":
            q = _extracted_quote(f.evidence)
            if q:
                forbidden.append(" ".join(q.lower().split()))

    ex = report.explanation
    if ex is not None:
        for l in ex.confidence_lines:
            sources[l.id] = Source(id=l.id, kind=KIND_ADJUSTMENT,
                                   label=f"{l.points:+g} {l.label}",
                                   text=f"{l.code} {l.label} {l.reading} {l.points:+g}",
                                   codes={l.code}, points=float(l.points))
        for l in ex.risk_lines:
            if l.id in ("BONUS", "FLOOR"):
                sources[l.id] = Source(id=l.id, kind=KIND_SCORE, label=l.label,
                                       text=f"{l.label} {l.reading} {l.points:+g}")
        route_text = f"{ex.routing.primary} {ex.routing.statement} {report.routing_reason}"
        sources["ROUTE"] = Source(id="ROUTE", kind=KIND_ROUTE, label="Routing decision",
                                  text=route_text)

    for n, doc in enumerate(knowledge, 1):
        sources[f"K{n}"] = Source(id=f"K{n}", kind=KIND_KNOWLEDGE, label=doc.title,
                                  text=f"{doc.title} {doc.text}",
                                  codes=set(re.findall(r"\b(?:DQ|FL|VF|MEM|ML|SEC)_[A-Z0-9_]+\b",
                                                       doc.text)))
    for n, c in enumerate(cases, 1):
        text = (f"{c.case_id} {c.decision} {' '.join(c.shared_codes)} "
                f"{c.reason} {c.feedback} {c.shared_fingerprints}")
        sources[f"P{n}"] = Source(id=f"P{n}", kind=KIND_CASE,
                                  label=f"{c.case_id} ({c.decision.title()})",
                                  text=text, codes=set(c.shared_codes), decision=c.decision)

    return ExplainContext(report=report, sources=sources, knowledge=knowledge, cases=cases,
                          allowed_numbers=_allowed_numbers(report),
                          omitted_findings=len(report.findings) - len(kept),
                          forbidden_phrases=forbidden)
