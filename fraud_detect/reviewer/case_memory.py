"""Past decided cases as retrieval context (retrieval-augmented generation, stage two).

Finds the earlier cases that look most like the current one, so the reviewer notes can say
"a person rejected a case with the same pattern, and wrote why". Rules that follow
Node 5's:

* only cases a human finally decided (APPROVED or REJECTED), never PENDING or NEEDS_INFO,
* only cases the audit chain confirms (see ``CaseStore.recall_decided_cases``),
* never the case being screened,
* the reviewer's text is treated as untrusted: unescaped, masked and defanged before use.

Similarity is plain set overlap on finding codes and transaction fingerprints, with a small
bonus for the same company. No model is involved, so the choice of cases is reproducible.
"""
from __future__ import annotations

import html
from typing import Iterable

from ..models import DecidedCase, RiskReport
from . import masking
from .context import SimilarCase

MAX_CASES = 3
MIN_SCORE = 0.25
W_CODES, W_FINGERPRINTS, W_COMPANY = 0.6, 0.4, 0.15
_IGNORED_PREFIXES = ("MEM_", "DQ_")      # memory findings come FROM past cases; DQ_ is reading noise


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _signal_codes(codes: Iterable[str]) -> set[str]:
    return {c for c in codes if not c.startswith(_IGNORED_PREFIXES)}


def score_case(report: RiskReport, past: DecidedCase) -> tuple[float, set[str], int, bool]:
    """(similarity, shared finding codes, shared fingerprints, same company)."""
    cur_codes = _signal_codes(f.code for f in report.findings)
    old_codes = _signal_codes(past.finding_codes)
    cur_fps = {fp.key() for fp in report.transaction_fingerprints}
    old_fps = set(past.fingerprint_keys)
    same = bool(report.uen and past.uen and report.uen == past.uen)
    score = (W_CODES * _jaccard(cur_codes, old_codes) + W_FINGERPRINTS * _jaccard(cur_fps, old_fps)
             + (W_COMPANY if same else 0.0))
    return score, cur_codes & old_codes, len(cur_fps & old_fps), same


def _safe_text(text: str | None, report: RiskReport, past: DecidedCase) -> str:
    return masking.clean_text(
        html.unescape(text or ""), limit=masking.MAX_NOTE_CHARS,
        holder=report.account_holder, uen=report.uen,
        other_holders=[past.account_holder])


def find_similar_cases(report: RiskReport, store, *, k: int = MAX_CASES,
                       min_score: float = MIN_SCORE) -> list[SimilarCase]:
    """The ``k`` most similar human-decided cases, best first, ties broken by case id."""
    scored = []
    for past in store.recall_decided_cases(exclude_case_id=report.case_id):
        score, shared, shared_fps, same = score_case(report, past)
        if score >= min_score:
            scored.append((score, past, shared, shared_fps, same))
    scored.sort(key=lambda t: (-t[0], t[1].case_id))
    return [SimilarCase(
        case_id=p.case_id, decision=p.decision, score=round(score, 3),
        shared_codes=tuple(sorted(shared)), shared_fingerprints=fps, same_company=same,
        reason=_safe_text(p.reason, report, p), feedback=_safe_text(p.feedback, report, p))
        for score, p, shared, fps, same in scored[:k]]
