"""Checks a reviewer narrative against the case it explains.

Three families of checks, which are also the evaluation criteria in ``evaluation.py``:

* **Faithfulness**: every claim cites sources that exist and fit the claim's role, and
  what the claim says (finding codes, page and row numbers, figures, a past case's
  outcome) is backed by the sources it cites. The biggest finding must not be left out.
* **Consistency**: the narrative does not contradict the actual scores, bands or
  routing, and does not issue a verdict. Only a human approves or rejects.
* **Injection**: nothing that looks like an instruction to the system, and no text lifted
  from an embedded document instruction, appears in the output.

Any violation makes the caller fall back to the deterministic template, so a model can
only ever add words, never change what the screening decided.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional

from ..models import Claim, ReviewerNarrative, Routing
from ..nodes import risk_synthesis as node4
from ..nodes.security import _INSTRUCTION_PATTERNS   # reused: same notion of "instruction-like"
from .context import (
    KIND_ADJUSTMENT, KIND_CASE, KIND_FINDING, KIND_KNOWLEDGE, KIND_SCORE, NUM_RE, PAGE_RE,
    ExplainContext, Source, norm_number, numbers_in,
)

# ---- limits ---------------------------------------------------------------- #
MAX_CLAIM_CHARS = 400
MAX_LONG_CHARS = 700          # confidence_reading and routing_explanation
MAX_DRIVERS, MAX_CHECKS, MAX_SIMILAR = 6, 4, 3
MAX_CITES = 8

# ---- violation kinds and their families ------------------------------------ #
STRUCTURE, FORMAT = "STRUCTURE", "FORMAT"
CITATION_UNKNOWN, CITATION_ROLE = "CITATION_UNKNOWN", "CITATION_ROLE"
FAITHFUL_CODE, FAITHFUL_PAGE, FAITHFUL_ROW = "FAITHFUL_CODE", "FAITHFUL_PAGE", "FAITHFUL_ROW"
FAITHFUL_NUMBER, FAITHFUL_OUTCOME, OMISSION = "FAITHFUL_NUMBER", "FAITHFUL_OUTCOME", "OMISSION"
FAITHFUL_QUOTE = "FAITHFUL_QUOTE"
CONSISTENCY_SCORE, CONSISTENCY_BAND = "CONSISTENCY_SCORE", "CONSISTENCY_BAND"
CONSISTENCY_ROUTING, VERDICT = "CONSISTENCY_ROUTING", "VERDICT"
CONSISTENCY_DIRECTION = "CONSISTENCY_DIRECTION"
INJECTION_PATTERN, INJECTION_ECHO = "INJECTION_PATTERN", "INJECTION_ECHO"

FAITHFULNESS = {CITATION_UNKNOWN, CITATION_ROLE, FAITHFUL_CODE, FAITHFUL_PAGE, FAITHFUL_ROW,
                FAITHFUL_NUMBER, FAITHFUL_OUTCOME, FAITHFUL_QUOTE, OMISSION}
CONSISTENCY = {CONSISTENCY_SCORE, CONSISTENCY_BAND, CONSISTENCY_ROUTING, CONSISTENCY_DIRECTION,
               VERDICT}
INJECTION = {INJECTION_PATTERN, INJECTION_ECHO}
STRUCTURAL = {STRUCTURE, FORMAT}


def family(kind: str) -> str:
    if kind in FAITHFULNESS:
        return "faithfulness"
    if kind in CONSISTENCY:
        return "consistency"
    if kind in INJECTION:
        return "injection"
    return "structure"


@dataclass(frozen=True)
class Violation:
    kind: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind} at {self.where}: {self.detail}"


# ---- patterns -------------------------------------------------------------- #
# The security layer's own list, minus "force_approval": that pattern matches the words
# "auto-clear", which are ordinary vocabulary in an explanation. Verdict-style wording is
# caught separately by _VERDICT_RES.
_OUTPUT_INSTRUCTION_PATTERNS = [(c, l, rx) for c, l, rx in _INSTRUCTION_PATTERNS
                                if c != "force_approval"]
_ID_RE = re.compile(r"\b(?:[FAKP]\d+|BONUS|FLOOR|ROUTE)\b")
_CODE_RE = re.compile(r"\b(?:DQ|FL|VF|MEM|ML|SEC|CONF|POL)_[A-Z0-9_]+\b")
_PAGE_RE = PAGE_RE
ROW_RE = _ROW_RE = re.compile(r"\brows?\s*#?\s*(\d+)\b", re.IGNORECASE)
# Internal labels that must not leak into prose. Finding codes (FL_, VF_, ...) are fine; these
# are enum names such as VERY_HIGH, which no analyst should have to read.
_RAW_LABEL_RE = re.compile(
    r"\b(?:VERY_HIGH|AUTO_CLEAR|HUMAN_REVIEW|MATERIAL_FINDING|LOW_CONFIDENCE|RISK_THRESHOLD|NEEDS_INFO)\b")
_BAD_TEXT_RE = re.compile(r"<|>|https?://|www\.|javascript:|```|\[\[|\]\]", re.IGNORECASE)

_SCORE_RE = {
    "risk": re.compile(r"\brisk(?:\s+score)?\s*(?:of|is|was|at|=|:)?\s*(\d{1,3})(?!\d|,)", re.IGNORECASE),
    "confidence": re.compile(r"\bconfidence(?:\s+score)?\s*(?:of|is|was|at|=|:)?\s*(\d{1,3})(?!\d|,)",
                             re.IGNORECASE),
}
_BAND_WORDS = r"(very high|high|medium|low)"
_BAND_RE = {
    "risk": (re.compile(rf"\brisk\s+(?:band|level|rating|score)?\s*(?:is|was|of|at|=|:)\s*(?:in the\s+)?{_BAND_WORDS}\b",
                        re.IGNORECASE),
             re.compile(rf"\b{_BAND_WORDS}[\s-]+risk\s+(?:band|level|rating|score|case|statement|application)\b",
                        re.IGNORECASE)),
    "confidence": (re.compile(rf"\bconfidence\s+(?:band|level|rating|score)?\s*(?:is|was|of|at|=|:)\s*(?:in the\s+)?{_BAND_WORDS}\b",
                              re.IGNORECASE),
                   re.compile(rf"\b{_BAND_WORDS}[\s-]+confidence\b", re.IGNORECASE)),
}

# "The statement is genuine." is a verdict; "genuine instalments" is just an adjective, so the
# word must end its clause. "Check whether the payments are genuine." is a suggested check, not a
# verdict: a conditional or verifying sentence is exempt (see _is_conditional).
_STATE_VERDICT_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:\w+\s+)?(?:fraudulent|forged|fake|genuine|authentic|legitimate)"
    r"(?=\s*(?:[.,;:!?)]|$|\band\b|\bbut\b|\bor\b))", re.IGNORECASE)
_CONDITIONAL_RE = re.compile(
    r"\b(?:whether|if|verify|verifies|verifying|confirm|confirms|confirming|check|checks|checking|"
    r"determine|establish|assess|ascertain|ensure|test|unclear|unknown)\b", re.IGNORECASE)


# "A past case approved the application after checking invoices" reports what a person decided
# about an EARLIER case. That is checked against the cited case's decision (FAITHFUL_OUTCOME);
# it is not a verdict on the application being screened. Only the past tense, with "the" (not
# "this") and an explicit past-case word in the same sentence, is exempt.
_PAST_DECISION_RE = re.compile(
    r"\b(?:approved|rejected|declined|denied)\s+the\s+(?:loan|application|applicant|customer|case|"
    r"statement|company)\b", re.IGNORECASE)
_PAST_CASE_WORD_RE = re.compile(r"\b(?:case|cases|earlier|previous|past|prior)\b", re.IGNORECASE)


def _is_report_of_a_past_decision(role: str, claim: Claim, ctx: ExplainContext, text: str, m) -> bool:
    if role != "similar_cases" or not _PAST_DECISION_RE.fullmatch(m.group(0)):
        return False
    if not any(x in ctx.sources and ctx.sources[x].kind == KIND_CASE for x in claim.cites):
        return False
    start = max(text.rfind(ch, 0, m.start()) for ch in ".;!?") + 1
    end_candidates = [i for i in (text.find(ch, m.end()) for ch in ".;!?") if i != -1]
    sentence = text[start:min(end_candidates) if end_candidates else len(text)]
    return bool(_PAST_CASE_WORD_RE.search(sentence))


def _is_conditional(text: str, pos: int) -> bool:
    """True when the sentence containing ``pos`` is asking or checking, not asserting."""
    start = max(text.rfind(ch, 0, pos) for ch in ".;!?") + 1
    return bool(_CONDITIONAL_RE.search(text[start:pos]))


_VERDICT_RES = [_STATE_VERDICT_RE] + [re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:approve|approved|approving|reject|rejected|rejecting|decline|declined|declining|deny|denied)\s+(?:the|this)\s+(?:loan|application|applicant|customer|case|statement|company)\b",
    r"\bshould\s+(?:be\s+)?(?:approved|rejected|declined|denied|cleared)\b",
    r"\brecommend(?:s|ed)?\s+(?:approv|reject|declin|den|clear)\w*",
    r"\bsafe\s+to\s+(?:approve|clear|proceed|lend)\b",
)]
_HUMAN_REVIEW_CONTRADICTIONS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:is|was|been|being|were)\s+auto[- ]?cleared\b",
    r"\bno\s+(?:further\s+|human\s+)?review\s+(?:is\s+)?(?:needed|required|necessary)\b",
    r"\broutes?d?\s+to\s+auto[- ]?clear\b",
)]
_AUTO_CLEAR_CONTRADICTIONS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:escalated|routed|referred|sent)\s+to\s+(?:a\s+)?(?:human|analyst|reviewer)\b",
    r"\brequires?\s+(?:a\s+)?human\s+review\b",
)]
_OUTCOME_WORDS = {"REJECTED": re.compile(r"\brejected\b", re.IGNORECASE),
                  "APPROVED": re.compile(r"\b(?:approved|cleared)\b", re.IGNORECASE)}

# ---- direction of confidence adjustments ----------------------------------- #
# "Raised by 10 points" must agree with the sign of the cited adjustment. Only verbs are used
# (bare "higher" / "lower" are ambiguous: "lower risk"). "Agents raised findings" is about
# findings, not about confidence, so it is blanked out first.
_UP_RE = re.compile(r"\b(?:rais(?:e|ed|es|ing)|rose|ris(?:e|es|ing)|increas(?:e|ed|es|ing)|"
                    r"boost(?:ed|s)?|improv(?:e|ed|es)|adds?|added|gain(?:ed|s)?|plus)\b", re.IGNORECASE)
_DOWN_RE = re.compile(r"\b(?:lowered|lowers|lowering|drop(?:ped|s)?|fell|falls|fallen|"
                      r"decreas(?:e|ed|es|ing)|reduc(?:e|ed|es|ing)|cut|subtract(?:ed|s)?|"
                      r"deduct(?:ed|s)?|declin(?:e|ed|es)|minus|penalis(?:e|ed)|penaliz(?:e|ed))\b",
                      re.IGNORECASE)
_STARTING_RE = re.compile(r"\b(?:start(?:s|ed|ing)?|initial(?:ly)?|baseline|base|began|begins?)\b",
                          re.IGNORECASE)
_SIGNED_RE = re.compile(r"(?<![\w.])([+\-\u2212])\s?\d")
_ABOUT_FINDINGS_RE = re.compile(
    r"\b(?:agents?|checks?|nodes?|detectors?)\s+(?:also\s+)?rais(?:e|ed|es)\b|"
    r"\brais(?:e|ed|es)\s+(?:\w+\s+){0,2}?(?:findings?|flags?|concerns?|issues?|signals?|evidence)\b",
    re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+|,?\s+\b(?:but|while|whereas|although|though|yet)\b\s+",
                              re.IGNORECASE)
_ID_MENTION_RE = re.compile(r"\b(A\d+)\b")
_TOPICS = {
    "CONF_DATA_QUALITY": re.compile(r"data[\s-]?quality|unreadable|reading problem|missing (?:header )?field|pars(?:e|ing)", re.IGNORECASE),
    "CONF_NO_TRANSACTIONS": re.compile(r"\btransactions?\b", re.IGNORECASE),
    "CONF_CORROBORATION": re.compile(r"\bagents?\b|\bagree|corroborat|independent|support each other", re.IGNORECASE),
    "CONF_SINGLE_DIMENSION": re.compile(r"one-sided|single|only (?:the )?(?:visual|financial)|visual or|one (?:dimension|side)", re.IGNORECASE),
    "CONF_THIN_EVIDENCE": re.compile(r"\bthin|only one finding|one finding|boundary", re.IGNORECASE),
}

_BAND_NAMES = {"VERY_HIGH": "very high", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}


# ---- normalising ----------------------------------------------------------- #
_MD_LEAD_RE = re.compile(r"^\s*(?:[-*\u2022]\s+|#+\s+|\d+[.)]\s+)")


def normalise_text(text: str) -> str:
    """Strip markdown the layout does not want and collapse whitespace.

    Done before validation so a model that wraps a word in ``**`` is not penalised for
    something the renderer would ignore anyway.
    """
    t = (text or "").replace("**", "").replace("`", "")
    t = " ".join(_MD_LEAD_RE.sub("", line) for line in t.splitlines())
    return " ".join(t.split())


def normalise_narrative(n: ReviewerNarrative) -> ReviewerNarrative:
    def fix(c: Claim) -> Claim:
        seen: list[str] = []
        for x in c.cites:
            x = x.strip()
            if x and x not in seen:
                seen.append(x)
        return Claim(text=normalise_text(c.text), cites=seen)
    return ReviewerNarrative(
        headline=fix(n.headline), risk_drivers=[fix(c) for c in n.risk_drivers],
        confidence_reading=fix(n.confidence_reading),
        routing_explanation=fix(n.routing_explanation),
        suggested_checks=[fix(c) for c in n.suggested_checks],
        similar_cases=[fix(c) for c in n.similar_cases])


# ---- helpers --------------------------------------------------------------- #
def _claims(n: ReviewerNarrative) -> Iterator[tuple[str, str, Claim]]:
    yield "headline", "headline", n.headline
    for i, c in enumerate(n.risk_drivers):
        yield f"risk_drivers[{i}]", "risk_drivers", c
    yield "confidence_reading", "confidence_reading", n.confidence_reading
    yield "routing_explanation", "routing_explanation", n.routing_explanation
    for i, c in enumerate(n.suggested_checks):
        yield f"suggested_checks[{i}]", "suggested_checks", c
    for i, c in enumerate(n.similar_cases):
        yield f"similar_cases[{i}]", "similar_cases", c


def _band_ok(what: str, claimed: str, actual: str, score: int) -> bool:
    if claimed == _BAND_NAMES[actual]:
        return True
    if actual == "VERY_HIGH" and claimed == "high":
        return True      # imprecise, not contradictory
    # In this system "low confidence" also means "below the auto-clear minimum", which
    # covers more scores than the LOW band does (see risk_synthesis.MIN_CONFIDENCE_FOR_AUTO).
    return what == "confidence" and claimed == "low" and score < node4.MIN_CONFIDENCE_FOR_AUTO


def _strip_refs(text: str) -> str:
    """Text without ids, codes and page/row phrases (they are checked on their own)."""
    t = _ID_RE.sub(" ", text)
    t = _CODE_RE.sub(" ", t)
    t = _PAGE_RE.sub(" ", t)
    return _ROW_RE.sub(" ", t)


_QUOTE_RE = re.compile("\u201c([^\u201d]{1,400})\u201d")


def _visible(c: Claim, ctx: ExplainContext) -> tuple[str, list[str]]:
    """The claim text without its *verified* quotations, and the quotations that are not.

    A quotation (curly quotes) is verified when it appears word for word in a source the
    claim cites, such as a past reviewer's note. Verified quotes are somebody else's words,
    so they are not checked as this narrative's own statements about the case.
    """
    blob = " ".join(" ".join(ctx.sources[x].text.lower().split())
                    for x in c.cites if x in ctx.sources)
    bad: list[str] = []

    def drop(m: re.Match) -> str:
        quoted = " ".join(m.group(1).lower().split())
        if quoted and quoted in blob:
            return " "
        bad.append(m.group(1))
        return m.group(0)
    return _QUOTE_RE.sub(drop, c.text), bad


def _verbatim(raw: str, blob: str) -> bool:
    """``raw`` appears in ``blob`` as a whole number, not inside a longer run of digits."""
    return re.search(rf"(?<!\d){re.escape(raw)}(?!\d)", blob) is not None


def _window_hit(text_l: str, phrase: str, n: int = 6) -> bool:
    words = phrase.split()
    if len(words) < n:
        return bool(phrase) and phrase in text_l
    return any(" ".join(words[i:i + n]) in text_l for i in range(len(words) - n + 1))


# ---- the checks ------------------------------------------------------------ #
def _check_structure(n: ReviewerNarrative, ctx: ExplainContext) -> list[Violation]:
    out: list[Violation] = []
    if len(n.risk_drivers) > MAX_DRIVERS:
        out.append(Violation(STRUCTURE, "risk_drivers", f"more than {MAX_DRIVERS} items"))
    if len(n.suggested_checks) > MAX_CHECKS:
        out.append(Violation(STRUCTURE, "suggested_checks", f"more than {MAX_CHECKS} items"))
    if len(n.similar_cases) > MAX_SIMILAR:
        out.append(Violation(STRUCTURE, "similar_cases", f"more than {MAX_SIMILAR} items"))
    if n.similar_cases and not ctx.cases:
        out.append(Violation(STRUCTURE, "similar_cases", "no past case was provided"))
    for where, role, c in _claims(n):
        if not c.text.strip():
            out.append(Violation(STRUCTURE, where, "empty text"))
        limit = MAX_LONG_CHARS if role in ("confidence_reading", "routing_explanation") else MAX_CLAIM_CHARS
        if len(c.text) > limit:
            out.append(Violation(STRUCTURE, where, f"{len(c.text)} characters (limit {limit})"))
        if not c.cites:
            out.append(Violation(CITATION_UNKNOWN, where, "no citation"))
        if len(c.cites) > MAX_CITES:
            out.append(Violation(STRUCTURE, where, f"more than {MAX_CITES} citations"))
        if _BAD_TEXT_RE.search(c.text):
            out.append(Violation(FORMAT, where, "markup, link or code fence in the text"))
        m = _RAW_LABEL_RE.search(c.text)
        if m:
            out.append(Violation(FORMAT, where, f"internal label {m.group(0)} in the text; "
                                                 "write it as words"))
    return out


def _check_citations(n: ReviewerNarrative, ctx: ExplainContext) -> list[Violation]:
    out: list[Violation] = []
    need = {
        "risk_drivers": ({KIND_FINDING, KIND_SCORE}, "a finding or a score component"),
        "confidence_reading": ({KIND_ADJUSTMENT}, "a confidence adjustment"),
        "suggested_checks": ({KIND_FINDING, KIND_KNOWLEDGE}, "a finding or knowledge entry"),
        "similar_cases": ({KIND_CASE}, "a past case"),
    }
    for where, role, c in _claims(n):
        unknown = [x for x in c.cites if x not in ctx.sources]
        for x in unknown:
            out.append(Violation(CITATION_UNKNOWN, where, f"{x!r} is not a source of this case"))
        known = [ctx.sources[x] for x in c.cites if x in ctx.sources]
        if role in need:
            kinds, label = need[role]
            if known and not any(s.kind in kinds for s in known):
                out.append(Violation(CITATION_ROLE, where, f"must cite {label}"))
        if role == "routing_explanation" and "ROUTE" in ctx.sources and "ROUTE" not in c.cites:
            out.append(Violation(CITATION_ROLE, where, "must cite ROUTE"))
    return out


def _check_faithfulness(n: ReviewerNarrative, ctx: ExplainContext) -> list[Violation]:
    out: list[Violation] = []
    for where, role, c in _claims(n):
        cited: list[Source] = [ctx.sources[x] for x in c.cites if x in ctx.sources]
        if not cited:
            continue
        blob = " ".join(s.text for s in cited)
        pages = set().union(*(s.pages for s in cited))
        rows = set().union(*(s.rows for s in cited))
        text, unverified = _visible(c, ctx)
        for q in unverified:
            out.append(Violation(FAITHFUL_QUOTE, where, f"quotation not found in the cited sources: {q[:60]!r}"))

        for code in sorted(set(_CODE_RE.findall(text))):
            if code not in blob:
                out.append(Violation(FAITHFUL_CODE, where, f"{code} is not in the cited sources"))
        for m in _PAGE_RE.finditer(_ID_RE.sub(" ", text)):
            nums = {int(m.group(1))}
            if m.group(2):
                nums |= set(range(int(m.group(1)), min(int(m.group(2)), int(m.group(1)) + 50) + 1))
            for p in sorted(nums - pages):
                out.append(Violation(FAITHFUL_PAGE, where, f"page {p} is not in the cited sources"))
        for m in _ROW_RE.finditer(text):
            if int(m.group(1)) not in rows:
                out.append(Violation(FAITHFUL_ROW, where,
                                     f"row {m.group(1)} is not in the cited sources"))

        supported = ctx.allowed_numbers | numbers_in(blob)
        for m in NUM_RE.finditer(_strip_refs(text)):
            raw, tok = m.group(0), norm_number(m.group(0))
            if len(tok.replace(".", "")) < 2 or tok in supported or _verbatim(raw, blob):
                continue      # a figure copied verbatim from a cited source is supported
            out.append(Violation(FAITHFUL_NUMBER, where, f"{tok} is not in the cited sources"))

        if role == "similar_cases":
            decisions = {s.decision for s in cited if s.kind == KIND_CASE}
            for outcome, rx in _OUTCOME_WORDS.items():
                other = "APPROVED" if outcome == "REJECTED" else "REJECTED"
                if rx.search(text) and not _OUTCOME_WORDS[other].search(text) \
                        and outcome not in decisions:
                    out.append(Violation(FAITHFUL_OUTCOME, where,
                                         f"says {outcome.lower()} but the cited case was "
                                         f"{', '.join(sorted(d for d in decisions if d)).lower()}"))

    top = max((s for s in ctx.finding_sources()), key=lambda s: (s.weight, -(s.finding_index or 0)),
              default=None)
    if top is not None and top.weight > 0:
        cited_codes: set[str] = set()
        for c in [n.headline, *n.risk_drivers]:
            for x in c.cites:
                if x in ctx.sources:
                    cited_codes |= ctx.sources[x].codes
        if not (top.codes & cited_codes):
            out.append(Violation(OMISSION, "risk_drivers",
                                 f"leaves out the heaviest finding ({next(iter(top.codes))})"))
    return out


def _check_consistency(n: ReviewerNarrative, ctx: ExplainContext) -> list[Violation]:
    out: list[Violation] = []
    r = ctx.report
    actual = {"risk": (r.risk_score, r.risk_band.value),
              "confidence": (r.confidence_score, r.confidence_band.value)}
    for where, role, c in _claims(n):
        text, _ = _visible(c, ctx)
        for what, rx in _SCORE_RE.items():
            for m in rx.finditer(text):
                if _STARTING_RE.search(text[max(0, m.start() - 30):m.end()]):
                    continue      # "started confidence at 85" names the starting point, not the result
                if int(m.group(1)) != actual[what][0]:
                    out.append(Violation(CONSISTENCY_SCORE, where,
                                         f"states {what} {m.group(1)}, the report says {actual[what][0]}"))
        for what, rxs in _BAND_RE.items():
            for rx in rxs:
                for m in rx.finditer(text):
                    if not _band_ok(what, m.group(1).lower(), actual[what][1], actual[what][0]):
                        out.append(Violation(CONSISTENCY_BAND, where,
                                             f"states {what} band {m.group(1).lower()}, "
                                             f"the report says {actual[what][1].lower()}"))
        bad = (_HUMAN_REVIEW_CONTRADICTIONS if r.routing == Routing.HUMAN_REVIEW
               else _AUTO_CLEAR_CONTRADICTIONS)
        for rx in bad:
            if rx.search(text):
                out.append(Violation(CONSISTENCY_ROUTING, where,
                                     f"contradicts the routing ({r.routing.value})"))
        for rx in _VERDICT_RES:
            for m in rx.finditer(text):
                if rx is _STATE_VERDICT_RE and _is_conditional(text, m.start()):
                    continue
                if _is_report_of_a_past_decision(role, c, ctx, text, m):
                    continue
                out.append(Violation(VERDICT, where, f"gives a verdict: {m.group(0)!r}"))
                break
        out += _check_direction(where, text, c, ctx)
    return out


def _polarity(clause: str) -> Optional[bool]:
    """True when a clause says something went up, False when down, None when it does not say
    or says both (then it cannot be checked)."""
    t = _ABOUT_FINDINGS_RE.sub(" ", clause)
    up = bool(_UP_RE.search(t))
    down = bool(_DOWN_RE.search(t))
    for m in _SIGNED_RE.finditer(t):
        if m.group(1) == "+":
            up = True
        else:
            down = True
    return None if up == down else up


def _adjustments_in(clause: str, cited: list[Source]) -> list[Source]:
    """The cited adjustments a clause is about: by id, else by topic."""
    named = {m.group(1) for m in _ID_MENTION_RE.finditer(clause)}
    by_id = [s for s in cited if s.id in named]
    if by_id:
        return by_id
    return [s for s in cited if any(_TOPICS.get(code) and _TOPICS[code].search(clause)
                                    for code in s.codes)]


def _check_direction(where: str, text: str, c: Claim, ctx: ExplainContext) -> list[Violation]:
    """Raised / lowered must match the sign of the adjustment the sentence is about."""
    cited = [ctx.sources[x] for x in c.cites
             if x in ctx.sources and ctx.sources[x].kind == KIND_ADJUSTMENT
             and ctx.sources[x].points is not None and "CONF_BASE" not in ctx.sources[x].codes]
    if not cited:
        return []
    out: list[Violation] = []
    signs = {s.points > 0 for s in cited}
    for clause in _CLAUSE_SPLIT_RE.split(text):
        up = _polarity(clause)
        if up is None:
            continue
        targets = _adjustments_in(clause, cited)
        if not targets and len(signs) == 1:
            targets = cited               # every cited adjustment points the same way
        for s in targets:
            if (s.points > 0) != up:
                out.append(Violation(
                    CONSISTENCY_DIRECTION, where,
                    f"says confidence was {'raised' if up else 'lowered'}, but {s.id} "
                    f"({s.label}) moves it {'up' if s.points > 0 else 'down'}"))
                break
    return out


def _check_injection(n: ReviewerNarrative, ctx: ExplainContext) -> list[Violation]:
    out: list[Violation] = []
    for where, _role, c in _claims(n):
        for _cat, label, rx in _OUTPUT_INSTRUCTION_PATTERNS:
            if rx.search(c.text):
                out.append(Violation(INJECTION_PATTERN, where, f"contains an {label}"))
        low = " ".join(c.text.lower().split())
        for phrase in ctx.forbidden_phrases:
            if _window_hit(low, phrase):
                out.append(Violation(INJECTION_ECHO, where,
                                     "repeats text taken from an embedded document instruction"))
    return out


def validate(narrative: ReviewerNarrative, ctx: ExplainContext) -> list[Violation]:
    """All violations of ``narrative`` against ``ctx`` (empty list means it may be shown)."""
    return (_check_structure(narrative, ctx) + _check_citations(narrative, ctx)
            + _check_faithfulness(narrative, ctx) + _check_consistency(narrative, ctx)
            + _check_injection(narrative, ctx))


def summarise(violations: list[Violation], limit: int = 3) -> str:
    """Short text for logs and the stored ``fallback_reason``."""
    head = "; ".join(str(v) for v in violations[:limit])
    more = len(violations) - limit
    return head + (f"; and {more} more" if more > 0 else "")
