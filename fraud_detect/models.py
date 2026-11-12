"""Shared data models — the structured output contract (strict JSON schema).

Every agent emits typed, validated data. Missing mandatory fields are represented
as explicit ``None`` values and surfaced as data-quality findings rather than being
guessed / hallucinated.
"""
from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class RiskBand(str, Enum):
    LOW = "LOW"            # 1-25
    MEDIUM = "MEDIUM"      # 26-50
    HIGH = "HIGH"          # 51-75
    VERY_HIGH = "VERY_HIGH"  # 76-100


class ConfidenceBand(str, Enum):
    LOW = "LOW"            # 1-25
    MEDIUM = "MEDIUM"      # 26-50
    HIGH = "HIGH"          # 51-75
    VERY_HIGH = "VERY_HIGH"  # 76-100


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Routing(str, Enum):
    AUTO_CLEAR = "AUTO_CLEAR"                      # low risk -> auto approval recommendation
    HUMAN_REVIEW = "HUMAN_REVIEW"                  # medium/high risk or low confidence


class ReviewDecision(str, Enum):
    PENDING = "PENDING"             # screened, awaiting a human reviewer
    NEEDS_INFO = "NEEDS_INFO"       # reviewer has asked the customer for more info
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class CreditDecision(str, Enum):
    """The banker / loan officer's FINAL lending decision.

    Distinct from ReviewDecision (which is the fraud-screening outcome). A banker
    only reaches this stage once the document's authenticity is settled: the
    analyst cleared the fraud concern (APPROVED) or the statement auto-cleared.
    The screening decides whether the document can be *trusted*; the banker
    decides whether to *lend*.
    """
    PENDING = "PENDING"             # fraud-cleared, awaiting the loan officer
    APPROVED = "APPROVED"           # loan approved
    DECLINED = "DECLINED"           # loan declined on credit grounds
    REFERRED = "REFERRED"           # escalated to a senior officer / credit committee


# --------------------------------------------------------------------------- #
# Findings & evidence
# --------------------------------------------------------------------------- #
class Finding(BaseModel):
    """A single explainable signal produced by an agent.

    ``code`` is a stable reason code (audit-friendly). ``evidence`` is a
    human-readable explanation. ``weight`` is the point contribution toward the
    aggregate risk score (0-100 scale contributions, clamped later).
    """
    agent: str
    code: str
    title: str
    severity: Severity
    weight: float = 0.0
    evidence: str
    page: Optional[int] = None
    field: Optional[str] = None


# --------------------------------------------------------------------------- #
# Node 1 — ingestion output
# --------------------------------------------------------------------------- #
class Transaction(BaseModel):
    index: int
    date: Optional[date_type] = None
    date_raw: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None          # signed: credits +, debits -
    balance: Optional[float] = None
    page: Optional[int] = None
    # Optional Node 1 (multi-layout) additions; old stored rows omit them.
    value_date_raw: Optional[str] = None      # second date column text, when printed
    direction_method: Optional[str] = None    # "column" | "printed_sign" | "balance_change"


class SectionSummary(BaseModel):
    """One monthly statement section as read by Node 1 (all fields optional)."""
    period_start: Optional[date_type] = None
    period_end: Optional[date_type] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    opening: Optional[float] = None
    closing: Optional[float] = None
    total_deposits: Optional[float] = None
    total_withdrawals: Optional[float] = None


class DocumentMetadata(BaseModel):
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    mod_date: Optional[str] = None
    title: Optional[str] = None


class FontUsage(BaseModel):
    """Per-page font fingerprint, used by the visual forensics agent."""
    page: int
    fonts: dict[str, int] = Field(default_factory=dict)  # font name -> glyph count


class PageText(BaseModel):
    """Extracted text for a single page (used by the security layer so it can
    report the exact page an embedded instruction was found on)."""
    page: int
    text: str


class StatementSegment(BaseModel):
    """One contiguous statement section inside a (possibly combined) PDF.

    A combined PDF can staple several monthly statements together; each printed
    "Statement Period:" header starts a new segment. ``page_start`` / ``page_end``
    delimit the pages that segment spans. All fields are Optional-with-default so
    partially-parsed or hand-built segments still validate.
    """
    period_start: Optional[date_type] = None
    period_end: Optional[date_type] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    # Basename of the file this segment was ingested from. Used only for
    # human-readable evidence on a MULTI-FILE submission (so duplicate-month
    # evidence can name the colliding files); never used for month identity.
    # Optional-with-default so single-file/combined-PDF and hand-built segments
    # are unchanged.
    source: Optional[str] = None


class IngestionResult(BaseModel):
    source_path: str
    page_count: int
    raw_text: str
    page_texts: list[PageText] = Field(default_factory=list)
    metadata: DocumentMetadata
    fonts: list[FontUsage] = Field(default_factory=list)

    # One entry per "Statement Period:" header found in the PDF. Empty for older
    # stored reports / hand-built results; a single-period PDF yields exactly one.
    segments: list[StatementSegment] = Field(default_factory=list)

    # Parsed statement header
    account_holder: Optional[str] = None
    uen: Optional[str] = None
    account_number: Optional[str] = None
    statement_period_start: Optional[date_type] = None
    statement_period_end: Optional[date_type] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    declared_currency: Optional[str] = None

    transactions: list[Transaction] = Field(default_factory=list)

    # Multi-layout parser additions (optional; absent on older results).
    layout_profile: Optional[str] = None
    parse_coverage: Optional[float] = None
    sections: list[SectionSummary] = Field(default_factory=list)

    # Data-quality flags collected during parsing
    data_quality_findings: list[Finding] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Node 2 / Node 3 — agent outputs
# --------------------------------------------------------------------------- #
class AgentReport(BaseModel):
    agent: str
    findings: list[Finding] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Cross-case memory
# --------------------------------------------------------------------------- #
class TxnFingerprint(BaseModel):
    """A compact, comparable signature of one transaction/entity.

    Two transactions are considered "similar" when their fingerprints match on
    counterparty + amount_band + direction (see memory node). Stored on each
    RiskReport so future cases can be cross-referenced against this one.
    """
    counterparty: str = ""        # normalized (lower-case, suffixes stripped)
    amount_band: str = "na"       # bucketed amount, e.g. "145,000-150,000"
    direction: str = "na"         # credit | debit | na
    pattern_tag: str = ""         # e.g. "RAPID_PASS_THROUGH" / "STRUCTURING"

    def key(self) -> str:
        """Match key — intentionally excludes pattern_tag so a plain repeat of
        the same counterparty+amount matches a prior flagged one too."""
        return f"{self.counterparty}|{self.amount_band}|{self.direction}"


# --------------------------------------------------------------------------- #
# Node 4 companion — deterministic explanation of the two scores
# --------------------------------------------------------------------------- #
class RiskLine(BaseModel):
    """One line of the risk-score breakdown. The ``points`` of all lines add up to the
    final risk score exactly (see ``fraud_detect.reviewer.breakdown``)."""
    id: str                          # "F3" (finding), "BONUS", "ROUND", "LIMIT" or "FLOOR"
    kind: str                        # FINDING | CORROBORATION_BONUS | ROUNDING | RANGE_LIMIT | MEDIUM_FLOOR
    points: float
    label: str                       # short name, e.g. the finding title
    reading: str                     # plain-language reading for a reviewer
    finding_code: Optional[str] = None
    severity: Optional[Severity] = None
    page: Optional[int] = None
    field: Optional[str] = None


class ConfidenceLine(BaseModel):
    """One line of the confidence-score breakdown (``A1`` is the starting point)."""
    id: str                          # "A1", "A2", ...
    code: str                        # CONF_BASE | CONF_DATA_QUALITY | ... | CONF_LIMIT
    points: int
    label: str                       # Node 4's own wording of the adjustment
    reading: str                     # plain-language reading for a reviewer


class RoutingCause(BaseModel):
    """What actually caused the routing decision."""
    primary: str                     # MATERIAL_FINDING | LOW_CONFIDENCE | RISK_THRESHOLD | AUTO_CLEAR
    statement: str
    also_applies: list[str] = Field(default_factory=list)   # other independent human-review triggers
    finding_ids: list[str] = Field(default_factory=list)    # F-ids of MEDIUM+ findings, when relevant


class ScoreExplanation(BaseModel):
    """Deterministic, free-to-compute explanation of risk, confidence and routing.

    Built by ``fraud_detect.reviewer.breakdown`` from Node 4's own output and checked
    against it: ``reconciled`` is False when the lines do not add up to the scores or
    the derived routing differs, which means this explanation must not be trusted.
    """
    risk_lines: list[RiskLine] = Field(default_factory=list)
    risk_total: float = 0.0
    confidence_lines: list[ConfidenceLine] = Field(default_factory=list)
    confidence_total: float = 0.0
    routing: RoutingCause
    # What the confidence score means for the analyst, derived from the score and the routing
    # (policy, not prose a model writes). Empty on explanations stored before it existed.
    confidence_meaning: str = ""
    reconciled: bool = True
    notes: list[str] = Field(default_factory=list)          # why it did not reconcile


# --------------------------------------------------------------------------- #
# Optional LLM reviewer — advisory narrative (never changes a score or a routing)
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    """One statement of the narrative plus the evidence it rests on.

    ``cites`` holds citation ids that must exist for this case: ``F<n>`` findings,
    ``A<n>`` confidence adjustments, ``BONUS`` / ``FLOOR`` score components, ``ROUTE``
    (the routing cause), ``K<n>`` curated knowledge and ``P<n>`` past decided cases.
    """
    text: str
    cites: list[str] = Field(default_factory=list)


class ReviewerNarrative(BaseModel):
    """Fixed layout of the reviewer notes. The model fills the text; code does the layout."""
    headline: Claim
    risk_drivers: list[Claim] = Field(default_factory=list)
    confidence_reading: Claim
    routing_explanation: Claim
    suggested_checks: list[Claim] = Field(default_factory=list)
    similar_cases: list[Claim] = Field(default_factory=list)


class ReviewerNotes(BaseModel):
    """The narrative as stored with the case, with what an auditor needs to replay it."""
    source: str                                  # "LLM" | "BASELINE"
    narrative: ReviewerNarrative
    model: Optional[str] = None
    prompt_version: str = ""
    knowledge_version: Optional[str] = None
    knowledge_ids: list[str] = Field(default_factory=list)   # curated documents retrieved
    case_ids: list[str] = Field(default_factory=list)        # past decided cases retrieved
    citations: dict[str, str] = Field(default_factory=dict)  # citation id -> readable label
    fallback_reason: Optional[str] = None        # why the template was used instead of the LLM
    # Citation fixes applied to the model's answer before it was validated (see reviewer.repair).
    repairs: list[str] = Field(default_factory=list)
    seconds: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# Node 4 — synthesis
# --------------------------------------------------------------------------- #
class RiskReport(BaseModel):
    case_id: str
    source_path: str
    account_holder: Optional[str] = None
    uen: Optional[str] = None
    statement_period: Optional[str] = None

    risk_score: int                 # 1-100
    risk_band: RiskBand
    confidence_score: int           # 1-100 (inverted-style: certainty of the risk call)
    confidence_band: ConfidenceBand

    routing: Routing
    routing_reason: str

    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""

    # Parsed transaction rows, persisted so the analyst "review flagged
    # transactions" screen can show each row alongside the findings that cite it.
    # Optional (defaults to empty) so older stored reports still validate.
    transactions: list[Transaction] = Field(default_factory=list)
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    declared_currency: Optional[str] = None

    # Per-segment coverage summary (period + page range for each statement
    # section found in the source PDF). A single-period statement yields exactly
    # one segment; a combined multi-month PDF yields several. Optional (defaults
    # to empty) so older stored reports still validate and single-segment cases
    # render unchanged.
    statement_segments: list[StatementSegment] = Field(default_factory=list)

    # Cross-case memory: fingerprints of this case's transactions, persisted so
    # that future screenings can detect recurrence of the same entity/amount.
    transaction_fingerprints: list[TxnFingerprint] = Field(default_factory=list)

    # Plain-language breakdown of the two scores and the routing (deterministic, built
    # right after Node 4). Optional so reports stored before it existed still validate.
    explanation: Optional[ScoreExplanation] = None
    # Advisory narrative from the optional LLM reviewer (off by default). It explains the
    # scores above; it never sets or changes them.
    reviewer_notes: Optional[ReviewerNotes] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# Human-in-the-loop
# --------------------------------------------------------------------------- #
class ReviewRecord(BaseModel):
    case_id: str
    decision: ReviewDecision = ReviewDecision.PENDING
    reviewer: Optional[str] = None
    reason: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    # Feedback captured to improve future model training (per the plan).
    feedback_for_training: Optional[str] = None
    # When the reviewer requests more information (NEEDS_INFO), the customer-safe
    # message they want relayed to the applicant is stored here so the customer
    # portal can show it. Never contains internal scores/evidence.
    info_request: Optional[str] = None
    info_requested_at: Optional[datetime] = None


class CreditRecord(BaseModel):
    """The banker / loan officer's final lending decision on a fraud-cleared case."""
    case_id: str
    decision: CreditDecision = CreditDecision.PENDING
    officer: Optional[str] = None
    rationale: Optional[str] = None
    # Optional credit inputs the officer may capture for the file.
    approved_amount: Optional[float] = None     # amount actually approved (may differ from requested)
    conditions: Optional[str] = None            # covenants / conditions attached to an approval
    decided_at: Optional[datetime] = None


class DecidedCase(BaseModel):
    """A past case a human reviewer finally decided, with the reviewer's own reasoning.

    Read-only view built by ``CaseStore.recall_decided_cases`` for retrieval. Only
    APPROVED and REJECTED cases appear here; pending and needs-info cases never do.
    """
    case_id: str
    decision: str                                   # "APPROVED" | "REJECTED"
    account_holder: Optional[str] = None
    uen: Optional[str] = None
    risk_score: int = 0
    finding_codes: list[str] = Field(default_factory=list)
    fingerprint_keys: list[str] = Field(default_factory=list)
    reason: Optional[str] = None                    # as typed by the reviewer (HTML-escaped on entry)
    feedback: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class Case(BaseModel):
    """Everything persisted about one submitted statement."""
    case_id: str
    report: RiskReport
    review: ReviewRecord
    # The banker's lending decision. Optional/defaulted so older stored cases
    # (persisted before this field existed) still validate.
    credit: CreditRecord = Field(default_factory=lambda: CreditRecord(case_id=""))
    audit_log: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_credit_case_id(self) -> "Case":
        # Backfill the credit record's case_id (for cases created before the
        # credit stage existed, or when the default factory was used).
        if self.credit and not self.credit.case_id:
            self.credit.case_id = self.case_id
        return self
