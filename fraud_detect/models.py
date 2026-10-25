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

from pydantic import BaseModel, Field


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


class IngestionResult(BaseModel):
    source_path: str
    page_count: int
    raw_text: str
    page_texts: list[PageText] = Field(default_factory=list)
    metadata: DocumentMetadata
    fonts: list[FontUsage] = Field(default_factory=list)

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

    # Cross-case memory: fingerprints of this case's transactions, persisted so
    # that future screenings can detect recurrence of the same entity/amount.
    transaction_fingerprints: list[TxnFingerprint] = Field(default_factory=list)

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


class Case(BaseModel):
    """Everything persisted about one submitted statement."""
    case_id: str
    report: RiskReport
    review: ReviewRecord
    audit_log: list[dict] = Field(default_factory=list)
