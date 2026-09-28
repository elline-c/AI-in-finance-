"""Shared data contracts, severities, and scoring bands.

Everything downstream depends on these definitions, so the "structured output
contract" from the ideation lives here in one place. Findings are the atomic
unit of evidence: each carries a severity, a human-readable explanation, and the
raw evidence it was derived from — this is what makes the final report auditable
(MAS FEAT).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Severity + risk contribution
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    """How serious a single finding is. Points feed the aggregate risk score."""

    INFO = "info"          # observation, no risk contribution
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # strong tampering indicator

    @property
    def points(self) -> int:
        return {
            "info": 0,
            "low": 6,
            "medium": 16,
            "high": 30,
            "critical": 48,
        }[self.value]


class Node(str, Enum):
    INGESTION = "ingestion"
    VISUAL_FORENSICS = "visual_forensics"
    FINANCIAL_LOGIC = "financial_logic"
    RISK_SYNTHESIS = "risk_synthesis"


@dataclass
class Finding:
    """A single piece of evidence produced by an agent."""

    node: Node
    code: str                      # stable machine code, e.g. "BALANCE_MISMATCH"
    title: str                     # short human label
    severity: Severity
    explanation: str               # readable "why" for the analyst / audit
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node"] = self.node.value
        d["severity"] = self.severity.value
        d["points"] = self.severity.points
        return d


# --------------------------------------------------------------------------- #
# Risk + confidence bands (exact numbers from the ideation)
# --------------------------------------------------------------------------- #
def risk_band(score: int) -> str:
    if score <= 25:
        return "Low"
    if score <= 50:
        return "Medium"
    if score <= 75:
        return "High"
    return "Very High"


def confidence_band(score: int) -> str:
    # inverted scale: high number == high confidence
    if score >= 76:
        return "High"
    if score >= 51:
        return "Medium"
    if score >= 26:
        return "Low"
    return "Very Low"


# --------------------------------------------------------------------------- #
# Data-quality report from ingestion (drives escalation on poor data)
# --------------------------------------------------------------------------- #
@dataclass
class DataQuality:
    ocr_confidence: float                    # 0..1
    missing_mandatory_fields: list[str] = field(default_factory=list)
    unreadable_pages: int = 0
    total_pages: int = 1
    format_issues: list[str] = field(default_factory=list)

    @property
    def is_poor(self) -> bool:
        return (
            self.ocr_confidence < 0.80
            or bool(self.missing_mandatory_fields)
            or self.unreadable_pages > 0
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_poor"] = self.is_poor
        return d


# --------------------------------------------------------------------------- #
# The normalized statement produced by Node 1 (the "OCR contract")
# --------------------------------------------------------------------------- #
@dataclass
class Transaction:
    date: Optional[str]            # ISO date or None if unreadable
    description: Optional[str]
    counterparty: Optional[str]
    amount: Optional[float]        # signed: negative = debit, positive = credit
    balance: Optional[float]       # running balance after the transaction


@dataclass
class DocumentMetadata:
    author: Optional[str] = None
    creator_tool: Optional[str] = None      # e.g. "Adobe Photoshop 25.0"
    producer_tool: Optional[str] = None
    creation_date: Optional[str] = None     # ISO
    modification_date: Optional[str] = None  # ISO


@dataclass
class SupportingDoc:
    doc_type: Optional[str]        # invoice | receipt | delivery_order | ...
    references_txn_id: Optional[int] = None  # index into transactions
    amount: Optional[float] = None
    counterparty: Optional[str] = None


@dataclass
class NormalizedStatement:
    """Node 1 output — the strict, typed contract every later node consumes."""

    account_holder: Optional[str]
    account_number: Optional[str]
    currency: Optional[str]
    period_start: Optional[str]              # ISO
    period_end: Optional[str]                # ISO
    opening_balance: Optional[float]
    closing_balance: Optional[float]
    transactions: list[Transaction] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    supporting_docs: list[SupportingDoc] = field(default_factory=list)
    data_quality: DataQuality = field(
        default_factory=lambda: DataQuality(ocr_confidence=1.0)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_holder": self.account_holder,
            "account_number": self.account_number,
            "currency": self.currency,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "opening_balance": self.opening_balance,
            "closing_balance": self.closing_balance,
            "transactions": [asdict(t) for t in self.transactions],
            "metadata": asdict(self.metadata),
            "supporting_docs": [asdict(s) for s in self.supporting_docs],
            "data_quality": self.data_quality.to_dict(),
        }
