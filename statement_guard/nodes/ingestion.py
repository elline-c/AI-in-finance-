"""Node 1: Ingestion & Preprocessing Agent.

In production this would: convert the uploaded PDF to high-res images, run OCR,
parse the layout (tables/headers/footers), and extract file metadata. In this
prototype the OCR/parse result is supplied as JSON matching the input schema;
this node's job is to *normalize and validate it against the strict contract* —
enforcing types, filling explicit nulls for missing data, and flagging data
quality issues rather than guessing (the "handling missing data" rule).
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts import (
    DataQuality,
    DocumentMetadata,
    Finding,
    Node,
    NormalizedStatement,
    Severity,
    SupportingDoc,
    Transaction,
)

MANDATORY_FIELDS = [
    "account_holder",
    "account_number",
    "currency",
    "period_start",
    "period_end",
    "opening_balance",
    "closing_balance",
]


def _num(value: Any) -> Optional[float]:
    """Coerce a value to float, tolerating currency symbols / thousands sep."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = (
            value.replace(",", "").replace("$", "").replace("SGD", "").strip()
        )
        # accept trailing/leading spaces and parenthesised negatives
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        try:
            n = float(cleaned)
            return -n if neg else n
        except ValueError:
            return None
    return None


class IngestionAgent:
    node = Node.INGESTION

    def run(self, raw: dict[str, Any]) -> tuple[NormalizedStatement, list[Finding]]:
        findings: list[Finding] = []

        ocr_conf = float(raw.get("ocr_confidence", 1.0))
        unreadable = int(raw.get("unreadable_pages", 0))
        total_pages = int(raw.get("total_pages", 1))

        transactions: list[Transaction] = []
        format_issues: list[str] = []
        for i, t in enumerate(raw.get("transactions", [])):
            date = t.get("date")
            if date and not self._looks_iso(date):
                format_issues.append(f"transaction[{i}].date not ISO: {date!r}")
            transactions.append(
                Transaction(
                    date=date,
                    description=t.get("description"),
                    counterparty=t.get("counterparty"),
                    amount=_num(t.get("amount")),
                    balance=_num(t.get("balance")),
                )
            )

        meta_raw = raw.get("metadata", {}) or {}
        metadata = DocumentMetadata(
            author=meta_raw.get("author"),
            creator_tool=meta_raw.get("creator_tool"),
            producer_tool=meta_raw.get("producer_tool"),
            creation_date=meta_raw.get("creation_date"),
            modification_date=meta_raw.get("modification_date"),
        )

        supporting = [
            SupportingDoc(
                doc_type=s.get("doc_type"),
                references_txn_id=s.get("references_txn_id"),
                amount=_num(s.get("amount")),
                counterparty=s.get("counterparty"),
            )
            for s in raw.get("supporting_docs", [])
        ]

        stmt = NormalizedStatement(
            account_holder=raw.get("account_holder"),
            account_number=raw.get("account_number"),
            currency=raw.get("currency"),
            period_start=raw.get("period_start"),
            period_end=raw.get("period_end"),
            opening_balance=_num(raw.get("opening_balance")),
            closing_balance=_num(raw.get("closing_balance")),
            transactions=transactions,
            metadata=metadata,
            supporting_docs=supporting,
        )

        missing = [f for f in MANDATORY_FIELDS if getattr(stmt, f) in (None, "")]
        stmt.data_quality = DataQuality(
            ocr_confidence=ocr_conf,
            missing_mandatory_fields=missing,
            unreadable_pages=unreadable,
            total_pages=total_pages,
            format_issues=format_issues,
        )

        # --- findings ---------------------------------------------------- #
        if missing:
            findings.append(
                Finding(
                    self.node,
                    "MISSING_MANDATORY_FIELDS",
                    "Mandatory fields missing after extraction",
                    Severity.MEDIUM,
                    "OCR/parse could not populate required fields; returned null "
                    "rather than guessing. Human review needed to supply them: "
                    + ", ".join(missing),
                    {"missing_fields": missing},
                )
            )
        if ocr_conf < 0.80:
            findings.append(
                Finding(
                    self.node,
                    "LOW_OCR_CONFIDENCE",
                    "Low OCR confidence",
                    Severity.MEDIUM,
                    f"Average OCR confidence {ocr_conf:.0%} is below the 80% "
                    "threshold; extracted values may be unreliable.",
                    {"ocr_confidence": ocr_conf},
                )
            )
        if unreadable > 0:
            findings.append(
                Finding(
                    self.node,
                    "UNREADABLE_PAGES",
                    "Unreadable page(s) detected",
                    Severity.MEDIUM,
                    f"{unreadable} of {total_pages} page(s) could not be read "
                    "(poor scan quality).",
                    {"unreadable_pages": unreadable, "total_pages": total_pages},
                )
            )
        if format_issues:
            findings.append(
                Finding(
                    self.node,
                    "FORMAT_INCONSISTENCY",
                    "Inconsistent date/number formatting",
                    Severity.LOW,
                    "Some fields did not match the expected format and were "
                    "normalized where possible.",
                    {"issues": format_issues},
                )
            )

        return stmt, findings

    @staticmethod
    def _looks_iso(date: str) -> bool:
        # YYYY-MM-DD
        parts = date.split("-")
        return (
            len(parts) == 3
            and len(parts[0]) == 4
            and parts[0].isdigit()
            and parts[1].isdigit()
            and parts[2].isdigit()
        )
