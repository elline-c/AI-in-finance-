"""Node 2: Visual Forensics Agent.

Detects *document tampering* signals. In production, image manipulation
inspection would run pixel/layer/compression analysis on the rendered images. In
this prototype we work from the extracted metadata and declared layout signals,
which covers the strongest, most explainable tampering indicators:

  * Metadata inconsistency (editing-software signatures, modified-after-creation,
    metadata dates vs. printed statement period)
  * Content/format consistency (account number / currency formatting drift)
  * Altered-text hints surfaced by OCR (baseline/kerning anomaly flags)
  * Supporting-document relevance (is the invoice/receipt for the right txn?)
"""

from __future__ import annotations

from ..contracts import Finding, Node, NormalizedStatement, Severity

# Software that has no business producing a genuine bank statement.
EDITING_TOOLS = (
    "photoshop",
    "illustrator",
    "gimp",
    "inkscape",
    "coreldraw",
    "affinity",
    "canva",
    "paint.net",
)

RELEVANT_SUPPORT_TYPES = {"invoice", "receipt", "delivery_order", "contract", "payslip"}


class VisualForensicsAgent:
    node = Node.VISUAL_FORENSICS

    def run(self, stmt: NormalizedStatement) -> list[Finding]:
        findings: list[Finding] = []
        m = stmt.metadata

        # 1. Editing-software signature -------------------------------------
        tool_blob = " ".join(
            str(x).lower() for x in (m.creator_tool, m.producer_tool, m.author) if x
        )
        hit = next((t for t in EDITING_TOOLS if t in tool_blob), None)
        if hit:
            findings.append(
                Finding(
                    self.node,
                    "EDITING_SOFTWARE_SIGNATURE",
                    "Image-editing software signature in metadata",
                    Severity.CRITICAL,
                    f"Document metadata references '{hit}', an image/graphics "
                    "editor. Genuine bank statements are produced by banking "
                    "systems, not design tools — a strong tampering indicator.",
                    {
                        "creator_tool": m.creator_tool,
                        "producer_tool": m.producer_tool,
                    },
                )
            )

        # 2. Modified after creation ----------------------------------------
        if m.creation_date and m.modification_date:
            if m.modification_date > m.creation_date:
                findings.append(
                    Finding(
                        self.node,
                        "MODIFIED_AFTER_CREATION",
                        "File modified after it was created",
                        Severity.HIGH,
                        f"Modification date ({m.modification_date}) is later than "
                        f"creation date ({m.creation_date}). A statement exported "
                        "once should not be re-saved.",
                        {
                            "creation_date": m.creation_date,
                            "modification_date": m.modification_date,
                        },
                    )
                )

        # 3. Metadata date vs. printed statement period ---------------------
        if m.creation_date and stmt.period_end:
            # A file "created" before the period it reports on is impossible.
            if m.creation_date[:10] < stmt.period_end[:10]:
                findings.append(
                    Finding(
                        self.node,
                        "METADATA_PERIOD_MISMATCH",
                        "File created before the statement period ended",
                        Severity.HIGH,
                        f"File creation date {m.creation_date[:10]} precedes the "
                        f"statement period end {stmt.period_end[:10]}, which is "
                        "chronologically impossible for a genuine export.",
                        {
                            "creation_date": m.creation_date,
                            "period_end": stmt.period_end,
                        },
                    )
                )

        # 4. Account-number formatting drift --------------------------------
        acct = stmt.account_number or ""
        digits = [c for c in acct if c.isdigit()]
        if acct and len(digits) < 6:
            findings.append(
                Finding(
                    self.node,
                    "ACCOUNT_FORMAT_ANOMALY",
                    "Unusual account-number format",
                    Severity.LOW,
                    f"Account number {acct!r} has fewer digits than a standard "
                    "corporate account; possible OCR error or edit.",
                    {"account_number": acct},
                )
            )

        # 5. Altered-text hint from OCR layer -------------------------------
        # The OCR layer may flag baseline/kerning anomalies around figures.
        anomalies = [
            (i, t)
            for i, t in enumerate(stmt.transactions)
            if t.description and "[baseline-anomaly]" in t.description.lower()
        ]
        if anomalies:
            findings.append(
                Finding(
                    self.node,
                    "ALTERED_TEXT_ANOMALY",
                    "Font baseline/kerning anomaly near figures",
                    Severity.HIGH,
                    "OCR detected baseline/kerning irregularities where amounts "
                    "appear, consistent with digitally overwritten numbers.",
                    {"transaction_indexes": [i for i, _ in anomalies]},
                )
            )

        # 6. Supporting-document relevance ----------------------------------
        n_txn = len(stmt.transactions)
        for j, doc in enumerate(stmt.supporting_docs):
            if doc.doc_type and doc.doc_type.lower() not in RELEVANT_SUPPORT_TYPES:
                findings.append(
                    Finding(
                        self.node,
                        "IRRELEVANT_SUPPORT_DOC",
                        "Supporting document type not relevant",
                        Severity.MEDIUM,
                        f"Supporting doc #{j} is of type '{doc.doc_type}', which "
                        "is not an expected business record (invoice, receipt, "
                        "delivery order, etc.).",
                        {"doc_type": doc.doc_type},
                    )
                )
            if doc.references_txn_id is not None and not (
                0 <= doc.references_txn_id < n_txn
            ):
                findings.append(
                    Finding(
                        self.node,
                        "SUPPORT_DOC_DANGLING_REF",
                        "Supporting document references a missing transaction",
                        Severity.MEDIUM,
                        f"Supporting doc #{j} references transaction "
                        f"{doc.references_txn_id}, which does not exist in the "
                        "statement.",
                        {"references_txn_id": doc.references_txn_id},
                    )
                )
            elif doc.references_txn_id is not None and doc.amount is not None:
                txn = stmt.transactions[doc.references_txn_id]
                if txn.amount is not None and abs(abs(txn.amount) - abs(doc.amount)) > 0.01:
                    findings.append(
                        Finding(
                            self.node,
                            "SUPPORT_DOC_AMOUNT_MISMATCH",
                            "Supporting document amount mismatch",
                            Severity.MEDIUM,
                            f"Supporting doc #{j} states {doc.amount:.2f} but the "
                            f"referenced transaction is {txn.amount:.2f}.",
                            {"doc_amount": doc.amount, "txn_amount": txn.amount},
                        )
                    )

        return findings
