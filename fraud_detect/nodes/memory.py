"""Node 5 — Cross-Case Memory Agent.

Most fraud screening treats each statement in isolation. This agent gives the
pipeline a *memory*: before a case is finalised it compares the current
statement's transactions against every statement screened before (read back from
the case store) and raises explainable findings when a similar
transaction/entity has been seen in a *prior* case.

Why this matters:
    * If a similar transaction was CLEARED before, a repeat may be an exploit
      that already slipped past the control once — the report says so explicitly
      ("a similar transaction passed the check previously in CASE-X").
    * If a similar transaction was REJECTED before, this is a known-bad repeat.
    * If the same counterparty/pattern recurs across many cases, that itself is
      a signal worth a human's attention.

Governance (MAS FEAT): this agent only *adds evidence and escalates*. It never
auto-rejects, and every finding cites the prior case id so a reviewer can trace
the link.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..models import AgentReport, Finding, IngestionResult, Severity, TxnFingerprint
from ..store import fingerprint

AGENT = "Node5:Memory"

# Counterparties/amounts appearing in at least this many prior cases are called
# out as a high-frequency recurring pattern.
_FREQUENCY_THRESHOLD = 3


def _flagged_rows(findings: Optional[list[Finding]]) -> Optional[set[int]]:
    """Row indexes flagged by the visual/financial agents in THIS statement.

    Memory is restricted to these rows so routine recurring vendors (salary,
    utilities) never enter cross-case memory — we only remember/match when a
    *suspicious* transaction recurs. Returns None when there are no row-level
    findings (e.g. a clean statement), which means this statement contributes
    no memory and matches nothing.
    """
    if not findings:
        return None
    rows: set[int] = set()
    for f in findings:
        if f.field and f.field.startswith("row "):
            try:
                rows.add(int(f.field.split()[1]))
            except (ValueError, IndexError):
                continue
    return rows or None


def build_fingerprints(ing: IngestionResult,
                       findings: Optional[list[Finding]] = None) -> list[TxnFingerprint]:
    """Fingerprint the *suspicious* transactions in the current statement.

    Only transactions flagged by the visual/financial agents are fingerprinted
    and persisted, so routine payments (payroll, utilities) never enter the
    cross-case memory — this keeps recurrence matching meaningful and avoids
    flagging ordinary recurring vendors. If there are no row-level findings
    (e.g. a clean statement), nothing is stored.
    """
    focus = _flagged_rows(findings)
    if focus is None:
        return []
    fps: list[TxnFingerprint] = []
    for t in ing.transactions:
        if t.index in focus:
            fps.append(fingerprint(t.description, t.amount))
    return fps


def analyze(ing: IngestionResult, prior_memory: list[dict],
            current_case_id: Optional[str] = None,
            current_findings: Optional[list[Finding]] = None) -> AgentReport:
    """Cross-reference the current statement against prior-case fingerprints.

    ``prior_memory`` is the list returned by ``CaseStore.recall_prior_fingerprints``
    — each item carries a prior case id, its human decision, and one fingerprint.
    ``current_findings`` are the visual/financial findings for this statement;
    when present, memory matching focuses on the rows those agents flagged.
    """
    findings: list[Finding] = []
    focus_rows = _flagged_rows(current_findings)
    if focus_rows is None:
        # No suspicious rows in this statement -> nothing to cross-reference.
        return AgentReport(agent=AGENT, findings=findings)

    from ..store import fingerprint  # local import avoids any import cycle

    # Index prior memory by match-key for O(1) lookup.
    by_key: dict[str, list[dict]] = defaultdict(list)
    for item in prior_memory:
        fp = item["fingerprint"]
        # recall_prior_fingerprints returns pydantic TxnFingerprint instances.
        key = fp.key() if isinstance(fp, TxnFingerprint) else TxnFingerprint(**fp).key()
        by_key[key].append(item)

    seen_keys: set[str] = set()
    for t in ing.transactions:
        # Only cross-reference the suspicious rows (keeps memory precise/auditable).
        if t.index not in focus_rows:
            continue
        fp = fingerprint(t.description, t.amount)
        if not fp.counterparty:          # nothing meaningful to match on
            continue
        key = fp.key()
        matches = by_key.get(key, [])
        if not matches or key in seen_keys:
            continue
        seen_keys.add(key)

        rejected = [m for m in matches if m["decision"] == "REJECTED"]
        cleared = [m for m in matches
                   if m["decision"] == "APPROVED" or m["routing"] == "AUTO_CLEAR"]
        case_list = ", ".join(sorted({m["case_id"] for m in matches}))

        if rejected:
            prior = ", ".join(sorted({m["case_id"] for m in rejected}))
            amt = f"{t.amount:,.2f}" if t.amount is not None else "n/a"
            findings.append(Finding(
                agent=AGENT, code="MEM_RECURRING_KNOWN_BAD",
                title="Transaction matches one rejected in a prior case",
                severity=Severity.HIGH, weight=24, page=t.page,
                field=f"row {t.index}",
                evidence=(f"Transaction #{t.index} ('{t.description}', {amt}) is a "
                          f"near-identical match to a transaction a human REJECTED in "
                          f"prior case(s) {prior}. Treat as a known-bad repeat."),
            ))
        elif cleared:
            prior = ", ".join(sorted({m["case_id"] for m in cleared}))
            amt = f"{t.amount:,.2f}" if t.amount is not None else "n/a"
            findings.append(Finding(
                agent=AGENT, code="MEM_SIMILAR_PASSED_BEFORE",
                title="Similar transaction passed the check in a prior case",
                severity=Severity.MEDIUM, weight=14, page=t.page,
                field=f"row {t.index}",
                evidence=(f"Transaction #{t.index} ('{t.description}', {amt}) closely "
                          f"matches a transaction that previously PASSED screening in "
                          f"case(s) {prior}. A similar transaction was able to pass the "
                          f"check before — verify this is not a repeat exploit of the "
                          f"same route/counterparty."),
            ))

        # High-frequency recurrence across many prior cases (independent of
        # outcome): the same counterparty/amount seen repeatedly.
        distinct_cases = {m["case_id"] for m in matches}
        if len(distinct_cases) >= _FREQUENCY_THRESHOLD:
            amt = f"{t.amount:,.2f}" if t.amount is not None else "n/a"
            findings.append(Finding(
                agent=AGENT, code="MEM_HIGH_FREQUENCY_PATTERN",
                title="Counterparty/amount recurs across many prior cases",
                severity=Severity.MEDIUM, weight=10, page=t.page,
                field=f"row {t.index}",
                evidence=(f"The counterparty/amount for transaction #{t.index} "
                          f"('{t.description}', {amt}) has appeared in "
                          f"{len(distinct_cases)} prior cases ({case_list}). Repeated "
                          f"reuse of the same route across applications warrants review."),
            ))

    return AgentReport(agent=AGENT, findings=findings)
