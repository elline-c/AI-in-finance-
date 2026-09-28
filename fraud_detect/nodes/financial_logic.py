"""Node 3 — Financial Logic & Validation Agent.

Validates the *numbers and behaviour* of the statement:

    * Mathematical continuity — every running balance must equal the previous
      balance plus the row amount (row-by-row reconciliation).
    * Opening/closing consistency — first/last balances match the header.
    * Period continuity — no unexplained multi-week gaps between transactions.
    * Suspicious transactions — sudden deposits far above the historical baseline
      and inflows from unverified / shell-like counterparties.
    * Structuring — repeated amounts sitting just below the reporting threshold.
    * Entity cross-check — account-holder name vs the ACRA record for the UEN
      (catches statements reused under a different company).

Findings are explainable and carry the offending page/row where possible.
"""
from __future__ import annotations

from collections import Counter

from ..models import AgentReport, Finding, IngestionResult, Severity, Transaction
from ..reference_data import (
    ACRA_REGISTRY,
    CASHFLOW_BASELINE,
    KNOWN_VENDORS,
    STRUCTURING_THRESHOLD,
)

AGENT = "Node3:FinancialLogic"

_BALANCE_TOL = 0.01           # rounding tolerance
_GAP_DAYS = 25                # unexplained-gap threshold
_STRUCTURING_WINDOW = 0.05    # within 5% below threshold
_SUSPICIOUS_MULTIPLE = 3.0    # single credit > 3x avg monthly credit


def _reconcile(ing: IngestionResult) -> list[Finding]:
    findings: list[Finding] = []
    txns = ing.transactions
    if not txns:
        return findings

    prev_balance = ing.opening_balance
    for t in txns:
        if prev_balance is None or t.amount is None or t.balance is None:
            prev_balance = t.balance
            continue
        expected = round(prev_balance + t.amount, 2)
        if abs(expected - t.balance) > _BALANCE_TOL:
            findings.append(Finding(
                agent=AGENT, code="FL_BALANCE_BREAK",
                title="Running balance does not reconcile",
                severity=Severity.HIGH, weight=26, page=t.page,
                field=f"row {t.index}",
                evidence=(f"Row {t.index} ({t.date} {t.description}): "
                          f"expected balance {expected:,.2f} "
                          f"(prev {prev_balance:,.2f} {'+' if t.amount>=0 else '-'} "
                          f"{abs(t.amount):,.2f}) but statement shows {t.balance:,.2f}. "
                          f"Discrepancy of {t.balance - expected:,.2f} indicates an "
                          f"altered amount or balance."),
            ))
        prev_balance = t.balance
    return findings


def _closing_check(ing: IngestionResult) -> list[Finding]:
    findings: list[Finding] = []
    if ing.closing_balance is not None and ing.transactions:
        last = ing.transactions[-1].balance
        if last is not None and abs(last - ing.closing_balance) > _BALANCE_TOL:
            findings.append(Finding(
                agent=AGENT, code="FL_CLOSING_MISMATCH",
                title="Closing balance disagrees with last row",
                severity=Severity.MEDIUM, weight=12,
                evidence=(f"Header closing balance {ing.closing_balance:,.2f} does not "
                          f"match the final row balance {last:,.2f}."),
            ))
    return findings


def _period_continuity(ing: IngestionResult) -> list[Finding]:
    findings: list[Finding] = []
    dated = [t for t in ing.transactions if t.date]
    for a, b in zip(dated, dated[1:]):
        gap = (b.date - a.date).days
        if gap > _GAP_DAYS:
            findings.append(Finding(
                agent=AGENT, code="FL_PERIOD_GAP",
                title="Unexplained gap in statement period",
                severity=Severity.MEDIUM, weight=12, page=b.page,
                evidence=(f"{gap}-day gap between {a.date} and {b.date} with no "
                          f"transactions. Continuous business accounts rarely go this "
                          f"long silent; may indicate removed pages/rows."),
            ))
    return findings


def _suspicious_transactions(ing: IngestionResult) -> list[Finding]:
    findings: list[Finding] = []
    baseline = CASHFLOW_BASELINE.get(ing.uen or "", {})
    avg_credit = baseline.get("avg_monthly_credit")

    for t in ing.transactions:
        if t.amount is None or t.amount <= 0:
            continue
        # Sudden massive deposit vs baseline
        if avg_credit and t.amount > _SUSPICIOUS_MULTIPLE * avg_credit:
            findings.append(Finding(
                agent=AGENT, code="FL_SUDDEN_DEPOSIT",
                title="Sudden deposit far above historical baseline",
                severity=Severity.HIGH, weight=20, page=t.page,
                field=f"row {t.index}",
                evidence=(f"Credit of {t.amount:,.2f} on {t.date} ({t.description}) is "
                          f"{t.amount/avg_credit:.1f}x the historical average monthly "
                          f"credit ({avg_credit:,.0f}). Off-baseline inflow warrants review."),
            ))
        # Unverified counterparty on a large inflow
        desc = (t.description or "").strip()
        looks_shell = any(k in desc.upper() for k in ("UNKNOWN", "LLC", "HOLDINGS", "TT"))
        known = any(v.upper() in desc.upper() for v in KNOWN_VENDORS)
        if t.amount >= 50000 and looks_shell and not known:
            findings.append(Finding(
                agent=AGENT, code="FL_UNVERIFIED_COUNTERPARTY",
                title="Large inflow from unverified counterparty",
                severity=Severity.MEDIUM, weight=12, page=t.page,
                field=f"row {t.index}",
                evidence=(f"Large credit of {t.amount:,.2f} from '{desc}', which does not "
                          f"match any known/recurring vendor and has shell-like naming."),
            ))
    return findings


def _structuring(ing: IngestionResult) -> list[Finding]:
    findings: list[Finding] = []
    lo = STRUCTURING_THRESHOLD * (1 - _STRUCTURING_WINDOW)
    near = [t for t in ing.transactions
            if t.amount is not None and lo <= t.amount < STRUCTURING_THRESHOLD]
    if len(near) >= 3:
        rows = ", ".join(f"#{t.index}({t.amount:,.0f})" for t in near)
        findings.append(Finding(
            agent=AGENT, code="FL_STRUCTURING",
            title="Possible structuring below reporting threshold",
            severity=Severity.HIGH, weight=20, page=near[0].page,
            evidence=(f"{len(near)} deposits fall in {lo:,.0f}-{STRUCTURING_THRESHOLD:,.0f} "
                      f"(just under the {STRUCTURING_THRESHOLD:,.0f} reporting threshold): "
                      f"{rows}. Repeated near-threshold amounts suggest deliberate splitting."),
        ))
    return findings


def _entity_crosscheck(ing: IngestionResult) -> list[Finding]:
    findings: list[Finding] = []
    if not ing.uen:
        return findings
    registered = ACRA_REGISTRY.get(ing.uen)
    if registered is None:
        findings.append(Finding(
            agent=AGENT, code="FL_UEN_NOT_FOUND",
            title="UEN not found in registry",
            severity=Severity.MEDIUM, weight=10,
            evidence=f"UEN {ing.uen} has no matching ACRA record.",
        ))
    elif ing.account_holder and registered.lower() != ing.account_holder.strip().lower():
        findings.append(Finding(
            agent=AGENT, code="FL_UEN_NAME_MISMATCH",
            title="Account holder does not match ACRA record",
            severity=Severity.HIGH, weight=22,
            evidence=(f"Statement holder '{ing.account_holder}' does not match the ACRA "
                      f"record '{registered}' for UEN {ing.uen}. May indicate a statement "
                      f"reused under a different company."),
        ))
    return findings


def analyze(ing: IngestionResult) -> AgentReport:
    findings: list[Finding] = []
    findings += _reconcile(ing)
    findings += _closing_check(ing)
    findings += _period_continuity(ing)
    findings += _suspicious_transactions(ing)
    findings += _structuring(ing)
    findings += _entity_crosscheck(ing)
    return AgentReport(agent=AGENT, findings=findings)
