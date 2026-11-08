"""Cash-flow lens for the loan-officer console (FEAT-001, Part A).

Pure, deterministic helpers that turn a screened ``RiskReport`` into the
cash-flow picture a loan officer reads before deciding whether to lend. There is
NO fraud decision here and NO new network/LLM dependency: every number is either
derived straight from the parsed statement transactions or clearly labelled as a
band + rationale (never a fabricated precise statistic).

Two public functions:

  * ``cashflow_highlights(report)`` — the largest inflows/outflows, recurring
    counterparties, and "notable / unusual" items (above the account's own
    baseline average, large round-number transfers, and ANY row the screening
    pipeline already flagged via a finding whose ``field`` cites ``row N``).
  * ``cashflow_metrics(report)`` — average monthly inflow/outflow, net monthly
    cash flow, lowest balance in the period, a went-negative liquidity-stress
    flag, an inflow-volatility BAND (High/Medium/Low) with a stated rationale,
    and a count of large transactions above ``STRUCTURING_THRESHOLD``.

Transaction amounts are SIGNED (credits +, debits -), per the model contract.
All divisions are guarded; when an input is missing the value is ``None`` so the
caller/template can render a friendly em-dash substitute.
"""
from __future__ import annotations

import re
from datetime import date as date_type
from typing import Optional

from ..history import counterparty_key, monthly_credit_totals
from ..models import Finding, RiskReport, Transaction
from ..reference_data import CASHFLOW_BASELINE, STRUCTURING_THRESHOLD
from ..store import normalize_counterparty

# --------------------------------------------------------------------------- #
# Documented constants
# --------------------------------------------------------------------------- #
# How many items to list in each "largest inflows / largest outflows" table.
# Five keeps the officer's view focused on the material movements without
# drowning them in a full ledger reprint.
TOP_N = 5

# A counterparty seen this many times (or more) is treated as "recurring".
RECURRING_MIN_COUNT = 2

# The "large transaction" threshold. We deliberately REUSE the structuring /
# cash-reporting threshold from reference_data (SGD 10,000) rather than invent a
# second number, so "large" means the same thing across fraud screening and the
# lending view.
LARGE_TXN_THRESHOLD = STRUCTURING_THRESHOLD

# A deposit is "notably above baseline" when it exceeds the account's own
# historical average monthly credit by this multiple. Documented, not tuned.
ABOVE_BASELINE_MULTIPLE = 1.5

# A transfer counts as a "large round-number" item when it is at least this big
# AND lands on a round multiple of ROUND_STEP. Round-number large transfers are
# a classic thing an officer wants to eyeball.
ROUND_NUMBER_MIN = 10000.0
ROUND_STEP = 1000.0

# Matches a finding whose ``field`` ties it to a specific transaction row, e.g.
# "row 3" / "Row 12". Group 1 is the 0-or-1-based row number as printed.
_ROW_FIELD_RE = re.compile(r"\brow\s*(\d+)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Small internal helpers
# --------------------------------------------------------------------------- #
def _counterparty_label(txn: Transaction) -> str:
    """A human-readable counterparty/description for a highlight row."""
    key = counterparty_key(txn.description)
    if key:
        return key.title()
    return (txn.description or "").strip() or "(no description)"


def _date_str(d: Optional[date_type]) -> Optional[str]:
    return d.isoformat() if d is not None else None


def _is_round_number(amount: float) -> bool:
    """True when ``amount`` is a clean multiple of ROUND_STEP (e.g. 25,000)."""
    magnitude = abs(amount)
    return magnitude > 0 and (magnitude % ROUND_STEP == 0)


def _rows_flagged_by_findings(findings: list[Finding]) -> dict[int, list[Finding]]:
    """Map transaction-row number -> findings that cite that row via ``field``.

    Only findings whose ``field`` matches ``row N`` are included; a finding with
    no row reference is skipped (it is not row-specific).
    """
    out: dict[int, list[Finding]] = {}
    for f in findings or []:
        if not f.field:
            continue
        m = _ROW_FIELD_RE.search(f.field)
        if not m:
            continue
        row = int(m.group(1))
        out.setdefault(row, []).append(f)
    return out


def _txn_by_row(transactions: list[Transaction], row: int) -> Optional[Transaction]:
    """Resolve the transaction a ``row N`` finding points at.

    Findings cite rows using the transaction's own ``index`` (the pipeline's row
    identity). We match on ``index`` first and fall back to positional order so a
    hand-built report whose indices start at 1 still resolves.
    """
    for t in transactions:
        if t.index == row:
            return t
    # Positional fallback: treat row as 1-based position into the list.
    if 1 <= row <= len(transactions):
        return transactions[row - 1]
    return None


# --------------------------------------------------------------------------- #
# Public: highlights
# --------------------------------------------------------------------------- #
def cashflow_highlights(report: RiskReport) -> dict:
    """Top inflows/outflows, recurring counterparties and notable items.

    Every item carries ``date`` (ISO string or None), ``counterparty`` (string),
    ``amount`` (signed float) and a one-line ``why`` string. Safe on an empty
    report: all lists come back empty, nothing raises.
    """
    txns = list(report.transactions or [])

    inflows = [t for t in txns if t.amount is not None and t.amount > 0]
    outflows = [t for t in txns if t.amount is not None and t.amount < 0]

    top_inflows = sorted(inflows, key=lambda t: t.amount, reverse=True)[:TOP_N]
    # Largest outflows = most negative amounts.
    top_outflows = sorted(outflows, key=lambda t: t.amount)[:TOP_N]

    largest_inflows = [{
        "date": _date_str(t.date),
        "counterparty": _counterparty_label(t),
        "amount": t.amount,
        "why": f"One of the largest credits in the period ({t.amount:,.2f}).",
    } for t in top_inflows]

    largest_outflows = [{
        "date": _date_str(t.date),
        "counterparty": _counterparty_label(t),
        "amount": t.amount,
        "why": f"One of the largest debits in the period ({t.amount:,.2f}).",
    } for t in top_outflows]

    # ---- recurring counterparties (count >= RECURRING_MIN_COUNT) -------- #
    # Group by the strongest normalised key we can build: counterparty_key
    # strips payment rails + references; fall back to normalize_counterparty on
    # the raw description so a party with no rail prefix still groups.
    groups: dict[str, dict] = {}
    for t in txns:
        key = counterparty_key(t.description) or normalize_counterparty(t.description)
        if not key:
            continue
        g = groups.setdefault(key, {
            "counterparty": key.title(),
            "count": 0,
            "total_amount": 0.0,
        })
        g["count"] += 1
        if t.amount is not None:
            g["total_amount"] += t.amount
    recurring = [
        {
            "counterparty": g["counterparty"],
            "count": g["count"],
            "total_amount": g["total_amount"],
            "why": (f"Seen {g['count']} times in the period "
                    f"(net {g['total_amount']:,.2f})."),
        }
        for g in groups.values() if g["count"] >= RECURRING_MIN_COUNT
    ]
    recurring.sort(key=lambda g: (g["count"], abs(g["total_amount"])), reverse=True)

    # ---- notable / unusual items --------------------------------------- #
    notable: list[dict] = []

    # (a) Deposits well above the account's OWN baseline average monthly credit.
    baseline = CASHFLOW_BASELINE.get(report.uen or "") if report.uen else None
    if baseline:
        avg_credit = baseline.get("avg_monthly_credit")
        if avg_credit and avg_credit > 0:
            cutoff = avg_credit * ABOVE_BASELINE_MULTIPLE
            for t in inflows:
                if t.amount >= cutoff:
                    notable.append({
                        "date": _date_str(t.date),
                        "counterparty": _counterparty_label(t),
                        "amount": t.amount,
                        "why": (f"Deposit of {t.amount:,.2f} exceeds "
                                f"{ABOVE_BASELINE_MULTIPLE:g}x the account's "
                                f"baseline average monthly credit "
                                f"({avg_credit:,.2f})."),
                    })

    # (b) Large round-number transfers (either direction).
    for t in txns:
        if t.amount is None:
            continue
        if abs(t.amount) >= ROUND_NUMBER_MIN and _is_round_number(t.amount):
            notable.append({
                "date": _date_str(t.date),
                "counterparty": _counterparty_label(t),
                "amount": t.amount,
                "why": (f"Large round-number transfer ({t.amount:,.2f}); "
                        "round figures warrant a look."),
            })

    # (c) Any row the screening pipeline already flagged (field 'row N'). This is
    # what makes a tampered/escalated case show its fraud-flagged rows right
    # beside the lending picture.
    flagged = _rows_flagged_by_findings(report.findings)
    for row, fs in sorted(flagged.items()):
        t = _txn_by_row(txns, row)
        titles = "; ".join(sorted({f.title for f in fs if f.title}))
        notable.append({
            "date": _date_str(t.date) if t else None,
            "counterparty": _counterparty_label(t) if t else f"row {row}",
            "amount": t.amount if t else None,
            "why": (f"Flagged by screening: {titles}." if titles
                    else "Flagged by screening."),
            "flagged": True,
            "row": row,
        })

    return {
        "largest_inflows": largest_inflows,
        "largest_outflows": largest_outflows,
        "recurring_counterparties": recurring,
        "notable_items": notable,
    }


# --------------------------------------------------------------------------- #
# Public: metrics
# --------------------------------------------------------------------------- #
def _month_count(report: RiskReport) -> Optional[int]:
    """Number of months the statement spans, by the most reliable source.

    Preference order (each guarded):
      1. statement_segments period spans (distinct year-months covered);
      2. statement_period_start / statement_period_end;
      3. distinct transaction year-months.
    Returns None when nothing usable is present (so the caller guards divide).
    """
    def _span_months(start: Optional[date_type], end: Optional[date_type]) -> Optional[int]:
        if start is None or end is None or end < start:
            return None
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    # 1. segments
    starts = [s.period_start for s in report.statement_segments if s.period_start]
    ends = [s.period_end for s in report.statement_segments if s.period_end]
    if starts and ends:
        m = _span_months(min(starts), max(ends))
        if m:
            return m

    # 2. declared statement period
    m = _span_months(report.statement_period_start, report.statement_period_end)
    if m:
        return m

    # 3. distinct transaction year-months
    ym = {(t.date.year, t.date.month) for t in (report.transactions or []) if t.date}
    if ym:
        return len(ym)
    return None


def _volatility_band(monthly_inflows: list[float]) -> tuple[str, str]:
    """An inflow-volatility BAND + rationale, never a fake precise statistic.

    Uses the coefficient of variation (stdev / mean) of the per-month inflow
    totals, but reports it only as High / Medium / Low with a readable rationale.
    Fewer than two complete months -> 'n/a' with a rationale saying so.
    """
    vals = [v for v in monthly_inflows if v is not None]
    if len(vals) < 2:
        return ("n/a", "Not enough complete months to assess inflow volatility.")
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return ("n/a", "No positive monthly inflow to assess volatility against.")
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    cv = (variance ** 0.5) / mean
    if cv < 0.25:
        band = "Low"
    elif cv < 0.5:
        band = "Medium"
    else:
        band = "High"
    rationale = (
        f"{band} volatility: monthly inflows range "
        f"{min(vals):,.2f} to {max(vals):,.2f} around an average of "
        f"{mean:,.2f} across {len(vals)} complete months."
    )
    return (band, rationale)


def cashflow_metrics(report: RiskReport) -> dict:
    """Headline cash-flow metrics for the lending view.

    All amounts are derived from the signed transaction amounts. Monthly figures
    are None when the month count cannot be established (divide-by-zero guard).
    """
    txns = list(report.transactions or [])

    total_inflow = sum(t.amount for t in txns if t.amount and t.amount > 0)
    total_outflow = sum(-t.amount for t in txns if t.amount and t.amount < 0)

    months = _month_count(report)
    if months and months > 0:
        avg_monthly_inflow = total_inflow / months
        avg_monthly_outflow = total_outflow / months
        net_monthly_cashflow = (total_inflow - total_outflow) / months
    else:
        avg_monthly_inflow = None
        avg_monthly_outflow = None
        net_monthly_cashflow = None

    # Lowest balance + liquidity stress (balance ever negative).
    balances = [t.balance for t in txns if t.balance is not None]
    if report.opening_balance is not None:
        balances.append(report.opening_balance)
    if report.closing_balance is not None:
        balances.append(report.closing_balance)
    lowest_balance = min(balances) if balances else None
    went_negative = any(b < 0 for b in balances) if balances else False

    # Inflow volatility band from per-complete-month credit totals.
    seg_starts = [s.period_start for s in report.statement_segments if s.period_start]
    seg_ends = [s.period_end for s in report.statement_segments if s.period_end]
    period_start = min(seg_starts) if seg_starts else report.statement_period_start
    period_end = max(seg_ends) if seg_ends else report.statement_period_end
    monthly_inflows = monthly_credit_totals(txns, period_start, period_end)
    volatility_band, volatility_rationale = _volatility_band(monthly_inflows)

    # Large transactions above the documented threshold (either direction).
    large_txn_count = sum(
        1 for t in txns if t.amount is not None and abs(t.amount) >= LARGE_TXN_THRESHOLD
    )

    return {
        "month_count": months,
        "total_inflow": total_inflow,
        "total_outflow": total_outflow,
        "avg_monthly_inflow": avg_monthly_inflow,
        "avg_monthly_outflow": avg_monthly_outflow,
        "net_monthly_cashflow": net_monthly_cashflow,
        "lowest_balance": lowest_balance,
        "went_negative": went_negative,
        "liquidity_stress": went_negative,
        "volatility_band": volatility_band,
        "volatility_rationale": volatility_rationale,
        "large_txn_threshold": LARGE_TXN_THRESHOLD,
        "large_txn_count": large_txn_count,
    }
