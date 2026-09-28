"""Node 3: Financial Logic & Validation Agent.

Checks the *numbers and the narrative they tell*:

  * Mathematical consistency: opening + sum(txns) == closing, and every running
    balance row is internally consistent.
  * Statement period continuity: transactions fall inside the period, are in
    date order, and have no unexplained multi-week gaps.
  * Structuring & splitting: repeated deposits just under a reporting threshold.
  * Anomalous flows: sudden massive deposits vs. the statement's own baseline,
    and rapid pass-through (money in and straight back out).
"""

from __future__ import annotations

from datetime import date
from statistics import mean, pstdev

from ..contracts import Finding, Node, NormalizedStatement, Severity

STRUCTURING_THRESHOLD = 10_000.0      # common cash-reporting threshold
STRUCTURING_BAND = 0.10               # "just under" = within 10% below
GAP_DAYS = 21                         # unexplained gap trigger


def _d(iso: str | None) -> date | None:
    if not iso:
        return None
    try:
        y, m, dd = (int(x) for x in iso[:10].split("-"))
        return date(y, m, dd)
    except (ValueError, TypeError):
        return None


class FinancialLogicAgent:
    node = Node.FINANCIAL_LOGIC

    def run(self, stmt: NormalizedStatement) -> list[Finding]:
        findings: list[Finding] = []
        txns = stmt.transactions

        # 1. Statement-total reconciliation ---------------------------------
        amounts = [t.amount for t in txns if t.amount is not None]
        if (
            stmt.opening_balance is not None
            and stmt.closing_balance is not None
            and len(amounts) == len(txns)
        ):
            expected = round(stmt.opening_balance + sum(amounts), 2)
            actual = round(stmt.closing_balance, 2)
            if abs(expected - actual) > 0.01:
                findings.append(
                    Finding(
                        self.node,
                        "STATEMENT_TOTAL_MISMATCH",
                        "Opening + transactions ≠ closing balance",
                        Severity.CRITICAL,
                        f"Opening {stmt.opening_balance:.2f} plus net transactions "
                        f"{sum(amounts):+.2f} should give {expected:.2f}, but the "
                        f"stated closing balance is {actual:.2f} "
                        f"(off by {actual - expected:+.2f}).",
                        {
                            "opening": stmt.opening_balance,
                            "net_txn": round(sum(amounts), 2),
                            "expected_closing": expected,
                            "stated_closing": actual,
                        },
                    )
                )

        # 2. Row-by-row running-balance check -------------------------------
        prev = stmt.opening_balance
        for i, t in enumerate(txns):
            if prev is None or t.amount is None or t.balance is None:
                prev = t.balance if t.balance is not None else prev
                continue
            expected = round(prev + t.amount, 2)
            if abs(expected - round(t.balance, 2)) > 0.01:
                findings.append(
                    Finding(
                        self.node,
                        "RUNNING_BALANCE_MISMATCH",
                        "Running balance does not add up",
                        Severity.HIGH,
                        f"Transaction #{i}: previous balance {prev:.2f} "
                        f"{t.amount:+.2f} should be {expected:.2f}, but the row "
                        f"shows {t.balance:.2f}. A common sign of an inserted or "
                        "edited line.",
                        {
                            "index": i,
                            "prev_balance": prev,
                            "amount": t.amount,
                            "expected": expected,
                            "stated": t.balance,
                        },
                    )
                )
            prev = t.balance

        # 3. Period continuity: dates in range + ordered + no gaps ----------
        p_start, p_end = _d(stmt.period_start), _d(stmt.period_end)
        dates = [(_d(t.date), i) for i, t in enumerate(txns)]
        last = None
        for dt, i in dates:
            if dt is None:
                continue
            if p_start and dt < p_start or p_end and dt > p_end:
                findings.append(
                    Finding(
                        self.node,
                        "TXN_OUTSIDE_PERIOD",
                        "Transaction dated outside the statement period",
                        Severity.HIGH,
                        f"Transaction #{i} dated {dt.isoformat()} falls outside "
                        f"the period {stmt.period_start}..{stmt.period_end}.",
                        {"index": i, "date": dt.isoformat()},
                    )
                )
            if last and dt < last:
                findings.append(
                    Finding(
                        self.node,
                        "TXN_OUT_OF_ORDER",
                        "Transactions not in chronological order",
                        Severity.MEDIUM,
                        f"Transaction #{i} dated {dt.isoformat()} precedes the "
                        "prior row — statements are normally strictly ordered.",
                        {"index": i, "date": dt.isoformat()},
                    )
                )
            if last and (dt - last).days > GAP_DAYS:
                findings.append(
                    Finding(
                        self.node,
                        "PERIOD_GAP",
                        "Unexplained gap in activity",
                        Severity.LOW,
                        f"{(dt - last).days} days with no transactions before "
                        f"#{i} ({dt.isoformat()}); verify no page is missing.",
                        {"index": i, "gap_days": (dt - last).days},
                    )
                )
            last = dt

        # 4. Structuring / splitting ----------------------------------------
        low, high = STRUCTURING_THRESHOLD * (1 - STRUCTURING_BAND), STRUCTURING_THRESHOLD
        near = [
            (i, t.amount)
            for i, t in enumerate(txns)
            if t.amount is not None and low <= t.amount < high
        ]
        if len(near) >= 3:
            findings.append(
                Finding(
                    self.node,
                    "STRUCTURING_PATTERN",
                    "Repeated deposits just under reporting threshold",
                    Severity.HIGH,
                    f"{len(near)} credits fall just below the "
                    f"{STRUCTURING_THRESHOLD:,.0f} reporting threshold — a classic "
                    "structuring/splitting pattern.",
                    {"count": len(near), "indexes": [i for i, _ in near]},
                )
            )

        # 5. Anomalous deposit vs. the statement's own baseline -------------
        credits = [t.amount for t in txns if t.amount and t.amount > 0]
        if len(credits) >= 4:
            mu, sigma = mean(credits), pstdev(credits)
            if sigma > 0:
                for i, t in enumerate(txns):
                    if t.amount and t.amount > 0 and (t.amount - mu) > 4 * sigma:
                        findings.append(
                            Finding(
                                self.node,
                                "ANOMALOUS_DEPOSIT",
                                "Sudden deposit far above baseline",
                                Severity.MEDIUM,
                                f"Transaction #{i} credit {t.amount:,.2f} is more "
                                f"than 4σ above the mean credit ({mu:,.2f}); "
                                "inconsistent with historical cash-flow pattern.",
                                {"index": i, "amount": t.amount,
                                 "mean_credit": round(mu, 2)},
                            )
                        )

        # 6. Rapid pass-through (in then straight back out) -----------------
        for i in range(len(txns) - 1):
            a, b = txns[i], txns[i + 1]
            if (
                a.amount and b.amount
                and a.amount > 0 and b.amount < 0
                and abs(abs(b.amount) - a.amount) / a.amount < 0.02
                and a.amount >= 5_000
            ):
                da, db = _d(a.date), _d(b.date)
                if da and db and (db - da).days <= 2:
                    findings.append(
                        Finding(
                            self.node,
                            "RAPID_PASS_THROUGH",
                            "Funds entered and exited almost immediately",
                            Severity.MEDIUM,
                            f"#{i} credit {a.amount:,.2f} is followed within "
                            f"{(db - da).days} day(s) by a near-identical debit "
                            f"({b.amount:,.2f}) to '{b.counterparty}' — a "
                            "pass-through pattern.",
                            {"in_index": i, "out_index": i + 1,
                             "amount": a.amount},
                        )
                    )
        return findings
