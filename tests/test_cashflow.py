"""Tests for fraud_detect.nodes.cashflow (FEAT-001, Part A).

Green-by-construction: RiskReport objects are hand-built (following the
construction style in tests/test_checklist.py) and every asserted value is
computed by hand from the exact helper logic. Covers top inflow/outflow ordering
and size, recurring-counterparty grouping, notable-item detection (above-baseline
deposit via a CASHFLOW_BASELINE uen, a large round-number transfer, and a row
flagged by a 'row N' finding), monthly metrics with the divide-by-zero guard,
lowest-balance + went-negative, the volatility band + rationale, and the
large-transaction count against the 10,000 threshold. Exercises BOTH a clean
report and a tampered-style report with findings citing rows.
"""
from __future__ import annotations

from datetime import date

from fraud_detect.models import (
    ConfidenceBand,
    Finding,
    RiskBand,
    RiskReport,
    Routing,
    Severity,
    StatementSegment,
    Transaction,
)
from fraud_detect.nodes.cashflow import (
    LARGE_TXN_THRESHOLD,
    TOP_N,
    cashflow_highlights,
    cashflow_metrics,
)
from fraud_detect.reference_data import STRUCTURING_THRESHOLD


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _report(*, transactions, findings=None, uen="201812345A",
            opening_balance=1000.0, closing_balance=2000.0,
            period_start=date(2024, 1, 1), period_end=date(2024, 3, 31),
            segments=None):
    return RiskReport(
        case_id="CASE-CF",
        source_path="test.pdf",
        account_holder="Meridian Trading Pte Ltd",
        uen=uen,
        statement_period="2024-01-01 to 2024-03-31",
        account_number="123-456-789",
        statement_period_start=period_start,
        statement_period_end=period_end,
        risk_score=1,
        risk_band=RiskBand.LOW,
        confidence_score=85,
        confidence_band=ConfidenceBand.VERY_HIGH,
        routing=Routing.AUTO_CLEAR,
        routing_reason="clean",
        findings=list(findings or []),
        transactions=list(transactions),
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        declared_currency="SGD",
        statement_segments=list(segments or []),
    )


def _txn(index, d, desc, amount, balance):
    return Transaction(index=index, date=d, description=desc,
                       amount=amount, balance=balance)


# --------------------------------------------------------------------------- #
# Clean report: ordering, recurring grouping, metrics
# --------------------------------------------------------------------------- #
def _clean_txns():
    # Three complete months Jan/Feb/Mar 2024. Credits + / debits -.
    return [
        # January
        _txn(1, date(2024, 1, 5), "FAST FROM ACME PTE LTD", 40000.0, 41000.0),
        _txn(2, date(2024, 1, 20), "GIRO SP SERVICES", -5000.0, 36000.0),
        # February
        _txn(3, date(2024, 2, 5), "FAST FROM ACME PTE LTD", 45000.0, 81000.0),
        _txn(4, date(2024, 2, 18), "GIRO SP SERVICES", -5000.0, 76000.0),
        # March
        _txn(5, date(2024, 3, 5), "FAST FROM ACME PTE LTD", 50000.0, 126000.0),
        _txn(6, date(2024, 3, 9), "TRANSFER TO SUPPLIER ONE", -12000.0, 114000.0),
    ]


def test_largest_inflows_ordering_and_size():
    rep = _report(transactions=_clean_txns())
    h = cashflow_highlights(rep)
    inflows = h["largest_inflows"]
    assert len(inflows) == 3  # three credits present, <= TOP_N
    # Descending by amount: 50000, 45000, 40000
    assert [i["amount"] for i in inflows] == [50000.0, 45000.0, 40000.0]
    # All three are the same counterparty "Acme" after rail/suffix stripping.
    assert inflows[0]["counterparty"] == "Acme"
    assert TOP_N == 5


def test_largest_outflows_ordering_most_negative_first():
    rep = _report(transactions=_clean_txns())
    h = cashflow_highlights(rep)
    outflows = h["largest_outflows"]
    # Most negative first: -12000, -5000, -5000
    assert [o["amount"] for o in outflows] == [-12000.0, -5000.0, -5000.0]


def test_recurring_counterparties_grouped():
    rep = _report(transactions=_clean_txns())
    h = cashflow_highlights(rep)
    recurring = {g["counterparty"]: g for g in h["recurring_counterparties"]}
    # Acme appears 3x (all credits), SP Services 2x (all debits).
    assert recurring["Acme"]["count"] == 3
    assert recurring["Acme"]["total_amount"] == 135000.0
    assert recurring["Sp Services"]["count"] == 2
    assert recurring["Sp Services"]["total_amount"] == -10000.0
    # Supplier One appears once -> not recurring.
    assert "Supplier One" not in recurring


def test_metrics_monthly_and_large_count():
    rep = _report(transactions=_clean_txns())
    m = cashflow_metrics(rep)
    # Period Jan 1 - Mar 31 2024 -> 3 months.
    assert m["month_count"] == 3
    # total_inflow = 40000 + 45000 + 50000 = 135000
    assert m["total_inflow"] == 135000.0
    # total_outflow = 5000 + 5000 + 12000 = 22000
    assert m["total_outflow"] == 22000.0
    assert m["avg_monthly_inflow"] == 45000.0            # 135000 / 3
    assert round(m["avg_monthly_outflow"], 6) == round(22000.0 / 3, 6)
    # net monthly = (135000 - 22000) / 3
    assert round(m["net_monthly_cashflow"], 6) == round(113000.0 / 3, 6)
    # Large transactions (abs >= 10000): the three credits + the 12000 debit = 4.
    assert m["large_txn_count"] == 4
    assert m["large_txn_threshold"] == STRUCTURING_THRESHOLD == LARGE_TXN_THRESHOLD == 10000.0


def test_lowest_balance_and_went_negative_flag():
    rep = _report(transactions=_clean_txns())
    m = cashflow_metrics(rep)
    # Balances include opening 1000 and closing 2000 plus all row balances; the
    # lowest is the opening 1000, none negative.
    assert m["lowest_balance"] == 1000.0
    assert m["went_negative"] is False
    assert m["liquidity_stress"] is False


def test_went_negative_true_when_balance_dips_below_zero():
    txns = [
        _txn(1, date(2024, 1, 5), "FAST FROM ACME PTE LTD", 5000.0, 5000.0),
        _txn(2, date(2024, 1, 20), "TRANSFER TO X", -8000.0, -3000.0),
    ]
    rep = _report(transactions=txns, opening_balance=1000.0,
                  closing_balance=-3000.0)
    m = cashflow_metrics(rep)
    assert m["lowest_balance"] == -3000.0
    assert m["went_negative"] is True
    assert m["liquidity_stress"] is True


def test_volatility_band_present_with_rationale():
    rep = _report(transactions=_clean_txns())
    m = cashflow_metrics(rep)
    # Three complete months of inflows 40000/45000/50000 -> a band + rationale.
    assert m["volatility_band"] in ("Low", "Medium", "High")
    assert m["volatility_rationale"]


# --------------------------------------------------------------------------- #
# Divide-by-zero / missing-period guard
# --------------------------------------------------------------------------- #
def test_metrics_guard_no_period_no_dates():
    # No statement period AND no transaction dates -> month_count None, monthly
    # figures None (no crash).
    txns = [Transaction(index=1, amount=1000.0, balance=1000.0)]
    rep = _report(transactions=txns, period_start=None, period_end=None)
    m = cashflow_metrics(rep)
    assert m["month_count"] is None
    assert m["avg_monthly_inflow"] is None
    assert m["avg_monthly_outflow"] is None
    assert m["net_monthly_cashflow"] is None
    # Volatility cannot be assessed with no complete months.
    assert m["volatility_band"] == "n/a"
    assert m["volatility_rationale"]


def test_empty_report_does_not_crash():
    rep = _report(transactions=[], period_start=None, period_end=None,
                  opening_balance=None, closing_balance=None)
    h = cashflow_highlights(rep)
    assert h["largest_inflows"] == []
    assert h["largest_outflows"] == []
    assert h["recurring_counterparties"] == []
    assert h["notable_items"] == []
    m = cashflow_metrics(rep)
    assert m["month_count"] is None
    assert m["lowest_balance"] is None
    assert m["went_negative"] is False
    assert m["large_txn_count"] == 0


# --------------------------------------------------------------------------- #
# Notable items: above-baseline deposit + large round-number + flagged row
# --------------------------------------------------------------------------- #
def test_notable_above_baseline_deposit():
    # Baseline avg_monthly_credit for 201812345A is 85000; cutoff = 85000 * 1.5
    # = 127500. A single 130000 deposit exceeds it.
    txns = [
        _txn(1, date(2024, 1, 5), "FAST FROM ACME PTE LTD", 130000.0, 131000.0),
        _txn(2, date(2024, 2, 5), "FAST FROM ACME PTE LTD", 40000.0, 171000.0),
    ]
    rep = _report(transactions=txns)
    h = cashflow_highlights(rep)
    whys = [n["why"] for n in h["notable_items"]]
    assert any("baseline average monthly credit" in w for w in whys)
    # The 40000 deposit is below the cutoff -> no baseline note for it.
    baseline_items = [n for n in h["notable_items"]
                      if "baseline average monthly credit" in n["why"]]
    assert all(n["amount"] == 130000.0 for n in baseline_items)


def test_notable_large_round_number_transfer():
    txns = [
        _txn(1, date(2024, 1, 5), "TRANSFER TO OFFSHORE", -25000.0, -24000.0),
        _txn(2, date(2024, 1, 6), "COFFEE", -12.34, -24012.34),
    ]
    rep = _report(transactions=txns, uen="999999999Z")  # no baseline for this uen
    h = cashflow_highlights(rep)
    whys = [n["why"] for n in h["notable_items"]]
    assert any("round-number transfer" in w for w in whys)
    round_items = [n for n in h["notable_items"]
                   if "round-number transfer" in n["why"]]
    assert all(n["amount"] == -25000.0 for n in round_items)


def test_notable_flagged_row_from_finding():
    # Tampered-style: a finding cites 'row 2'. The helper maps it to the txn
    # whose index == 2 and marks it flagged.
    txns = [
        _txn(1, date(2024, 1, 5), "FAST FROM ACME PTE LTD", 5000.0, 6000.0),
        _txn(2, date(2024, 1, 20), "SUDDEN DEPOSIT", 9000.0, 15000.0),
    ]
    findings = [
        Finding(agent="Node5:Memory", code="RAPID_PASS_THROUGH",
                title="Rapid pass-through", severity=Severity.HIGH, weight=20,
                evidence="funds in and out quickly", field="row 2"),
    ]
    rep = _report(transactions=txns, findings=findings, uen="999999999Z")
    h = cashflow_highlights(rep)
    flagged = [n for n in h["notable_items"] if n.get("flagged")]
    assert len(flagged) == 1
    assert flagged[0]["row"] == 2
    assert flagged[0]["amount"] == 9000.0
    assert "Rapid pass-through" in flagged[0]["why"]


def test_findings_without_row_field_are_ignored():
    txns = [_txn(1, date(2024, 1, 5), "FAST FROM ACME PTE LTD", 5000.0, 6000.0)]
    findings = [
        Finding(agent="Node1:Ingestion", code="DQ_MISSING_UEN",
                title="Missing UEN", severity=Severity.MEDIUM, weight=5,
                evidence="no uen", field="header"),
    ]
    rep = _report(transactions=txns, findings=findings, uen="999999999Z")
    h = cashflow_highlights(rep)
    assert [n for n in h["notable_items"] if n.get("flagged")] == []


def test_metrics_uses_segments_for_month_count():
    # When segments are present they drive the month count. Two segments spanning
    # Jan-Feb 2024 -> 2 months even though the declared period says 3.
    segs = [
        StatementSegment(period_start=date(2024, 1, 1), period_end=date(2024, 1, 31)),
        StatementSegment(period_start=date(2024, 2, 1), period_end=date(2024, 2, 29)),
    ]
    txns = [
        _txn(1, date(2024, 1, 5), "FAST FROM ACME PTE LTD", 20000.0, 21000.0),
        _txn(2, date(2024, 2, 5), "FAST FROM ACME PTE LTD", 20000.0, 41000.0),
    ]
    rep = _report(transactions=txns, segments=segs,
                  period_start=date(2024, 1, 1), period_end=date(2024, 3, 31))
    m = cashflow_metrics(rep)
    assert m["month_count"] == 2
    assert m["avg_monthly_inflow"] == 20000.0  # 40000 / 2
