"""Reference data used for cross-checking (ACRA-like registry, known vendors).

In production this would be backed by ACRA records, prior applications, and a
historical baseline store. Here it is a small in-memory fixture so the pipeline
has something to cross-reference against.
"""
from __future__ import annotations

# UEN -> registered entity name (ACRA-like).
ACRA_REGISTRY: dict[str, str] = {
    "201812345A": "Meridian Trading Pte Ltd",
    "200923456B": "Skyline Logistics Pte Ltd",
    "201534567C": "Harbourfront Foods Pte Ltd",
}

# Recurring, verified vendor / counterparty names seen in historical baselines.
KNOWN_VENDORS: set[str] = {
    "SP Services",
    "M1 Limited",
    "CPF Board",
    "IRAS GST",
    "DBS Merchant Payout",
    "Ninja Van SG",
    "Sheng Siong Distribution",
    "Comfort Transport",
}

# Historical monthly cash-flow baseline (very rough) per UEN, for anomaly checks.
CASHFLOW_BASELINE: dict[str, dict[str, float]] = {
    "201812345A": {"avg_monthly_credit": 85000.0, "avg_monthly_debit": 72000.0},
    "200923456B": {"avg_monthly_credit": 120000.0, "avg_monthly_debit": 98000.0},
    "201534567C": {"avg_monthly_credit": 60000.0, "avg_monthly_debit": 51000.0},
}

# Structuring threshold (e.g. cash transaction reporting threshold in SGD).
STRUCTURING_THRESHOLD = 10000.0
