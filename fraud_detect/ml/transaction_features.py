"""Feature engineering for the transaction-fraud model.

Stage 1: single-row features joined with static, per-account KYC attributes
from ``accounts.csv``.

Stage 2 (this module, now active): account-history features computed per account
over rolling time windows — this is where the real lift is (the dataset data card
reports single-row baseline PR-AUC ~0.57 -> +account history ~0.84). The windows
capture the AML typologies the data card describes: fan-in/out bursts, rapid
pass-through (out/in ratio spikes), structuring (many deposits in a short window),
and dormant-reactivation (a long inter-transaction gap followed by a large inflow).

NO-LOOK-AHEAD CONTRACT (the reason account history is easy to get wrong):
    * Rows are sorted by ``account_id`` then ``timestamp`` before any window.
    * Every rolling window is PAST-AND-CURRENT ONLY — it includes the current row
      and earlier rows, never future rows. Using a centred or trailing-future
      window would leak the outcome into the features.
    * Windows are grouped by ``account_id`` so one account's history never bleeds
      into another's.

LEAKAGE GUARDS (from the dataset data card — do not relax):
    * Target column is ``is_fraud``.
    * ``fraud_pattern`` and ``ring_id`` ARE labels -> never used as features.
    * ``eval_tags.csv`` is for error analysis only -> never joined as features.
    * The dataset-provided ``split`` column is authoritative -> never re-split
      by row (that leaks account history across splits). Because internal
      transfers are kept within a split, the per-account windows below also stay
      inside a single split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "is_fraud"
SPLIT_COL = "split"

# Columns that leak the label or are otherwise forbidden as features.
LEAKY_COLUMNS = {"is_fraud", "fraud_pattern", "ring_id", "split"}

# Raw transaction columns we turn into model features.
_CATEGORICAL = [
    "direction",
    "channel",
    "counterparty_type",
    "counterparty_country",
    "merchant_category",
    "segment",            # from accounts.csv
    "age_band",           # from accounts.csv
    "business_type",      # from accounts.csv
    "kyc_risk_rating",    # from accounts.csv
]
_NUMERIC = [
    "amount_sgd",
    "balance_after_sgd",
    "hour",               # engineered from timestamp
    "day_of_week",        # engineered from timestamp
    "declared_monthly_income_sgd",      # from accounts.csv
    "declared_monthly_turnover_sgd",    # from accounts.csv
    "account_age_days",   # engineered: timestamp - open_date
]

# Stage 2 account-history features (engineered per account, past-and-current
# only). Kept in their own list so a caller can train a single-row-only model by
# excluding them, and so the no-look-ahead contract is easy to audit in one place.
_HISTORY = [
    "log_amount",          # log1p(amount) — stabilises the heavy right tail
    "n_txn_24h",           # txns on this account in the last 24h (burst detector)
    "n_txn_7d",            # txns on this account in the last 7 days
    "n_cr_7d",             # credit count in last 7d (fan-in)
    "cr_sum_7d",           # total credited in last 7d
    "dr_sum_7d",           # total debited in last 7d
    "out_in_ratio_7d",     # dr_sum_7d / (cr_sum_7d + 1) — pass-through / mule exit
    "gap_hours",           # hours since this account's previous txn (dormancy)
    "amt_vs_acct_median",  # amount / expanding median amount for the account
]

FEATURE_COLUMNS = _NUMERIC + _HISTORY + _CATEGORICAL


def load_joined(data_dir: str) -> pd.DataFrame:
    """Load transactions joined with static account attributes.

    ``account_fraud_pattern`` and ``is_suspicious_account`` from accounts.csv are
    label-derived and are intentionally dropped.
    """
    import os

    txns = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    accts = pd.read_csv(os.path.join(data_dir, "accounts.csv"))

    # Keep only safe, static account attributes (drop account-level labels).
    acct_safe = accts[[
        "account_id", "segment", "open_date", "age_band",
        "declared_monthly_income_sgd", "declared_monthly_turnover_sgd",
        "business_type", "kyc_risk_rating",
    ]].copy()

    df = txns.merge(acct_safe, on="account_id", how="left")
    return df


def _add_account_history(out: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
    """Add Stage 2 per-account rolling-window features in place.

    All windows are PAST-AND-CURRENT ONLY (they include the current row and
    earlier rows for the same account, never future rows), so no feature can see
    the outcome it is used to predict. Rows are temporarily sorted by
    ``(account_id, timestamp)``; the original row order is restored before return
    so the caller's indexing/splitting is unaffected.
    """
    out = out.copy()
    out["_ts"] = ts
    out["_orig_order"] = np.arange(len(out))

    out["log_amount"] = np.log1p(pd.to_numeric(out["amount_sgd"], errors="coerce")
                                 .fillna(0.0).clip(lower=0.0))
    direction = out.get("direction", pd.Series("", index=out.index)).astype(str)
    is_cr = (direction.str.upper() == "CR").astype("float64")
    amount = pd.to_numeric(out["amount_sgd"], errors="coerce").fillna(0.0)
    out["_is_cr"] = is_cr
    out["_cr_amt"] = amount * is_cr
    out["_dr_amt"] = amount * (1.0 - is_cr)

    # Sort by account then time so rolling windows see history in order.
    out = out.sort_values(["account_id", "_ts"]).reset_index(drop=True)

    # Time-indexed rolling needs the timestamp as the index per group.
    g = out.set_index("_ts").groupby("account_id", sort=False)
    out["n_txn_24h"] = g["amount_sgd"].rolling("24h").count().to_numpy()
    out["n_txn_7d"] = g["amount_sgd"].rolling("7D").count().to_numpy()
    out["n_cr_7d"] = g["_is_cr"].rolling("7D").sum().to_numpy()
    out["cr_sum_7d"] = g["_cr_amt"].rolling("7D").sum().to_numpy()
    out["dr_sum_7d"] = g["_dr_amt"].rolling("7D").sum().to_numpy()
    out["out_in_ratio_7d"] = out["dr_sum_7d"] / (out["cr_sum_7d"] + 1.0)

    # Hours since the account's previous txn (first txn on an account -> 0).
    out["gap_hours"] = (out.groupby("account_id")["_ts"].diff()
                        .dt.total_seconds() / 3600.0).fillna(0.0)

    # Amount vs the account's running (expanding, past-and-current) median.
    acct_amount = pd.to_numeric(out["amount_sgd"], errors="coerce")
    acct_median = out.groupby("account_id")["amount_sgd"].transform(
        lambda s: s.expanding().median())
    out["amt_vs_acct_median"] = acct_amount / (acct_median + 1.0)

    # Restore the caller's original row order and drop scratch columns.
    out = out.sort_values("_orig_order").reset_index(drop=True)
    out = out.drop(columns=["_ts", "_orig_order", "_is_cr", "_cr_amt", "_dr_amt"])
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with engineered feature columns + target + split.

    The input is the joined transactions/accounts frame from ``load_joined``.
    """
    out = df.copy()

    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["hour"] = ts.dt.hour.astype("float64")
    out["day_of_week"] = ts.dt.dayofweek.astype("float64")

    open_dt = pd.to_datetime(out["open_date"], errors="coerce", utc=True)
    out["account_age_days"] = (ts - open_dt).dt.days.astype("float64")

    # Stage 2: per-account rolling-window history (past-and-current only).
    out = _add_account_history(out, ts)

    # Ensure every expected categorical column exists and is string-typed so the
    # one-hot/ordinal encoder sees consistent dtypes (missing -> "NA").
    for col in _CATEGORICAL:
        if col not in out.columns:
            out[col] = "NA"
        out[col] = out[col].fillna("NA").astype(str)

    for col in _NUMERIC + _HISTORY:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    keep = FEATURE_COLUMNS + [TARGET, SPLIT_COL]
    return out[keep]


def split_frames(feat: pd.DataFrame):
    """Split into (X_train, y_train, X_val, y_val, X_test, y_test) using the
    dataset-provided ``split`` column — never a random row split."""
    parts = {}
    for name in ("train", "val", "test"):
        sub = feat[feat[SPLIT_COL] == name]
        X = sub[FEATURE_COLUMNS]
        y = sub[TARGET].astype(int)
        parts[name] = (X, y)
    return parts
