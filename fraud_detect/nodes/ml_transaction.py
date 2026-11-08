"""Node 6a — ML Transaction-Fraud Agent.

Scores each parsed transaction with the trained AML classifier and raises an
explainable Finding for rows above the alert threshold. It ADDS evidence; it
never auto-rejects, matching the governance stance of the rule-based nodes.

Fail-safe: if the model is missing, the input is unusable, or scoring errors,
the node returns no findings (it never fabricates a score).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..models import AgentReport, Finding, IngestionResult, Severity
from ..ml import transaction_features as tf
from ..ml.gnn import gnn_lookup

AGENT = "Node6a:MLTransaction"
# Distinct sub-label so the GNN signal counts as an independent agent for Node 4
# corroboration, while staying within Node 6a (see Design: Components §5).
AGENT_GNN = "Node6a:MLTransaction:GNN"

# Probability thresholds -> severity/weight. Governed values (should move behind
# the approved-configuration snapshot from spec task 7.2).
_HIGH_PROB = 0.80
_MEDIUM_PROB = 0.50

# GNN graph-score thresholds -> severity/weight. Kept consistent with the GBM
# finding's weights above so the two independent signals contribute comparably.
_GNN_HIGH = 0.80
_GNN_MEDIUM = 0.50


# The per-transaction model is switched OFF in the live app by default. Trained
# on the available dataset, it either relies on fields a PDF statement does not
# carry (channel, counterparty, KYC) or, restricted to statement-only fields,
# mistakes ordinary business-sized transactions for fraud: it raised false alarms
# on clean statements without adding detection the rule-based checks lack. Set
# this environment variable to 1/true/yes to switch it back on (e.g. once it is
# retrained on representative statements). The GNN lookup in this node is
# separate and is not affected.
ENABLE_ENV = "FRAUD_ENABLE_TXN_MODEL"


def load_live_model(path: str = "models/txn_fraud_clf.joblib", env=None):
    """Model for live screening, or ``None`` while the model is switched off.

    Returns ``None`` unless ``FRAUD_ENABLE_TXN_MODEL`` is set to a true value;
    then behaves like :func:`load_model` (also ``None`` if the file is missing).
    """
    import os

    env = os.environ if env is None else env
    if str(env.get(ENABLE_ENV, "")).strip().lower() not in ("1", "true", "yes", "on"):
        return None
    return load_model(path)


def load_model(path: str = "models/txn_fraud_clf.joblib"):
    """Load the saved pipeline bundle, or return None if unavailable (fail-safe)."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def _feature_spec(model):
    """Return ``(feature builder, column list)`` matching how ``model`` was trained.

    Bundles from ``train_transaction`` record their ``feature_set``: ``"pdf"``
    models are trained on statement-only fields with date-only timestamps, so
    they must be fed by ``build_pdf_features``. Bundles without the key predate
    feature sets and used the full column list.
    """
    if isinstance(model, dict) and model.get("feature_set") == "pdf":
        return tf.build_pdf_features, list(model["feature_columns"])
    if isinstance(model, dict) and model.get("feature_columns"):
        return tf.build_features, list(model["feature_columns"])
    return tf.build_features, tf.FEATURE_COLUMNS


def _rows_from_ingestion(ing: IngestionResult) -> pd.DataFrame:
    """Build a model-ready frame from parsed transactions.

    The ingestion model carries less than the training dataset, so unavailable
    columns fall back to safe defaults; build_features tolerates missing columns.
    """
    records = []
    for t in ing.transactions:
        records.append({
            "account_id": ing.uen or "NA",
            "timestamp": t.date.isoformat() if t.date else None,
            "direction": "CR" if (t.amount or 0) >= 0 else "DR",
            "amount_sgd": abs(t.amount) if t.amount is not None else 0.0,
            "channel": "NA",
            "counterparty_type": "NA",
            "counterparty_country": "NA",
            "merchant_category": "NA",
            "balance_after_sgd": t.balance if t.balance is not None else 0.0,
            "open_date": None,
            "segment": "NA",
            "age_band": "NA",
            "business_type": "NA",
            "kyc_risk_rating": "NA",
            "declared_monthly_income_sgd": 0.0,
            "declared_monthly_turnover_sgd": 0.0,
            "meta_index": t.index,
            "meta_desc": t.description,
            "meta_amount": t.amount,
            "meta_page": t.page,
        })
    return pd.DataFrame(records)


def analyze(ing: IngestionResult, model=None, gnn_scores=None) -> AgentReport:
    """Score transactions with the GBM classifier and, independently, consult the
    precomputed GNN graph-fraud score for the account.

    The two scorers are independent evidence sources: the GBM path emits
    ``ML_TXN_FRAUD_LIKELIHOOD`` findings exactly as before, and the GNN path emits
    at most one ``ML_TXN_GNN_RISK`` finding keyed on ``ing.uen``. Either path may
    run without the other — when ``gnn_scores`` is ``None`` the behaviour is
    byte-for-byte identical to the pre-GNN node.
    """
    findings: list[Finding] = []
    # A GNN finding only makes sense for a screened statement with transactions
    # (the account under review). With no transactions there is nothing to do for
    # either path.
    if not ing.transactions:
        return AgentReport(agent=AGENT, findings=findings)

    # --- Existing GBM path (unchanged) -------------------------------------- #
    if model is not None:
        _append_gbm_findings(ing, model, findings)

    # --- GNN graph-score path (independent evidence) ------------------------ #
    _append_gnn_finding(ing, gnn_scores, findings)

    return AgentReport(agent=AGENT, findings=findings)


def _append_gbm_findings(ing: IngestionResult, model, findings: list[Finding]) -> None:
    """Score each row with the GBM pipeline and append ML_TXN_FRAUD_LIKELIHOOD
    findings. Behaviourally identical to the original inline implementation."""
    try:
        df = _rows_from_ingestion(ing)
        build, columns = _feature_spec(model)
        feat = build(df.assign(is_fraud=0, split="train"))  # dummy target/split cols
        X = feat[columns]
        pipe = model["pipeline"] if isinstance(model, dict) else model
        probs = pipe.predict_proba(X)[:, 1]
    except Exception:
        # Malformed input / model mismatch -> fail safe, no fabricated findings.
        return

    for row, prob in zip(df.itertuples(index=False), probs):
        prob = float(prob)
        if prob >= _HIGH_PROB:
            sev, weight = Severity.HIGH, 22
        elif prob >= _MEDIUM_PROB:
            sev, weight = Severity.MEDIUM, 12
        else:
            continue
        amt = f"{row.meta_amount:,.2f}" if row.meta_amount is not None else "n/a"
        findings.append(Finding(
            agent=AGENT, code="ML_TXN_FRAUD_LIKELIHOOD",
            title="Model flags elevated transaction-fraud likelihood",
            severity=sev, weight=weight, page=row.meta_page,
            field=f"row {row.meta_index}",
            evidence=(f"Transaction #{row.meta_index} ('{row.meta_desc}', {amt}) scored "
                      f"{prob:.0%} fraud likelihood by the AML classifier. "
                      f"Evidence for a reviewer; never an automatic rejection."),
        ))


def _append_gnn_finding(ing: IngestionResult, gnn_scores, findings: list[Finding]) -> None:
    """Look up the account's precomputed GNN graph-fraud score and, when elevated,
    append exactly one ML_TXN_GNN_RISK finding.

    Fail-safe: no store, unknown key, or a None/below-threshold score -> no finding.
    The lookup imports only joblib (no torch), so this is safe on the live path.
    """
    account_id = ing.uen or "NA"
    score = gnn_lookup.score_account(gnn_scores, account_id)
    if score is None:
        return

    if score >= _GNN_HIGH:
        sev, weight = Severity.HIGH, 22
    elif score >= _GNN_MEDIUM:
        sev, weight = Severity.MEDIUM, 12
    else:
        return

    findings.append(Finding(
        agent=AGENT_GNN, code="ML_TXN_GNN_RISK",
        title="Graph model flags elevated account-level fraud risk",
        severity=sev, weight=weight,
        field=f"account {account_id}",
        evidence=(f"Account {account_id} scored {score:.0%} graph-derived fraud risk "
                  f"by the GNN AML model (precomputed over the internal transfer "
                  f"graph). Evidence for a reviewer; never an automatic rejection."),
    ))
