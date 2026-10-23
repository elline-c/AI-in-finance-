"""Node 6b — ML Statement-Forgery Agent.

Scores the whole document with the trained forgery classifier and raises a
single explainable Finding when the document scores as likely-tampered. Adds
evidence only; never auto-rejects.

Fail-safe: missing model, unreadable file, or scoring error -> no findings.
"""
from __future__ import annotations

from ..models import AgentReport, Finding, IngestionResult, Severity
from ..ml import statement_features as sf

AGENT = "Node6b:MLStatement"

_HIGH_PROB = 0.80
_MEDIUM_PROB = 0.50


def load_model(path: str = "models/statement_forgery_clf.joblib"):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def _top_signals(feats: dict) -> str:
    """Human-readable summary of the strongest file-derived tampering signals."""
    bits = []
    if feats.get("balance_reconcile_breaks", 0) > 0:
        bits.append(f"{int(feats['balance_reconcile_breaks'])} balance-reconciliation break(s)")
    if feats.get("has_editing_software"):
        bits.append("editor software in metadata")
    if feats.get("moddate_after_creation"):
        bits.append("modified after creation")
    if feats.get("n_distinct_date_formats", 0) > 1:
        bits.append("mixed date formats")
    if feats.get("duplicate_amount_overlap", 0) > 0:
        bits.append("repeated amount tokens (possible overlay)")
    if feats.get("n_currency_styles", 0) > 1:
        bits.append("mixed currency notation")
    return "; ".join(bits) if bits else "aggregate document features"


def analyze(ing: IngestionResult, model=None) -> AgentReport:
    findings: list[Finding] = []
    if model is None:
        return AgentReport(agent=AGENT, findings=findings)

    feats = sf.extract_features(ing.source_path)
    if feats is None:
        return AgentReport(agent=AGENT, findings=findings)

    try:
        import pandas as pd
        clf = model["model"] if isinstance(model, dict) else model
        X = pd.DataFrame([[feats[name] for name in sf.FEATURE_NAMES]],
                         columns=sf.FEATURE_NAMES)
        prob = float(clf.predict_proba(X)[0][1])
    except Exception:
        return AgentReport(agent=AGENT, findings=findings)

    if prob >= _HIGH_PROB:
        sev, weight = Severity.HIGH, 24
    elif prob >= _MEDIUM_PROB:
        sev, weight = Severity.MEDIUM, 12
    else:
        return AgentReport(agent=AGENT, findings=findings)

    findings.append(Finding(
        agent=AGENT, code="ML_STATEMENT_FORGERY_LIKELIHOOD",
        title="Model flags elevated statement-forgery likelihood",
        severity=sev, weight=weight,
        evidence=(f"Forgery classifier scored this document {prob:.0%} likely "
                  f"tampered. Strongest file-derived signals: {_top_signals(feats)}. "
                  f"Evidence for a reviewer; never an automatic rejection."),
    ))
    return AgentReport(agent=AGENT, findings=findings)
