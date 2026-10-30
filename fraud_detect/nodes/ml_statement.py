"""Node 6b — ML Statement-Forgery Agent.

Scores the whole document with the trained forgery classifier and raises a
single explainable Finding when the document scores as likely-tampered. Adds
evidence only; never auto-rejects.

Fail-safe: missing model, unreadable file, or scoring error -> no findings.

Abstains (no finding) on documents whose size lies outside what the model was
trained on. Its training documents are single-month statements, so a 12-month
statement is outside its experience; see ``assess``.
"""
from __future__ import annotations

from ..models import AgentReport, Finding, IngestionResult, Severity
from ..ml import statement_features as sf

AGENT = "Node6b:MLStatement"

# Fallback cutoffs for model files that carry no thresholds of their own.
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


def assess(feats: dict, model) -> dict:
    """Score one document's features with ``model`` (a bundle dict or a bare estimator).

    Returns ``{"prob", "abstain", "outside", "medium", "high"}``. ``abstain`` is True
    when a size feature (pages, text objects, money values) lies outside the range of
    the documents the model was trained on: its training documents are single months,
    so it cannot judge much longer or larger statements, and guessing there is what
    made every generated statement look forged. ``outside`` names those features.

    Bundles written by the current trainer carry their own ``feature_names``,
    ``thresholds`` and ``size_ranges``; older bundles fall back to the module
    defaults and have no guard, exactly as they behaved before.
    """
    import pandas as pd

    bundle = model if isinstance(model, dict) else {}
    clf = bundle["model"] if isinstance(model, dict) else model
    names = list(bundle.get("feature_names") or sf.FEATURE_NAMES)
    thresholds = bundle.get("thresholds") or {}
    medium = float(thresholds.get("medium", _MEDIUM_PROB))
    high = float(thresholds.get("high", _HIGH_PROB))

    outside = [
        name for name, (lo, hi) in (bundle.get("size_ranges") or {}).items()
        if name in feats and not (lo <= feats[name] <= hi)
    ]
    if outside:
        return {"prob": None, "abstain": True, "outside": outside, "medium": medium, "high": high}

    X = pd.DataFrame([[feats[name] for name in names]], columns=names)
    prob = float(clf.predict_proba(X)[0][1])
    return {"prob": prob, "abstain": False, "outside": [], "medium": medium, "high": high}


def analyze(ing: IngestionResult, model=None) -> AgentReport:
    findings: list[Finding] = []
    if model is None:
        return AgentReport(agent=AGENT, findings=findings)

    feats = sf.extract_features(ing.source_path)
    if feats is None:
        return AgentReport(agent=AGENT, findings=findings)

    try:
        result = assess(feats, model)
    except Exception:
        return AgentReport(agent=AGENT, findings=findings)

    # Outside the sizes the model was trained on -> stay silent rather than guess.
    if result["abstain"]:
        return AgentReport(agent=AGENT, findings=findings)

    prob = result["prob"]
    if prob >= result["high"]:
        sev, weight = Severity.HIGH, 24
    elif prob >= result["medium"]:
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
