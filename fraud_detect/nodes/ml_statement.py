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


def assess_by_month(segments: list[dict], model) -> dict:
    """Score a long statement one calendar month at a time and keep the highest score.

    ``segments`` comes from ``statement_features.month_segments``. Each month is
    scored with ``assess``, so the size guard still applies to every month: a month
    that is itself too small or too large abstains and is left out. If every month
    abstains the result abstains. The cutoffs are the bundle's ``multi_month_thresholds``
    when present (the maximum of several scores needs higher cutoffs than a single
    score to keep the same false-alarm rate), otherwise its ordinary ``thresholds``.

    Returns the ``assess`` fields plus ``months`` (months in the statement),
    ``scored`` (months actually scored), ``worst`` (label of the highest month) and
    ``worst_feats`` (that month's features).
    """
    bundle = model if isinstance(model, dict) else {}
    results = [(s, assess(s["features"], model)) for s in segments]
    scored = [(s, r) for s, r in results if not r["abstain"]]
    first = results[0][1] if results else {"medium": _MEDIUM_PROB, "high": _HIGH_PROB}
    medium, high = first["medium"], first["high"]
    mm = bundle.get("multi_month_thresholds")
    if mm:
        medium, high = float(mm.get("medium", medium)), float(mm.get("high", high))
    base = {"months": len(segments), "scored": len(scored), "worst": None,
            "worst_feats": None, "medium": medium, "high": high}
    if not scored:
        outside = sorted({n for _, r in results for n in r["outside"]})
        return {**base, "prob": None, "abstain": True, "outside": outside}
    seg, best = max(scored, key=lambda sr: sr[1]["prob"])
    return {**base, "prob": best["prob"], "abstain": False, "outside": [],
            "worst": seg["label"], "worst_feats": seg["features"]}


def assess_document(path: str, feats: dict, model) -> dict:
    """Score the whole document; if it is longer than the single-month documents the
    model was trained on, score it month by month instead (see ``assess_by_month``).

    Month-by-month scoring runs only for a bundle that carries
    ``multi_month_thresholds``, i.e. cutoffs calibrated on genuine multi-month
    statements (``python -m fraud_detect.ml.calibrate_statement_months``). Without them the node keeps
    abstaining on long statements: measured on ``data/corpus_large`` the current
    model scores genuine months as high as tampered ones (no cutoff keeps false
    alarms below 10% and still catches anything), so using the ordinary cutoffs on
    the maximum month would flag almost every genuine long statement.

    A document that fits the trained sizes is scored exactly as before. The result
    always carries ``months`` / ``scored`` / ``worst`` / ``worst_feats`` (``None`` when
    the whole document was scored)."""
    result = assess(feats, model)
    result.update({"months": None, "scored": None, "worst": None, "worst_feats": None})
    if not result["abstain"]:
        return result
    if not (isinstance(model, dict) and model.get("multi_month_thresholds")):
        return result
    segments = sf.month_segments(path)
    if len(segments) < 2:
        return result
    by_month = assess_by_month(segments, model)
    # Nothing scorable per month either: keep the whole-document reason for abstaining.
    return result if by_month["abstain"] else by_month


def analyze(ing: IngestionResult, model=None) -> AgentReport:
    findings: list[Finding] = []
    if model is None:
        return AgentReport(agent=AGENT, findings=findings)

    feats = sf.extract_features(ing.source_path)
    if feats is None:
        return AgentReport(agent=AGENT, findings=findings)

    try:
        result = assess_document(ing.source_path, feats, model)
    except Exception:
        return AgentReport(agent=AGENT, findings=findings)

    # Outside the sizes the model was trained on -> stay silent rather than guess.
    if result["abstain"]:
        return AgentReport(agent=AGENT, findings=findings)

    prob = result["prob"]
    where = ""
    if result["worst"] is not None:
        feats = result["worst_feats"]
        where = (f" The statement is longer than the single-month documents the model was "
                 f"trained on, so it was scored month by month ({result['scored']} of "
                 f"{result['months']} months scorable); this is the highest month, "
                 f"{result['worst']}.")
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
                  f"tampered. Strongest file-derived signals: {_top_signals(feats)}.{where} "
                  f"Evidence for a reviewer; never an automatic rejection."),
    ))
    return AgentReport(agent=AGENT, findings=findings)
