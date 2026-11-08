"""Current-model (baseline) loading and scoring through the live node helpers.

Read-only: nothing is written here. Node 6b is scored with ``ml_statement.assess``; Node 6a is
fed exactly the columns it was trained on, chosen by ``ml_transaction._feature_spec``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fraud_detect.ml.retraining.cutoffs import Cutoffs, CutoffRule, choose_cutoffs
from fraud_detect.nodes import ml_statement, ml_transaction

STATUS_OK = "ok"
NO_BASELINE = "no baseline"
BASELINE_FAILED = "baseline failed"
UNLOADABLE = "unloadable"

_DEFAULT_CUTOFFS = (0.5, 0.8)
# Defaults ml_transaction._rows_from_ingestion uses for fields a statement does not carry.
_LIVE_DEFAULTS = {
    "channel": "NA", "counterparty_type": "NA", "counterparty_country": "NA",
    "merchant_category": "NA", "open_date": None, "segment": "NA", "age_band": "NA",
    "business_type": "NA", "kyc_risk_rating": "NA",
    "declared_monthly_income_sgd": 0.0, "declared_monthly_turnover_sgd": 0.0,
}


@dataclass(frozen=True)
class ModelHandle:
    node: str
    path: Path | None
    status: str
    bundle: object | None
    sha256: str | None


@dataclass(frozen=True)
class BaselineScores:
    node: str
    dataset: str
    status: str
    detail: str | None
    probs: np.ndarray | None
    abstain: np.ndarray | None
    sha256: str | None


def load_current(node, path) -> ModelHandle:
    p = Path(path) if path is not None else None
    try:
        if p is None or not p.is_file():
            return ModelHandle(node, p, NO_BASELINE, None, None)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        loader = ml_transaction.load_model if node == "6a" else ml_statement.load_model
        bundle = loader(str(p))
        if bundle is None:
            return ModelHandle(node, p, UNLOADABLE, None, digest)
        return ModelHandle(node, p, STATUS_OK, bundle, digest)
    except Exception:
        return ModelHandle(node, p, UNLOADABLE, None, None)


def wrap(node, bundle, path=None) -> ModelHandle:
    return ModelHandle(node, Path(path) if path is not None else None, STATUS_OK, bundle, None)


def with_guard(bundle, guard: bool) -> dict:
    if guard:
        return bundle
    return {k: v for k, v in dict(bundle).items() if k != "size_ranges"}


def _not_scored(handle, dataset, node) -> BaselineScores | None:
    sha = handle.sha256
    if handle.status == NO_BASELINE:
        return BaselineScores(node, dataset, NO_BASELINE, None, None, None, sha)
    if handle.status != STATUS_OK or handle.bundle is None:
        return _failed(node, dataset, "model file cannot be loaded", sha)
    return None


def _failed(node, dataset, why, sha=None) -> BaselineScores:
    detail = f"baseline failed for Node {node} on dataset {dataset}: {why}"
    return BaselineScores(node, dataset, BASELINE_FAILED, detail, None, None, sha)


def score_b(handle, feats, *, dataset, guard=True) -> BaselineScores:
    early = _not_scored(handle, dataset, "6b")
    if early is not None:
        return early
    try:
        bundle = with_guard(handle.bundle, guard)
        results = [ml_statement.assess(f, bundle) for f in feats]
        abstain = np.array([bool(r["abstain"]) for r in results], dtype=bool)
        probs = np.array([np.nan if r["abstain"] else float(r["prob"]) for r in results],
                         dtype=float)
        return BaselineScores("6b", dataset, STATUS_OK, None, probs, abstain, handle.sha256)
    except Exception as exc:  # any scoring error becomes a reported state
        return _failed("6b", dataset, f"{type(exc).__name__}: {exc}", handle.sha256)


def score_a(handle, rows, *, dataset) -> BaselineScores:
    early = _not_scored(handle, dataset, "6a")
    if early is not None:
        return early
    try:
        model = handle.bundle
        build, columns = ml_transaction._feature_spec(model)
        frame = rows.assign(is_fraud=0, split="train")  # same dummy columns as the live node
        for col, default in _LIVE_DEFAULTS.items():  # what the live node fills for absent fields
            if col not in frame.columns:
                frame[col] = default
        feat = build(frame)
        pipe = model["pipeline"] if isinstance(model, dict) else model
        probs = np.asarray(pipe.predict_proba(feat[columns])[:, 1], dtype=float)
        if len(probs) != len(rows):
            return _failed("6a", dataset, "row count changed while building features", handle.sha256)
        return BaselineScores("6a", dataset, STATUS_OK, None, probs,
                              np.zeros(len(probs), dtype=bool), handle.sha256)
    except Exception as exc:
        return _failed("6a", dataset, f"{type(exc).__name__}: {exc}", handle.sha256)


def deployed_cutoffs(handle) -> tuple[float, float]:
    """(medium, high) the live node runs with; Node 6a ignores any stored thresholds (F9)."""
    if handle.node == "6b" and isinstance(handle.bundle, dict):
        t = handle.bundle.get("thresholds") or {}
        return (float(t.get("medium", _DEFAULT_CUTOFFS[0])),
                float(t.get("high", _DEFAULT_CUTOFFS[1])))
    return _DEFAULT_CUTOFFS


def matched_cutoffs(scores, is_genuine, rule: CutoffRule) -> Cutoffs | None:
    if scores is None or scores.status != STATUS_OK:
        return None
    return choose_cutoffs(scores.probs, is_genuine, rule)
