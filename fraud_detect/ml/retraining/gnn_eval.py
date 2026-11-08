"""GNN lookup accounting and flag combination (no torch).

Scores are read through the existing ``gnn_lookup.score_account``; an absent
account is recorded as "no score" and never given a substitute number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from fraud_detect.ml.gnn.gnn_lookup import score_account
from fraud_detect.ml.retraining.metrics import Rate, rate

GNN_NOT_RETRAINED_LINE = "GNN: not retrained, corpus has no transfer graph"
NO_SCORE = "no score"


@dataclass(frozen=True)
class GnnCoverage:
    total: int
    found: int
    no_score: int
    no_score_uens: Tuple[str, ...]
    scores: Dict[str, Optional[float]]
    share: Optional[float]
    text: str


def lookup_coverage(heldout_uens, store) -> GnnCoverage:
    uens = sorted(set(heldout_uens))
    scores: Dict[str, Optional[float]] = {}
    for u in uens:
        s = score_account(store, u) if store is not None else None
        scores[u] = None if s is None else float(s)
    absent = tuple(u for u in uens if scores[u] is None)
    total, found = len(uens), len(uens) - len(absent)
    if total == 0:
        share, text = None, "not computable (0/0)"
    else:
        share = 100.0 * found / total
        text = f"{share:.1f}% ({found}/{total})"
    return GnnCoverage(total, found, total - found, absent, scores, share, text)


def gnn_flags(scores: Sequence, cutoff: float) -> list:
    return [None if s is None else bool(s >= cutoff) for s in scores]


def combine_flags(gbm_flags: Sequence, gnn_flag_list: Sequence) -> list:
    if len(gbm_flags) != len(gnn_flag_list):
        raise ValueError("flag sequences must have the same length")
    return [bool(a) or bool(b) for a, b in zip(gbm_flags, gnn_flag_list)]


@dataclass(frozen=True)
class ModelRates:
    false_alarm: Rate
    detection: Rate
    status: str


def _rates(flags, genuine, positive, scored) -> ModelRates:
    fa_n = fa_d = det_n = det_d = 0
    for f, g, p, ok in zip(flags, genuine, positive, scored):
        if not ok:
            continue
        if g:
            fa_d += 1
            fa_n += bool(f)
        if p:
            det_d += 1
            det_n += bool(f)
    fa, det = rate(fa_n, fa_d), rate(det_n, det_d)
    status = "ok" if fa.value is not None and det.value is not None else "not computable"
    return ModelRates(fa, det, status)


def compare_models(gbm_flags, gnn_flag_list, is_genuine, is_positive) -> dict:
    n = len(gbm_flags)
    if not (len(gnn_flag_list) == len(is_genuine) == len(is_positive) == n):
        raise ValueError("all per-statement sequences must have the same length")
    all_ok = [True] * n
    scored = [f is not None for f in gnn_flag_list]
    combined = combine_flags(gbm_flags, gnn_flag_list)
    return {
        "gbm": _rates(gbm_flags, is_genuine, is_positive, all_ok),
        "gnn": _rates(gnn_flag_list, is_genuine, is_positive, scored),
        "combined": _rates(combined, is_genuine, is_positive, all_ok),
    }
