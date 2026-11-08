"""Metrics, seed statistics and honest-reporting helpers for the retraining pipeline.

scikit-learn is imported inside functions only. All figures are synthetic-data
figures, not real-world accuracy.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MIN_DOCS_DEFAULT = 30
MIN_SEEDS_DEFAULT = 5


# ---------------------------------------------------------------- ranking metrics
def top_k_count(n: int, fraction: float = 0.01) -> int:
    """ceil(fraction * n), at least 1. (The old report.py helper rounds instead.)"""
    # round first to avoid float noise such as 0.01 * 100 = 1.0000000000000002
    return max(1, int(math.ceil(round(fraction * n, 9))))


def roc_auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y), np.asarray(s)))


def pr_auc(y, s) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y), np.asarray(s)))


def precision_recall_at_top_k(y, s, k: int) -> Tuple[float, float]:
    y = np.asarray(y)
    s = np.asarray(s)
    if len(s) == 0 or k <= 0:
        return 0.0, 0.0
    k = min(int(k), len(s))
    order = np.argsort(-s, kind="stable")
    tp = int(y[order[:k]].sum())
    total = int(y.sum())
    return tp / k, (tp / total if total else 0.0)


# ---------------------------------------------------------------- rates and flags
@dataclass(frozen=True)
class Rate:
    value: Optional[float]
    numer: int
    denom: int
    text: str


def rate(numer: int, denom: int) -> Rate:
    numer, denom = int(numer), int(denom)
    if denom <= 0:
        return Rate(None, numer, denom, f"not computable ({numer}/{denom})")
    value = numer / denom
    return Rate(value, numer, denom, f"{value * 100:.1f}% ({numer}/{denom})")


def metric_flag(n_pos: int, n_neg: int, min_docs: int = MIN_DOCS_DEFAULT) -> str:
    if n_pos <= 0 or n_neg <= 0:
        return "not computable"
    if n_pos < min_docs or n_neg < min_docs:
        return "indicative only"
    return "ok"


# ---------------------------------------------------------------- per type / method
def _as_keys(k: Any) -> Tuple[str, ...]:
    if k is None:
        return ()
    if isinstance(k, str):
        return (k,) if k else ()
    return tuple(x for x in k if x)


def catch_rate_by(scores, is_tampered, keys: Sequence[Any], cutoff: float) -> Dict[str, Rate]:
    """Catch rate (score >= cutoff) among tampered documents, grouped by key."""
    scores = np.asarray(scores, dtype=float)
    tampered = np.asarray(is_tampered).astype(bool)
    counts: "OrderedDict[str, List[int]]" = OrderedDict()
    for sc, t, k in zip(scores, tampered, keys):
        if not t:
            continue
        for key in _as_keys(k):
            c = counts.setdefault(key, [0, 0])
            c[1] += 1
            if sc >= cutoff:
                c[0] += 1
    return {key: rate(c[0], c[1]) for key, c in counts.items()}


def empty_pairs(rows: Iterable[Dict[str, Any]], types: Sequence[str], splits: Sequence[str]) -> List[Tuple[str, str]]:
    """(type, split) pairs with no tampered document."""
    present = set()
    for r in rows:
        if not r.get("is_tampered"):
            continue
        for t in _as_keys(r.get("tamper_types")):
            present.add((t, r.get("split")))
    return [(t, s) for t in types for s in splits if (t, s) not in present]


def split_negatives(is_genuine, genuine_resaved) -> Tuple[np.ndarray, np.ndarray]:
    g = np.asarray(is_genuine).astype(bool)
    r = np.asarray(genuine_resaved).astype(bool)
    hard = g & r
    ordinary = g & ~r
    return hard, ordinary


# ---------------------------------------------------------------- seeds
@dataclass(frozen=True)
class SeedStats:
    n: int
    mean: float
    min: float
    max: float
    std: Optional[float]


def summarise_seeds(values: Sequence[float]) -> SeedStats:
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("summarise_seeds needs at least one value")
    std = float(np.std(arr, ddof=1)) if n > 1 else None
    return SeedStats(n, float(arr.mean()), float(arr.min()), float(arr.max()), std)


@dataclass(frozen=True)
class RunCompleteness:
    status: str
    failed: Tuple[Any, ...]


def seed_completeness(attempted: Sequence[Any], completed: Sequence[Any], min_seeds: int = MIN_SEEDS_DEFAULT) -> RunCompleteness:
    done = set(completed)
    failed: List[Any] = []
    for s in attempted:
        if s not in done and s not in failed:
            failed.append(s)
    n_done = len({s for s in done if s in set(attempted)}) if attempted else len(done)
    status = "complete" if n_done >= min_seeds else "incomplete"
    return RunCompleteness(status, tuple(failed))


@dataclass(frozen=True)
class SeedDifference:
    per_seed: List[float]
    min: float
    max: float


def seed_difference(candidate: Sequence[float], current: Sequence[float]) -> SeedDifference:
    """Candidate minus Current per seed, with min and max spread."""
    diffs = [float(a) - float(b) for a, b in zip(candidate, current)]
    if not diffs:
        raise ValueError("seed_difference needs at least one seed")
    return SeedDifference(diffs, min(diffs), max(diffs))
