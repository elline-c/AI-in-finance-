"""Size_Guard analysis for Node 6b (task 19.2).

Pure functions over in-memory tables. This module never touches a model file;
`switch` applies the chosen guard (Requirement 12.7).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from fraud_detect.ml.statement_features import SIZE_FEATURES
from fraud_detect.ml.train_statement import GUARD_QUANTILES

__all__ = [
    "SIZE_FEATURES", "GUARD_QUANTILES", "OutsideCount", "GuardOption", "GuardEvidence",
    "GuardRecommendation", "candidate_ranges", "outside_mask", "outside_summary",
    "apply_guard", "abstained_share", "recommend_guard", "require_size_guard_choice",
]

_REMOVE_POINTS = 5.0


@dataclass(frozen=True)
class OutsideCount:
    count: int
    total: int
    percent: Optional[float]


class GuardOption(Enum):
    KEEP_CURRENT_GUARD = "current"
    CANDIDATE_GUARD = "candidate"
    REMOVE_GUARD = "off"


@dataclass(frozen=True)
class GuardEvidence:
    cand_fa_12_month: float
    cand_fa_6_month: float
    fa_on: Mapping[str, float]
    fa_off: Mapping[str, float]
    sources_with_30_genuine: list
    cand_outside_share: float
    cur_outside_share: float
    outside_by_split: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class GuardRecommendation:
    option: GuardOption
    figures: dict
    condition_failure: Optional[str]


def candidate_ranges(train_sizes, quantiles=GUARD_QUANTILES) -> dict:
    """0.01 / 0.99 quantiles of the training documents' size features."""
    lo, hi = quantiles
    return {f: [float(train_sizes[f].quantile(lo)), float(train_sizes[f].quantile(hi))]
            for f in SIZE_FEATURES}


def outside_mask(sizes, ranges) -> np.ndarray:
    """True when any size feature is below lo or above hi (bounds inclusive)."""
    mask = np.zeros(len(sizes), dtype=bool)
    for f in SIZE_FEATURES:
        lo, hi = ranges[f]
        v = np.asarray(sizes[f], dtype=float)
        mask |= (v < lo) | (v > hi)
    return mask


def outside_summary(sizes_by_split, ranges) -> dict:
    out = {}
    for name, df in sizes_by_split.items():
        total = len(df)
        count = int(outside_mask(df, ranges).sum()) if total else 0
        out[name] = OutsideCount(count, total, 100.0 * count / total if total else None)
    return out


def apply_guard(flagged, outside) -> np.ndarray:
    return np.asarray(flagged, dtype=bool) & ~np.asarray(outside, dtype=bool)


def abstained_share(outside) -> float:
    return float(100.0 * np.asarray(outside, dtype=bool).mean())


def recommend_guard(ev: GuardEvidence) -> GuardRecommendation:
    fa12, fa6 = ev.cand_fa_12_month, ev.cand_fa_6_month
    gap_ok = fa12 - fa6 <= _REMOVE_POINTS
    off_ok = all(ev.fa_off[s] <= ev.fa_on[s] + _REMOVE_POINTS
                 for s in ev.sources_with_30_genuine)
    figures = asdict(ev)

    failure = None
    if not gap_ok:
        failure = (f"Candidate 12-month genuine false-alarm rate {fa12:.1f}% is more than "
                   f"{_REMOVE_POINTS:.0f} points above its 6-month rate {fa6:.1f}%")
    elif not off_ok:
        worse = [s for s in ev.sources_with_30_genuine
                 if ev.fa_off[s] > ev.fa_on[s] + _REMOVE_POINTS]
        failure = (f"guard-off false-alarm rate is more than {_REMOVE_POINTS:.0f} points above "
                   f"guard-on for: {', '.join(worse)} (12-month {fa12:.1f}%, 6-month {fa6:.1f}%)")

    if gap_ok and off_ok:
        option = GuardOption.REMOVE_GUARD
    elif ev.cand_outside_share < ev.cur_outside_share:
        option = GuardOption.CANDIDATE_GUARD
    else:
        option = GuardOption.KEEP_CURRENT_GUARD
    return GuardRecommendation(option, figures, failure)


def require_size_guard_choice(choice) -> str:
    valid = {o.value for o in GuardOption}
    if not isinstance(choice, str) or choice not in valid:
        raise ValueError(
            f"--size-guard must be given explicitly as one of {sorted(valid)}; got {choice!r}")
    return choice
