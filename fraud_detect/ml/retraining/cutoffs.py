"""Cutoff selection from validation scores only (design "Cutoff selection").

The cutoff is the lowest observed genuine validation score whose genuine
false-alarm rate (share of genuine scores ``>=`` the cutoff) is at or under
the target. Nothing here reads test or held-out data. Numpy only, imported
at module level; no heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

FIXED_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
LEVELS = ("document", "statement_max_row")


class CutoffError(Exception):
    """No cutoff could be chosen. ``kind`` is 'missing class' or 'target not met'."""

    def __init__(self, kind: str, detail: Optional[str] = None,
                 closest_fa: Optional[float] = None, cutoff: Optional[str] = None):
        self.kind = kind
        self.detail = detail
        self.closest_fa = closest_fa
        self.cutoff = cutoff
        msg = kind if detail is None else f"{kind}: {detail}"
        super().__init__(msg)


@dataclass(frozen=True)
class CutoffRule:
    target_fa_medium: float = 5.0   # percent
    target_fa_high: float = 1.0     # percent
    level: str = "document"

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")


@dataclass(frozen=True)
class Cutoffs:
    medium: float
    high: float
    achieved_fa_medium: float
    achieved_fa_high: float


@dataclass(frozen=True)
class CurveRow:
    name: str
    cutoff: float
    alert_rate: float
    false_alarm_rate: float
    catch_rate: float


def _fa(genuine: np.ndarray, cut: float) -> float:
    return 100.0 * float(np.sum(genuine >= cut)) / len(genuine)


def _pick(genuine: np.ndarray, target: float, name: str) -> tuple:
    # A target finer than one genuine document cannot be measured; allow one document.
    limit = max(target, 100.0 / len(genuine))
    for s in np.unique(genuine):  # ascending
        fa = _fa(genuine, s)
        if fa <= limit:
            return float(s), fa
    closest = _fa(genuine, genuine.max())
    raise CutoffError(
        "target not met",
        f"no {name} cutoff reaches a false-alarm rate of {target:.1f}%; "
        f"closest achievable is {closest:.1f}%",
        closest_fa=closest, cutoff=name)


def choose_cutoffs(scores, is_genuine, rule: CutoffRule) -> Cutoffs:
    s = np.asarray(scores, dtype=float)
    g = np.asarray(is_genuine, dtype=bool)
    if s.shape != g.shape:
        raise ValueError("scores and is_genuine must have the same shape")
    if not g.any():
        raise CutoffError("missing class", "no genuine documents in the validation data")
    if g.all():
        raise CutoffError("missing class", "no tampered documents in the validation data")
    genuine = s[g]
    medium, fa_m = _pick(genuine, rule.target_fa_medium, "medium")
    high, fa_h = _pick(genuine, rule.target_fa_high, "high")
    if high < medium:
        high, fa_h = medium, fa_m
    return Cutoffs(medium, high, fa_m, fa_h)


def operating_curve(scores, is_genuine, cutoffs: Mapping[str, float]) -> list:
    s = np.asarray(scores, dtype=float)
    g = np.asarray(is_genuine, dtype=bool)
    rows = []
    for name, cut in cutoffs.items():
        alerts = s >= cut
        rows.append(CurveRow(
            name=str(name), cutoff=float(cut),
            alert_rate=100.0 * float(alerts.mean()) if len(s) else 0.0,
            false_alarm_rate=100.0 * float(alerts[g].mean()) if g.any() else 0.0,
            catch_rate=100.0 * float(alerts[~g].mean()) if (~g).any() else 0.0))
    return rows


def format_operating_curve(scores, is_genuine, cutoffs: Mapping[str, float]) -> str:
    def line(r: CurveRow) -> str:
        return (f"{r.name} cutoff {r.cutoff:.3f}: alert rate {r.alert_rate:.1f}%, "
                f"false-alarm rate {r.false_alarm_rate:.1f}%, catch rate {r.catch_rate:.1f}%")

    lines = [line(r) for r in operating_curve(scores, is_genuine, cutoffs)]
    grid = {f"{g}": g for g in FIXED_GRID}
    lines += [line(r) for r in operating_curve(scores, is_genuine, grid)]
    return "\n".join(lines)


def statement_max_row(statement_ids: Sequence, row_scores) -> dict:
    out: dict = {}
    for sid, sc in zip(statement_ids, np.asarray(row_scores, dtype=float)):
        v = float(sc)
        if sid not in out or v > out[sid]:
            out[sid] = v
    return out
