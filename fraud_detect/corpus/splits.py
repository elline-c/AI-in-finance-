"""Train / validation / test splits and the held-out layout (design D7).

A company is hashed (SHA-256 via ``stable_seed``) into a fraction in [0, 1),
so its split does not depend on build order or on the layout of a document.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from fraud_detect.corpus.seeding import stable_seed

SPLITS = ("train", "validation", "test", "test_unseen_layout")
_BASE_SPLITS = SPLITS[:3]
_TOLERANCE = 0.001


def validate_fractions(fractions: Sequence[float]) -> None:
    """Raise ``ValueError`` if fractions are negative or do not sum to 1 (+/-0.001)."""
    values = tuple(fractions)
    if len(values) != 3:
        raise ValueError(f"split fractions must have 3 values, got {len(values)}")
    if any(f < 0 for f in values):
        raise ValueError(f"split fractions must not be negative: {values}")
    total = sum(values)
    if abs(total - 1.0) > _TOLERANCE:
        raise ValueError(f"split fractions must sum to 1 (within {_TOLERANCE}), got {total:.4f}")


def find_gaps(
    tamper_types: Sequence[str],
    counts_by_split: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """For each non-empty split, the sorted tamper types with zero documents."""
    gaps: dict[str, list[str]] = {}
    for split, counts in counts_by_split.items():
        if sum(counts.values()) == 0:
            continue
        gaps[split] = sorted(t for t in tamper_types if counts.get(t, 0) == 0)
    return gaps


def assign_split(
    master_seed: int,
    company_id: str,
    layout: str,
    hold_out_layout: Optional[str],
    fractions: Sequence[float],
) -> str:
    """Return the split of a document; one company always maps to one split."""
    validate_fractions(fractions)
    if hold_out_layout is not None and layout == hold_out_layout:
        return "test_unseen_layout"
    u = stable_seed(master_seed, "split", company_id) / 2**64
    total = float(sum(fractions))
    cumulative = 0.0
    for name, frac in zip(_BASE_SPLITS, fractions):
        cumulative += frac / total
        if u < cumulative:
            return name
    # Rounding fallback: last split with a positive fraction.
    for name, frac in reversed(list(zip(_BASE_SPLITS, fractions))):
        if frac > 0:
            return name
    return _BASE_SPLITS[0]
