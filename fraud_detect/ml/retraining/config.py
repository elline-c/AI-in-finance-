"""Run configuration for the retraining pipeline (Task 1.3).

Standard library only at import time. No ``torch``, ``sklearn`` or ``pdfplumber``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Tuple

#: Master seed reserved for the Awkward_Set spike. Must never collide with a
#: configured corpus seed or the held-out seed.
AWKWARD_MASTER_SEED = 20260701

#: Default root for everything the pipeline writes. Always git-ignored.
DEFAULT_OUTPUT_ROOT = "data/private/ml_retraining"


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """Every threshold from the design table, with its default value."""

    leak_auc: float = 0.98
    forgery_auc_gap: float = 0.20
    style_share_points: int = 10
    style_min_docs: int = 30
    resaved_min_share: int = 15
    type_min_docs: int = 20
    subtle_max_relative_change: int = 5
    min_seeds: int = 5
    indicative_min_docs: int = 30
    skip_max_share: int = 2
    row_exclusion_max_share: int = 1
    implausible_auc: float = 0.99
    guard_condition_points: int = 5
    hard_negative_block_points: int = 10
    target_fa_medium: float = 5.0
    target_fa_high: float = 1.0


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Frozen configuration for a single retraining run.

    All open-decision options are plain fields carrying the design's default,
    so a run is fully reproducible from its canonical JSON.
    """

    thresholds: Thresholds = dataclasses.field(default_factory=Thresholds)
    seeds: Tuple[int, ...] = (20260801, 20260802, 20260803, 20260804, 20260805)
    held_out_seed: int = 20260899
    held_out_layout: str = "two_col_ddmmyy_cr"

    # Open decision 1: Real_Style role in the Switch_Rule.
    real_style_role: str = "required"  # "required" or "veto_only"

    # Open decision 5: pooled vs per-corpus tamper-type minimum / coverage check.
    pooled_type_minimum: bool = True

    # Capability waiver (decision 3): recorded separately in the decision log;
    # this flag only controls whether the pipeline will proceed once a waiver
    # has been recorded for the capability gate.
    capability_waiver_allowed: bool = True

    # Node 6a baseline model path, overridable.
    node6a_baseline_path: str = "models/txn_fraud_clf.joblib"

    # Target false-alarm rates (duplicated here from Thresholds for direct
    # access by the Switch_Rule / cutoff selection code paths).
    target_fa_medium: float = 5.0
    target_fa_high: float = 1.0

    # Source paths, overridable; never read except through sources.py.
    real_style_dir: str = "data/private/ml_retraining/real_style"
    forgery_dir: str = "data/private/ml_retraining/forgery_dataset"

    # Output paths.
    output_root: str = DEFAULT_OUTPUT_ROOT

    def canonical_json(self) -> str:
        """Key-sorted, stable JSON representation of this configuration."""
        return json.dumps(_to_jsonable(self), sort_keys=True, separators=(",", ":"))

    @property
    def run_id(self) -> str:
        """SHA-256 prefix of the canonical JSON, used as a stable run identifier."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return digest[:16]


def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_to_jsonable(v) for v in obj]
    return obj
