"""Node 6b candidate training (task 15.2).

Fits the audited (and the informational all-11) statement-forgery candidates on the ``train``
Split only, chooses cutoffs on the ``validation`` Split only, and writes each candidate through
``safe_io.atomic_write``. Nothing under the real ``models/`` folder is touched: callers pass an
``out_path`` / ``out_dir`` that is git-ignored. Heavy dependencies are imported inside functions.
"""
from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import gates as _gates
from .audit import AUDIT_V1, FeatureAudit, model_input_columns
from .cutoffs import CutoffRule, choose_cutoffs, format_operating_curve, operating_curve
from .safe_io import atomic_write

MODEL_LABEL = "model trained on synthetic data"
GNN_LINE = "GNN: not retrained, corpus has no transfer graph"
ABLATION_NOTE = "not used by the Switch_Rule"

_VARIANTS = ("audited", "all11")
_SIZE_GUARD = {"mode": "candidate_quantiles", "quantiles": [0.01, 0.99]}


class TrainRefused(Exception):
    """A gate or Leak_Test suspect is still open; nothing was fitted or written."""

    def __init__(self, open_items, exit_code: int):
        self.open_items = tuple(open_items)
        self.exit_code = exit_code
        super().__init__("training refused, open items: " + ", ".join(self.open_items))


@dataclasses.dataclass
class CandidateResult:
    path: Path
    bundle: dict
    failed: bool = False
    marker_path: Optional[Path] = None


@dataclasses.dataclass
class TrainOutputs:
    per_seed: dict
    ablation: dict
    release: CandidateResult
    release_ablation: CandidateResult
    report_lines: list


def _check_open(gate_results, leak) -> None:
    open_items: list = []
    integrity = False
    for g in gate_results or ():
        if not g.passed:
            open_items.append(g.gate_id)
            if not g.waivable:
                integrity = True
    if leak is not None and leak.blocked:
        names = [s.name for s in leak.suspects] or ["leak-test"]
        open_items.extend(names)
    if open_items:
        raise TrainRefused(open_items, _gates.EXIT_INTEGRITY if integrity else _gates.EXIT_WAITING)


def _check_splits(train, valid) -> None:
    if len(train) == 0 or not (train["split"] == "train").all():
        raise ValueError("every training row must have split == 'train'")
    if len(valid) == 0 or not (valid["split"] == "validation").all():
        raise ValueError("every validation row must have split == 'validation'")


def _build_pipeline(names, audit: FeatureAudit, variant: str, n_train: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    from .neutralise import make_form

    neutral = {}
    if variant == "audited":
        neutral = {e.name: e.neutralised_form for e in audit.entries
                   if e.section == "model_feature" and e.node == "6b"
                   and e.decision == "neutralise" and e.name in names}
    plain = [n for n in names if n not in neutral]
    transformers = [("plain", "passthrough", plain)]
    for n, form in neutral.items():
        transformers.append((f"neutral_{n}", make_form(form), [n]))
    prep = ColumnTransformer(transformers, remainder="drop")
    # Same settings as train_statement.py.
    # The library default min_samples_leaf is 20, which equals the production behaviour for any
    # realistic corpus (n_train >= 80). On tiny fixtures it would forbid every split and give a
    # constant model, so the leaf size is capped at a quarter of the training rows there.
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, l2_regularization=1.0, random_state=42,
        min_samples_leaf=min(20, max(1, n_train // 4)))
    return Pipeline([("prep", prep), ("clf", clf)])


def train_candidate(train, valid, *, master_seed, held_out_layout, out_path, audit=AUDIT_V1,
                    rule=None, variant="audited", gate_results=(), leak=None,
                    is_ignored=None, verify: Optional[Callable] = None,
                    _pooled_seeds=None) -> CandidateResult:
    if rule is None:
        rule = CutoffRule()
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
    _check_open(gate_results, leak)
    _check_splits(train, valid)

    import joblib
    import numpy as np
    import threadpoolctl

    from fraud_detect.ml import statement_features as sf
    from fraud_detect.ml.train_statement import GUARD_QUANTILES

    names = list(sf.FEATURE_NAMES) if variant == "all11" else model_input_columns(audit, "6b")
    X_tr = train[names].astype(float)
    y_tr = train["is_tampered"].astype(int)
    pipe = _build_pipeline(names, audit, variant, len(train))
    with threadpoolctl.threadpool_limits(1):
        pipe.fit(X_tr, y_tr)

    scores = pipe.predict_proba(valid[names].astype(float))[:, 1]
    is_genuine = ~valid["is_tampered"].astype(bool).to_numpy()
    cut = choose_cutoffs(scores, is_genuine, rule)
    thresholds = {"medium": cut.medium, "high": cut.high}
    curve_cuts = {"medium": cut.medium, "high": cut.high}
    curve_cuts.update({f"{g}": g for g in (0.5, 0.7, 0.8, 0.9, 0.95)})
    curve = [dataclasses.asdict(r) for r in operating_curve(scores, is_genuine, curve_cuts)]

    lo, hi = GUARD_QUANTILES
    size_ranges = {f: [float(train[f].quantile(lo)), float(train[f].quantile(hi))]
                   for f in sf.SIZE_FEATURES}

    bundle: dict[str, Any] = {
        "model": pipe,
        "thresholds": thresholds,
        "size_ranges": size_ranges,
        "feature_names": names,
        "audit_version": audit.version,
        "audit_digest": audit.digest(),
        "master_seed": master_seed,
        "held_out_layout": held_out_layout,
        "size_guard": dict(_SIZE_GUARD),
        "cutoff_rule": dataclasses.asdict(rule),
        "validation_curve": curve,
        "model_label": MODEL_LABEL,
        "synthetic_data": True,
        "n_train": int(len(train)),
    }
    if _pooled_seeds is not None:
        bundle["pooled_seeds"] = sorted(_pooled_seeds)
    if variant == "all11":
        bundle["informational"] = True
        bundle["note"] = ABLATION_NOTE

    path = Path(out_path)
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    atomic_write(path, buf.getvalue(), is_ignored=is_ignored)

    row = train.iloc[0]
    feats = {c: float(row[c]) for c in sf.FEATURE_NAMES}
    failed = False
    marker: Optional[Path] = None
    try:
        if verify is not None:
            verify(bundle, feats)
        else:
            from fraud_detect.nodes import ml_statement
            ml_statement.assess(feats, joblib.load(path))
    except Exception as exc:  # the file stays; the marker says it cannot be used
        failed = True
        marker = Path(str(path) + ".failed")
        atomic_write(marker, f"candidate failed to load or score: {exc}\n", is_ignored=is_ignored)
    return CandidateResult(path=path, bundle=bundle, failed=failed, marker_path=marker)


def _curve_text(res: CandidateResult, valid) -> str:
    b = res.bundle
    scores = b["model"].predict_proba(valid[b["feature_names"]].astype(float))[:, 1]
    is_genuine = ~valid["is_tampered"].astype(bool).to_numpy()
    return format_operating_curve(scores, is_genuine, b["thresholds"])


def train_run(tables: Mapping, *, held_out_layout, out_dir, audit=AUDIT_V1, rule=None,
              gate_results=(), leak=None, is_ignored=None, verify=None) -> TrainOutputs:
    import pandas as pd

    if rule is None:
        rule = CutoffRule()
    _check_open(gate_results, leak)
    out_dir = Path(out_dir)
    seeds = sorted(tables)
    common = dict(held_out_layout=held_out_layout, audit=audit, rule=rule,
                  gate_results=gate_results, leak=leak, is_ignored=is_ignored, verify=verify)
    lines: list = [GNN_LINE]
    print(GNN_LINE)

    def emit(label, res, valid):
        header = f"{label} (validation operating curve)"
        text = _curve_text(res, valid)
        print(header)
        print(text)
        lines.append(header)
        lines.extend(text.splitlines())

    per_seed: dict = {}
    ablation: dict = {}
    for s in seeds:
        t = tables[s]
        tr, va = t[t["split"] == "train"], t[t["split"] == "validation"]
        base = out_dir / "models" / "6b"
        per_seed[s] = train_candidate(tr, va, master_seed=s, out_path=base / f"seed-{s}.joblib",
                                      variant="audited", **common)
        ablation[s] = train_candidate(tr, va, master_seed=s, out_path=base / f"seed-{s}-all11.joblib",
                                      variant="all11", **common)
        emit(f"seed {s} audited", per_seed[s], va)
        emit(f"seed {s} all11", ablation[s], va)

    pooled = pd.concat([tables[s] for s in seeds], ignore_index=True)
    ptr, pva = pooled[pooled["split"] == "train"], pooled[pooled["split"] == "validation"]
    release = train_candidate(ptr, pva, master_seed=None, out_path=out_dir / "release" / "6b.joblib",
                              variant="audited", _pooled_seeds=seeds, **common)
    release_ab = train_candidate(ptr, pva, master_seed=None,
                                 out_path=out_dir / "release" / "6b-all11.joblib",
                                 variant="all11", _pooled_seeds=seeds, **common)
    emit("release audited", release, pva)
    emit("release all11", release_ab, pva)
    return TrainOutputs(per_seed, ablation, release, release_ab, lines)
