"""Node 6a candidate training on the PDF feature set with statement-level cutoffs.

Fits on train rows only; cutoffs come from validation statement scores only.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from fraud_detect.ml import train_transaction
from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.retraining import safe_io
from fraud_detect.ml.retraining.audit import AUDIT_V1, model_input_columns
from fraud_detect.ml.retraining.cutoffs import CutoffRule, choose_cutoffs, operating_curve
from fraud_detect.ml.retraining.features_a import ForbiddenFeatureError

MODEL_LABEL = "model trained on synthetic data"
STATEMENT_LEVEL = "statement_max_row"
NODE6A_LIVE_CUTOFF_NOTE = (
    "Node 6a in the live pipeline uses fixed cutoffs (0.5 and 0.8) on the row score; "
    "the cutoffs stored here are statement-level validation cutoffs and are not used by the live node.")


@dataclass(frozen=True)
class StatementSets:
    genuine: list
    positive: list
    zero_fraud_tampered: list


@dataclass
class TrainedA:
    path: Path
    bundle: dict
    zero_fraud_statements: list


def detection_statements(features, statements) -> StatementSets:
    present = set(zip(features["corpus_key"], features["doc_id"]))
    fraud = {k for k, v in features.groupby(["corpus_key", "doc_id"])["y"].max().items() if v == 1}
    tampered = {}
    for r in statements.itertuples():
        tampered[(r.corpus_key, r.doc_id)] = bool(r.is_tampered)
    genuine, positive, zero = [], [], []
    for k, t in tampered.items():
        if k not in present:
            continue
        if not t:
            genuine.append(k)
        elif k in fraud:
            positive.append(k)
        else:
            zero.append(k)
    return StatementSets(sorted(genuine), sorted(positive), sorted(zero))


def statement_scores(pipeline, features, columns) -> dict:
    p = pipeline.predict_proba(features[list(columns)])[:, 1]
    out: dict = {}
    for k, v in zip(zip(features["corpus_key"], features["doc_id"]), p):
        v = float(v)
        if k not in out or v > out[k]:
            out[k] = v
    return out


def _restricted_pipeline(cat: List[str], num: List[str]) -> Pipeline:
    base = train_transaction.build_pipeline("pdf")
    pre = base.named_steps["pre"]
    enc = {n: t for n, t, _ in pre.transformers}
    pre2 = ColumnTransformer(
        transformers=[("cat", clone(enc["cat"]), cat), ("num", enc["num"], num)],
        remainder="drop")
    return Pipeline([("pre", pre2), ("clf", base.named_steps["clf"])])


def train_candidate(train_features, valid_features, statements, *, out_path, columns=None,
                    audit=AUDIT_V1, rule=None, master_seed=None, held_out_layout=None,
                    is_ignored=None) -> TrainedA:
    rule = rule if rule is not None else CutoffRule(level=STATEMENT_LEVEL)
    if rule.level != STATEMENT_LEVEL:
        raise ValueError(f"Node 6a uses level {STATEMENT_LEVEL!r}, got {rule.level!r}")
    requested = list(columns) if columns is not None else model_input_columns(audit, "6a")
    offenders = [c for c in requested if c not in tf.PDF_FEATURE_COLUMNS]
    if offenders:
        raise ForbiddenFeatureError(offenders)
    feature_columns = [c for c in tf.PDF_FEATURE_COLUMNS if c in set(requested)]
    cat = [c for c in tf.PDF_CATEGORICAL if c in feature_columns]
    num = [c for c in feature_columns if c not in cat]
    out_path = Path(out_path)
    safe_io.ensure_ignored(out_path, is_ignored=is_ignored)

    train = train_features
    if "split" in train.columns:
        train = train[train["split"] == "train"]
    pipe = _restricted_pipeline(cat, num)
    pipe.fit(train[feature_columns], train["y"].astype(int))

    sets = detection_statements(valid_features, statements)
    scores = statement_scores(pipe, valid_features, feature_columns)
    keep = sets.genuine + sets.positive
    gen_set = set(sets.genuine)
    s = [scores[k] for k in keep]
    g = [k in gen_set for k in keep]
    cuts = choose_cutoffs(s, g, rule)
    thresholds = {"medium": cuts.medium, "high": cuts.high}
    curve = [asdict(r) for r in operating_curve(s, g, thresholds)]

    bundle = {
        "pipeline": pipe,
        "feature_set": "pdf",
        "feature_columns": feature_columns,
        "categorical": cat,
        "numeric": num,
        "thresholds": thresholds,
        "cutoff_rule": {"target_fa_medium": rule.target_fa_medium,
                        "target_fa_high": rule.target_fa_high, "level": rule.level},
        "live_cutoff_note": NODE6A_LIVE_CUTOFF_NOTE,
        "zero_fraud_statements": {"count": len(sets.zero_fraud_tampered),
                                  "doc_ids": sorted(k[1] for k in sets.zero_fraud_tampered)},
        "validation_curve": curve,
        "audit_version": audit.version,
        "audit_digest": audit.digest(),
        "master_seed": master_seed,
        "held_out_layout": held_out_layout,
        "model_label": MODEL_LABEL,
        "synthetic_data": True,
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    safe_io.atomic_write(out_path, buf.getvalue(), is_ignored=is_ignored)
    return TrainedA(out_path, bundle, list(sets.zero_fraud_tampered))
