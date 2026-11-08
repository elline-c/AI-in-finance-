"""Per-dataset, per-seed evaluation of the Candidate and the Current model (Task 18.3).

Held-out data arrives only as in-memory ``EvalDataset`` objects, loaded through an opened vault.
Metrics come from ``metrics.py``; scoring goes through the ``baseline`` module. numpy and pandas
are imported inside functions. Figures are synthetic-data figures, not real-world accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fraud_detect.ml.retraining import baseline, comparison_report, metrics, sources
from fraud_detect.ml.retraining.safe_io import atomic_write

DATASET_IDS = tuple(sources.DATASETS)
SOURCES = (sources.SOURCE_FORGERY, sources.SOURCE_GENERATOR, sources.SOURCE_REAL_STYLE,
           sources.SOURCE_AWKWARD)
DIFFICULTY_LINE = "difficulty: not recorded in the corpus manifest (generator 1.0.0)"
FORGERY_ASYMMETRY = (
    "Forgery_Dataset asymmetry: " + comparison_report.FORGERY_NOTE
    + ", so a gap between the Candidate and the Current model there may not be like-for-like.")
_FORGERY_DATASETS = ("forgery_test", "forgery_ood")
_MONTH_DATASETS = ("test", "heldout_seed")


class MixedSourcesError(ValueError):
    """Datasets from different Sources or dataset ids were combined."""


@dataclass(frozen=True)
class EvalDoc:
    doc_id: str
    is_tampered: bool
    features: Optional[dict] = None
    rows: Any = None
    layout: Any = None
    n_months: Any = None
    tamper_types: tuple = ()
    tamper_method: Any = None
    genuine_resaved: bool = False
    file: Any = None
    case_type: Any = None


@dataclass(frozen=True)
class EvalDataset:
    dataset_id: str
    source: str
    docs: tuple

    def __post_init__(self):
        if self.dataset_id not in sources.DATASETS:
            raise ValueError(f"unknown dataset id: {self.dataset_id!r}")
        if self.source != sources.DATASETS[self.dataset_id]:
            raise MixedSourcesError(
                f"dataset {self.dataset_id!r} belongs to {sources.DATASETS[self.dataset_id]!r}, "
                f"not {self.source!r}")


def aggregate(datasets) -> EvalDataset:
    datasets = list(datasets)
    if not datasets:
        raise ValueError("nothing to aggregate")
    first = datasets[0]
    for d in datasets[1:]:
        if d.source != first.source or d.dataset_id != first.dataset_id:
            raise MixedSourcesError("datasets differ in Source or dataset id")
    docs = tuple(doc for d in datasets for doc in d.docs)
    return EvalDataset(first.dataset_id, first.source, docs)


@dataclass(frozen=True)
class UnitOutcome:
    path: Path
    skipped: bool
    result: dict


# ---------------------------------------------------------------- helpers
def _rate(numer, denom) -> dict:
    r = metrics.rate(int(numer), int(denom))
    return {"value": r.value, "numer": r.numer, "denom": r.denom}


def _rates(d) -> dict:
    return {k: _rate(v.numer, v.denom) for k, v in d.items()}


def _cut(c) -> dict:
    return {"medium": float(c.medium), "high": float(c.high)}


def _basis(scores, abstain, docs, cutoffs, dataset_id, n_zero) -> dict:
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    abstain = np.asarray(abstain, dtype=bool)
    tam = np.array([bool(d.is_tampered) for d in docs], dtype=bool)
    gen = ~tam
    med = float(cutoffs.medium)
    flagged = (~abstain) & (scores >= med)
    keep = ~abstain
    n_pos, n_neg = int((tam & keep).sum()), int((gen & keep).sum())
    auc = pr = None
    if n_pos > 0 and n_neg > 0:
        auc = metrics.roc_auc(tam[keep], scores[keep])
        pr = metrics.pr_auc(tam[keep], scores[keep])
    if dataset_id == "awkward":
        hard = gen.copy()
    else:
        hard = gen & np.array([bool(d.genuine_resaved) for d in docs], dtype=bool)
    ordinary = gen & ~hard
    n_abs = int(abstain.sum())

    def fa(mask):
        return _rate(int((flagged & mask).sum()), int(mask.sum()))

    guarded = np.where(abstain, -np.inf, scores)  # abstained documents are never caught
    fa_months = None
    if dataset_id in _MONTH_DATASETS:
        fa_months = {str(m): fa(gen & np.array([d.n_months == m for d in docs], dtype=bool))
                     for m in (12, 6)}
    return {
        "cutoffs": _cut(cutoffs),
        "n_pos": n_pos, "n_neg": n_neg, "n_abstained": n_abs,
        "abstained_share": 100.0 * n_abs / len(docs) if len(docs) else 0.0,
        "n_zero_fraud_tampered": int(n_zero),
        "roc_auc": auc, "pr_auc": pr,
        "flag": metrics.metric_flag(n_pos, n_neg),
        "catch_rate": _rate(int((flagged & tam).sum()), int(tam.sum())),
        "false_alarm_rate": fa(gen),
        "hard_negative_fa": fa(hard),
        "ordinary_fa": fa(ordinary),
        "by_type": _rates(metrics.catch_rate_by(guarded, tam, [d.tamper_types for d in docs], med)),
        "by_method": _rates(metrics.catch_rate_by(guarded, tam, [d.tamper_method for d in docs], med)),
        "fa_by_months": fa_months,
        "_flagged": flagged,
    }


def _score_6b(handle, docs, dataset_id, guard):
    return baseline.score_b(handle, [d.features for d in docs], dataset=dataset_id, guard=guard)


def _score_6a(handle, docs, dataset_id):
    """Statement scores = max row probability. Returns (BaselineScores-like, per-doc scores)."""
    import numpy as np
    import pandas as pd

    rows = pd.concat([d.rows for d in docs], ignore_index=True)
    sc = baseline.score_a(handle, rows, dataset=dataset_id)
    if sc.status != baseline.STATUS_OK:
        return sc, None
    out, pos = [], 0
    for d in docs:
        n = len(d.rows)
        out.append(float(np.max(sc.probs[pos:pos + n])) if n else 0.0)
        pos += n
    return sc, np.array(out, dtype=float)


def _strip(basis: dict) -> dict:
    return {k: v for k, v in basis.items() if not k.startswith("_")}


# ---------------------------------------------------------------- evaluation
def evaluate_dataset(node, dataset, *, candidate, current, cutoffs, matched=None, seed,
                     size_guard=True) -> dict:
    import numpy as np

    ds_id = dataset.dataset_id
    all_docs = list(dataset.docs)
    n_zero = 0
    docs = all_docs
    if node == "6a":
        def zero(d):
            return bool(d.is_tampered) and not bool((d.rows["y"] == 1).any())
        n_zero = sum(1 for d in all_docs if zero(d))
        docs = [d for d in all_docs if not zero(d)]
    n = len(docs)
    cand_handle = baseline.wrap(node, candidate)
    no_abs = np.zeros(n, dtype=bool)

    if node == "6b":
        cs = _score_6b(cand_handle, docs, ds_id, size_guard)
        cand_scores, cand_abs = cs.probs, cs.abstain
    else:
        cs, cand_scores = _score_6a(cand_handle, docs, ds_id)
        cand_abs = no_abs
    if cs.status != baseline.STATUS_OK:
        raise RuntimeError(cs.detail or f"candidate could not be scored for Node {node} on {ds_id}")
    cand_basis = _basis(cand_scores, cand_abs, docs, cutoffs, ds_id, n_zero)
    cand = {"matched": cand_basis, "deployed": cand_basis}

    state, detail, cur, diff = current.status, None, None, None
    cur_basis = None
    if current.status == baseline.STATUS_OK:
        if node == "6b":
            m_sc = _score_6b(current, docs, ds_id, False)
            d_sc = _score_6b(current, docs, ds_id, True)
            m_scores, m_abs = m_sc.probs, m_sc.abstain
            d_scores, d_abs = d_sc.probs, d_sc.abstain
            failed = m_sc if m_sc.status != baseline.STATUS_OK else d_sc
        else:
            m_sc, m_scores = _score_6a(current, docs, ds_id)
            d_sc, d_scores = m_sc, m_scores
            m_abs = d_abs = no_abs
            failed = m_sc
        if failed.status != baseline.STATUS_OK:
            state, detail = baseline.BASELINE_FAILED, failed.detail
        else:
            state = baseline.STATUS_OK
            if matched is None:
                raise ValueError("matched cutoffs are required when the Current model is ok")
            dm, dh = baseline.deployed_cutoffs(current)
            deployed = type("Cut", (), {"medium": dm, "high": dh})()
            cur_basis = _basis(m_scores, m_abs, docs, matched, ds_id, n_zero)
            cur = {"matched": cur_basis,
                   "deployed": _basis(d_scores, d_abs, docs, deployed, ds_id, n_zero)}
            diff = {"matched": {
                "roc_auc": _delta(cand_basis["roc_auc"], cur_basis["roc_auc"]),
                "catch_rate": _delta(cand_basis["catch_rate"]["value"], cur_basis["catch_rate"]["value"]),
                "false_alarm_rate": _delta(cand_basis["false_alarm_rate"]["value"],
                                           cur_basis["false_alarm_rate"]["value"])}}
    elif current.status != baseline.NO_BASELINE:
        state = baseline.BASELINE_FAILED
        detail = (f"baseline failed for Node {node} on dataset {ds_id}: model file cannot be loaded")
    else:
        state = baseline.NO_BASELINE

    notes = [DIFFICULTY_LINE]
    if ds_id in _FORGERY_DATASETS:
        notes.append(FORGERY_ASYMMETRY)

    per_file = None
    if ds_id == "real_style":
        per_file = {}
        for i, d in enumerate(docs):
            per_file[d.file] = {
                "label": 1 if d.is_tampered else 0,
                "candidate_flag": bool(cand_basis["_flagged"][i]),
                "current_flag": bool(cur_basis["_flagged"][i]) if cur_basis else None}

    result = {
        "node": node, "dataset": ds_id, "source": dataset.source, "seed": seed,
        "doc_ids": [d.doc_id for d in docs],
        "current_state": state, "current_detail": detail,
        "candidate": {k: _strip(v) for k, v in cand.items()},
        "current": {k: _strip(v) for k, v in cur.items()} if cur else None,
        "difference": diff, "notes": notes, "per_file": per_file,
    }
    return json.loads(json.dumps(result, default=float))


def _delta(a, b):
    return None if a is None or b is None else float(a) - float(b)


# ---------------------------------------------------------------- units
def _unit_path(results_root, node, dataset_id, seed) -> Path:
    return Path(results_root) / node / dataset_id / f"{seed}.json"


def run_unit(vault, node, dataset_id, seed, *, candidate, current, matched, results_root,
             monitor=None, is_ignored=None, size_guard=True) -> UnitOutcome:
    path = _unit_path(results_root, node, dataset_id, seed)
    if path.is_file():
        return UnitOutcome(path, True, json.loads(path.read_text(encoding="utf-8")))
    dataset = vault.load(dataset_id)
    result = evaluate_dataset(node, dataset, candidate=candidate, current=current,
                              cutoffs=vault.cutoffs(node), matched=matched, seed=seed,
                              size_guard=size_guard)
    cand = result["candidate"]["matched"]
    if monitor is not None and cand["roc_auc"] is not None:
        from fraud_detect.ml.retraining.monitor import UNSEEN_LAYOUT_SPLIT, Measurement
        monitor.observe(Measurement(
            name="roc_auc", value=cand["roc_auc"], source=dataset.source,
            split=UNSEEN_LAYOUT_SPLIT if dataset_id == "heldout_layout" else dataset_id,
            master_seed=seed, n_pos=cand["n_pos"], n_neg=cand["n_neg"], model="candidate",
            roc_auc=cand["roc_auc"]))
    atomic_write(path, json.dumps(result, indent=2, sort_keys=True), is_ignored=is_ignored)
    vault.record_score(f"{node}/{dataset_id}/{seed}", {
        "roc_auc": cand["roc_auc"], "catch_rate": cand["catch_rate"]["value"],
        "false_alarm_rate": cand["false_alarm_rate"]["value"]})
    return UnitOutcome(path, False, result)


def run_units(vault, units, **kw) -> list:
    return [run_unit(vault, node, ds, seed, **kw) for node, ds, seed in units]
