"""Retrain the Node 6b statement model with genuine months from a multi-month corpus.

The current model was trained on single-month statements from one dataset and
scores genuine months of the multi-month corpus (``data/corpus_large``) about as
high as tampered ones. This trainer keeps the original training rows and adds every
calendar month of the corpus's GENUINE documents, cut out with
``statement_features.month_segments``, labelled genuine. With ``--with-tampered`` it
also adds each tampered document's edited months (from its manifest), labelled
tampered; without them the model learns that the corpus layouts are always genuine
and ignores real tamper signals there. Only the corpus splits named
in ``--corpus-splits`` (default ``train``) are used; the others stay unseen so that
``calibrate_statement_months`` can test the model on them.

The original dataset's ``val`` / ``test_in_layout`` / ``test_ood_layout`` scores are
printed next to the current model's so a drop on single-month documents is visible.

The output is a NEW file; the live model (``models/statement_forgery_clf.joblib``)
is not touched. Synthetic data: no claim about real-world accuracy.

    python -m fraud_detect.ml.train_statement_months \
        --out models/statement_forgery_clf_months.joblib
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import statement_features as sf
from .train_statement import DEFAULT_THRESHOLDS, GUARD_QUANTILES, TARGET


def corpus_genuine_months(corpus: Path, splits: set[str]) -> pd.DataFrame:
    """One row per calendar month of every genuine multi-month document in ``splits``."""
    return corpus_months(corpus, splits, with_tampered=False)


def _month_key(label: str) -> str:
    """``month_segments`` label -> manifest month (``YYYY-MM``). Day-month layouts
    give ``month MM`` with no year; those are matched on the month number alone."""
    return label[-2:] if label.startswith("month ") else label


def corpus_months(corpus: Path, splits: set[str], *, with_tampered: bool) -> pd.DataFrame:
    """Month rows from the multi-month documents of ``splits``.

    Genuine documents (including genuine re-saves) give every month, labelled 0.
    With ``with_tampered``, a tampered document gives the months its manifest lists
    in ``label.affected_months``, labelled 1; its untouched months are left out, as
    they are neither clearly genuine nor tampered. A tampered document with no
    affected month (a document-level change such as metadata) gives every month,
    labelled 1. Month labels come only from the manifest, never from the features.
    """
    import json

    rows = []
    for rec in csv.DictReader(open(corpus / "labels.csv", newline="")):
        if rec["split"] not in splits:
            continue
        tampered = rec["is_tampered"] == "1"
        if tampered and not with_tampered:
            continue
        affected: set[str] | None = None
        if tampered:
            man = json.load(open(corpus / "manifests" / f"{rec['doc_id']}.json"))
            months = man["label"].get("affected_months") or []
            affected = {m for m in months} | {m[-2:] for m in months} if months else None
        for seg in sf.month_segments(str(corpus / rec["file"])):
            if affected is not None and seg["label"] not in affected \
                    and _month_key(seg["label"]) not in affected:
                continue
            rows.append({**seg["features"], "doc_id": f"{rec['doc_id']}:{seg['label']}",
                         "source_doc": rec["doc_id"], "split": "corpus_" + rec["split"],
                         TARGET: int(tampered)})
    return pd.DataFrame(rows)


def _auc(clf, X, y) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(X) == 0 or y.nunique() < 2:
        return None
    return float(roc_auc_score(y, clf.predict_proba(X[sf.FEATURE_NAMES])[:, 1]))


def main(argv=None) -> int:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-cache", default="data/private/statement_features_v2.csv",
                    help="cached features of the original single-month dataset")
    ap.add_argument("--corpus", default="data/corpus_large")
    ap.add_argument("--corpus-splits", default="train")
    ap.add_argument("--current", default="models/statement_forgery_clf.joblib")
    ap.add_argument("--out", default="models/statement_forgery_clf_months.joblib")
    ap.add_argument("--with-tampered", action="store_true",
                    help="also add the tampered months named in each tampered document's manifest")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.resolve() == Path(args.current).resolve():
        print("Refusing to overwrite the live model; choose a different --out.")
        return 1

    base = pd.read_csv(args.base_cache)
    splits = {s.strip() for s in args.corpus_splits.split(",") if s.strip()}
    extra = corpus_months(Path(args.corpus), splits, with_tampered=args.with_tampered)
    if extra.empty:
        print("No corpus months found; nothing to add.")
        return 1
    n_pos = int(extra[TARGET].sum())
    print(f"original train rows: {int((base.split == 'train').sum())}; added corpus months: "
          f"{len(extra) - n_pos} genuine + {n_pos} tampered "
          f"(from {extra.source_doc.nunique()} documents, splits {sorted(splits)})")
    if args.with_tampered:
        tam_docs = {r["doc_id"] for r in csv.DictReader(open(Path(args.corpus) / "labels.csv"))
                    if r["split"] in splits and r["is_tampered"] == "1"}
        missing = sorted(tam_docs - set(extra.loc[extra[TARGET] == 1, "source_doc"]))
        if missing:
            print(f"  WARNING: {len(missing)} tampered document(s) gave no month "
                  f"(affected month not found by the splitter): {missing[:5]}")

    train = pd.concat([base[base.split == "train"], extra], ignore_index=True)
    X_tr, y_tr = train[sf.FEATURE_NAMES].astype(float), train[TARGET].astype(int)
    # Same settings as train_statement.py. Balanced weights: the added rows are all
    # genuine, so without them the model's base rate would drift towards "genuine".
    w = np.where(y_tr == 1, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()), 1.0)
    clf = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300,
                                         l2_regularization=1.0, random_state=42)
    clf.fit(X_tr, y_tr, sample_weight=w)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current = joblib.load(args.current)
    cur = current["model"] if isinstance(current, dict) else current
    print("\nOriginal single-month dataset, ROC-AUC (current -> retrained):")
    for split in ("val", "test_in_layout", "test_ood_layout"):
        sub = base[base.split == split]
        X, y = sub[sf.FEATURE_NAMES].astype(float), sub[TARGET].astype(int)
        print(f"  {split:<16} {_auc(cur, X, y):.3f} -> {_auc(clf, X, y):.3f}")

    val = base[base.split == "val"]
    pv = clf.predict_proba(val[sf.FEATURE_NAMES].astype(float))[:, 1]
    yv = val[TARGET].to_numpy()
    print("\nSingle-month VALIDATION operating curve (genuine flagged / forged caught):")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        print(f"    >= {t:<4}  {(pv[yv == 0] >= t).mean():>4.0%} / {(pv[yv == 1] >= t).mean():>4.0%}")
    em = clf.predict_proba(extra[sf.FEATURE_NAMES].astype(float))[:, 1]
    ey = extra[TARGET].to_numpy()
    print("\nAdded corpus months (training data, so optimistic):")
    for lab, name in ((0, "genuine"), (1, "tampered")):
        if (ey == lab).any():
            print(f"  {name:<9} median score {np.median(em[ey == lab]):.3f}, "
                  f"share >= 0.7: {(em[ey == lab] >= 0.7).mean():.0%}")

    lo, hi = GUARD_QUANTILES
    size_ranges = {f: [float(X_tr[f].quantile(lo)), float(X_tr[f].quantile(hi))]
                   for f in sf.SIZE_FEATURES}
    print(f"Guard ranges (training p{lo:.0%}-p{hi:.0%}): {size_ranges}")

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": clf,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "size_ranges": size_ranges,
        "feature_names": list(sf.FEATURE_NAMES),
        "corpus_train_splits": sorted(splits),
        "corpus": str(args.corpus),
        "n_train": int(len(train)),
        "n_corpus_genuine_months": int((ey == 0).sum()),
        "n_corpus_tampered_months": int((ey == 1).sum()),
        "with_tampered": bool(args.with_tampered),
        "synthetic_data": True,
    }, out)
    print(f"Saved model -> {out} (live model unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
