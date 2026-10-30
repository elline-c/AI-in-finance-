"""Train the statement-forgery classifier.

Reads labels.csv, extracts file-derived features from each PDF (via
statement_features), trains on the dataset-provided ``train`` split, tunes
nothing automatically (val is reported), and — crucially — reports the
IN-LAYOUT test score and the OUT-OF-DISTRIBUTION layout test score SEPARATELY.

Per the data card: if test_ood_layout is far below test_in_layout, the model has
learned the bank templates rather than the tampering. Always read both numbers.

Usage:
    python -m fraud_detect.ml.train_statement \
        --data ~/Downloads/statement_forgery_dataset \
        --out models/statement_forgery_clf.joblib
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import statement_features as sf
from .report import binary_report, print_report

TARGET = "is_tampered"

# Severity cutoffs for the live node, chosen on the VALIDATION split (the data card
# says to tune thresholds there). The model's scores are bimodal, so the old 0.5/0.8
# cutoffs sat in a region where it is simply unsure. On validation, 0.7 flags 7% of
# genuine documents while catching 84% of forgeries, and 0.9 flags 4% / 75%. The
# curve is printed on every training run so these can be re-checked after retraining.
DEFAULT_THRESHOLDS = {"medium": 0.7, "high": 0.9}

# Documents whose size features fall outside this central range of the training
# documents are ones the model never saw; Node 6b abstains on them.
GUARD_QUANTILES = (0.01, 0.99)


def _build_feature_table(data_dir: str, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(labels)
    for i, rec in enumerate(labels.itertuples(index=False), start=1):
        pdf_path = os.path.join(data_dir, rec.file)
        feats = sf.extract_features(pdf_path)
        if feats is None:
            print(f"  skip unreadable {rec.file}")
            continue
        feats["doc_id"] = rec.doc_id
        feats["split"] = rec.split
        feats[TARGET] = int(rec.is_tampered)
        rows.append(feats)
        if i % 250 == 0:
            print(f"  extracted {i}/{n} ...")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to statement_forgery_dataset folder")
    ap.add_argument("--out", default="models/statement_forgery_clf.joblib")
    ap.add_argument("--cache", default="data/private/statement_features_v2.csv",
                    help="Where to cache extracted features (re-used on reruns). The "
                         "default is a v2 file: the original statement_features.csv holds "
                         "the earlier, wrong balance-reconciliation feature and must not "
                         "be reused.")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data)
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    cache = Path(args.cache)
    if cache.exists():
        print(f"Loading cached features from {cache}")
        table = pd.read_csv(cache)
    else:
        print(f"Extracting features from {len(labels)} PDFs (one-time) ...")
        table = _build_feature_table(data_dir, labels)
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
        print(f"Cached features -> {cache}")

    def xy(split_names):
        sub = table[table["split"].isin(split_names)]
        return sub[sf.FEATURE_NAMES], sub[TARGET].astype(int)

    X_tr, y_tr = xy(["train"])
    X_val, y_val = xy(["val"])
    X_in, y_in = xy(["test_in_layout"])
    X_ood, y_ood = xy(["test_ood_layout"])
    print(f"train={len(X_tr)} val={len(X_val)} "
          f"test_in_layout={len(X_in)} test_ood_layout={len(X_ood)}")

    clf = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, l2_regularization=1.0, random_state=42,
    )
    print("Fitting HistGradientBoostingClassifier ...")
    clf.fit(X_tr, y_tr)

    def rep(X, y, label):
        if len(X) == 0:
            return None
        scores = clf.predict_proba(X)[:, 1]
        preds = (scores >= 0.5).astype(int)
        r = binary_report(y, preds, scores, label)
        print_report(r)
        return r

    r_val = rep(X_val, y_val, "VAL")
    r_in = rep(X_in, y_in, "TEST_IN_LAYOUT")
    r_ood = rep(X_ood, y_ood, "TEST_OOD_LAYOUT (held-out Vellichor Bank)")

    if r_in and r_ood:
        gap = r_in["roc_auc"] - r_ood["roc_auc"]
        print(f"\nGeneralisation gap (in-layout ROC-AUC - OOD ROC-AUC): {gap:.3f}")
        if gap > 0.15:
            print("  WARNING: large gap -> model may be learning templates, "
                  "not tampering. Treat OOD score as the honest number.")

    # Validation operating curve: shows where the live thresholds come from.
    if len(X_val):
        pv = clf.predict_proba(X_val)[:, 1]
        yv = y_val.values
        print("\nVALIDATION operating curve (genuine flagged / forged caught):")
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            print(f"    >= {t:<4}  {(pv[yv == 0] >= t).mean():>4.0%} / {(pv[yv == 1] >= t).mean():>4.0%}")

    lo, hi = GUARD_QUANTILES
    size_ranges = {
        f: [float(X_tr[f].quantile(lo)), float(X_tr[f].quantile(hi))] for f in sf.SIZE_FEATURES
    }
    print(f"\nGuard ranges (training p{lo:.0%}-p{hi:.0%}): {size_ranges}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "thresholds": DEFAULT_THRESHOLDS,
            "size_ranges": size_ranges,
            "feature_names": sf.FEATURE_NAMES,
            "metrics": {
                "val": r_val, "test_in_layout": r_in, "test_ood_layout": r_ood,
            },
        },
        out,
    )
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    main()
