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
    ap.add_argument("--cache", default="data/private/statement_features.csv",
                    help="Where to cache extracted features (re-used on reruns)")
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
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
