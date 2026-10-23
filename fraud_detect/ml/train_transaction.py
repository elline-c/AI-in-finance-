"""Train the transaction-fraud (AML) classifier.

Reads the synthetic transaction dataset straight from disk (no upload step),
honours the dataset-provided split, drops label-bearing columns, trains a
gradient-boosted tree, and reports PR-AUC + precision/recall @ top-1% on the
held-out test split.

Usage:
    python -m fraud_detect.ml.train_transaction \
        --data ~/Downloads/transaction_fraud_dataset \
        --out models/txn_fraud_clf.joblib
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import transaction_features as tf
from .report import imbalanced_report, print_report


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                  sparse_output=False), tf._CATEGORICAL),
            ("num", "passthrough", tf._NUMERIC + tf._HISTORY),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=300,
        max_depth=None,
        l2_regularization=1.0,
        class_weight="balanced",   # rare-positive task (~1% fraud)
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to transaction_fraud_dataset folder")
    ap.add_argument("--out", default="models/txn_fraud_clf.joblib")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data)
    print(f"Loading transactions + accounts from {data_dir} ...")
    joined = tf.load_joined(data_dir)
    feat = tf.build_features(joined)
    parts = tf.split_frames(feat)

    (X_tr, y_tr) = parts["train"]
    (X_val, y_val) = parts["val"]
    (X_te, y_te) = parts["test"]
    print(f"train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}  "
          f"(train fraud rate {y_tr.mean():.4f})")

    pipe = build_pipeline()
    print("Fitting HistGradientBoostingClassifier ...")
    pipe.fit(X_tr, y_tr)

    def scores(X):
        return pipe.predict_proba(X)[:, 1]

    val_rep = imbalanced_report(y_val, scores(X_val), "VAL")
    test_rep = imbalanced_report(y_te, scores(X_te), "TEST")
    print_report(val_rep)
    print_report(test_rep)
    print("(Dataset baseline for reference: single-row PR-AUC ~0.57; "
          "+account history ~0.84. Stage 2 here is single-row + static KYC + "
          "per-account rolling history.)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_columns": tf.FEATURE_COLUMNS,
            "categorical": tf._CATEGORICAL,
            "numeric": tf._NUMERIC,
            "metrics": {"val": val_rep, "test": test_rep},
            "stage": "single_row_plus_static_kyc_plus_account_history",
        },
        out,
    )
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    main()
