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


# Which inputs the model is trained on. "pdf" (the default) uses only what a
# parsed PDF statement provides, which is what Node 6a sees at screening time.
# "full" uses every dataset column, including channel/counterparty/KYC fields a
# statement does not carry; it is kept only for offline comparison.
FEATURE_SETS = {
    "pdf": {
        "build": tf.build_pdf_features,
        "categorical": tf.PDF_CATEGORICAL,
        "numeric": tf.PDF_NUMERIC + tf._HISTORY,
    },
    "full": {
        "build": tf.build_features,
        "categorical": tf._CATEGORICAL,
        "numeric": tf._NUMERIC + tf._HISTORY,
    },
}


def build_pipeline(feature_set: str = "pdf") -> Pipeline:
    spec = FEATURE_SETS[feature_set]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                  sparse_output=False), spec["categorical"]),
            ("num", "passthrough", spec["numeric"]),
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
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="pdf",
                    help="pdf = only fields a PDF statement carries (default); "
                         "full = every dataset column, for offline comparison")
    args = ap.parse_args()
    spec = FEATURE_SETS[args.feature_set]
    columns = spec["numeric"] + spec["categorical"]

    data_dir = os.path.expanduser(args.data)
    print(f"Loading transactions + accounts from {data_dir} ...")
    joined = tf.load_joined(data_dir)
    feat = spec["build"](joined)

    def part(name):
        sub = feat[feat[tf.SPLIT_COL] == name]
        return sub[columns], sub[tf.TARGET].astype(int)

    (X_tr, y_tr) = part("train")
    (X_val, y_val) = part("val")
    (X_te, y_te) = part("test")
    print(f"train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}  "
          f"(train fraud rate {y_tr.mean():.4f})")

    pipe = build_pipeline(args.feature_set)
    print(f"Fitting HistGradientBoostingClassifier on feature set '{args.feature_set}' "
          f"({len(columns)} columns) ...")
    pipe.fit(X_tr, y_tr)

    def scores(X):
        return pipe.predict_proba(X)[:, 1]

    val_rep = imbalanced_report(y_val, scores(X_val), "VAL")
    test_rep = imbalanced_report(y_te, scores(X_te), "TEST")
    print_report(val_rep)
    print_report(test_rep)
    print("(Dataset baseline for reference: single-row PR-AUC ~0.57; "
          "+account history ~0.84. The 'full' feature set adds static KYC and "
          "channel/counterparty fields a PDF statement does not carry; the 'pdf' "
          "set uses only amount, direction, balance, date and account history.)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_set": args.feature_set,
            "feature_columns": columns,
            "categorical": spec["categorical"],
            "numeric": spec["numeric"],
            "metrics": {"val": val_rep, "test": test_rep},
        },
        out,
    )
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    main()
