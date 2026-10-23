"""Shared evaluation helpers — imbalance-aware metrics and honest reporting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def topk_precision_recall(y_true, scores, k_fraction: float = 0.01):
    """Precision and recall if we alert on the top ``k_fraction`` of scores.

    This mirrors how an AML team works a bounded alert queue, and is the metric
    the transaction dataset's own baseline reports (precision/recall @ top 1%).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0
    k = max(1, int(round(n * k_fraction)))
    order = np.argsort(-scores)
    top = order[:k]
    tp = int(y_true[top].sum())
    total_pos = int(y_true.sum())
    precision = tp / k
    recall = tp / total_pos if total_pos else 0.0
    return precision, recall, k


def imbalanced_report(y_true, scores, label: str = "") -> dict:
    """PR-AUC (primary for rare positives), ROC-AUC, and top-1% precision/recall."""
    y_true = np.asarray(y_true)
    pr_auc = float(average_precision_score(y_true, scores)) if y_true.sum() else float("nan")
    try:
        roc = float(roc_auc_score(y_true, scores)) if y_true.sum() else float("nan")
    except ValueError:
        roc = float("nan")
    p1, r1, k = topk_precision_recall(y_true, scores, 0.01)
    result = {
        "label": label,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "precision_at_top1pct": p1,
        "recall_at_top1pct": r1,
        "alerts_at_top1pct": k,
    }
    return result


def binary_report(y_true, y_pred, scores, label: str = "") -> dict:
    """Balanced-task report (used by the statement model): accuracy-free summary
    with precision/recall/F1 plus ROC-AUC and PR-AUC."""
    y_true = np.asarray(y_true)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc = float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        roc = float("nan")
    pr_auc = float(average_precision_score(y_true, scores)) if y_true.sum() else float("nan")
    return {
        "label": label,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }


def print_report(d: dict) -> None:
    lbl = d.get("label", "")
    print(f"--- {lbl} ---")
    for k, v in d.items():
        if k == "label":
            continue
        if isinstance(v, float):
            print(f"    {k:22}: {v:.4f}")
        else:
            print(f"    {k:22}: {v}")
