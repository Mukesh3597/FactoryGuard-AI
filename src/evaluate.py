# src/evaluate.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

@dataclass
class EvalResult:
    pr_auc: float
    best_threshold: float
    best_f1: float
    confusion_matrix: np.ndarray
    report: str

def _safe_f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    return (2 * precision * recall) / (precision + recall + 1e-12)

def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Returns (best_threshold, best_f1) using PR curve points.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = _safe_f1(precision, recall)

    # thresholds length is (len(precision)-1), align index safely
    best_idx = int(np.nanargmax(f1))
    if best_idx == 0:
        best_t = 0.5  # fallback
    else:
        # best_idx corresponds to precision/recall point; threshold index is best_idx-1
        t_idx = min(best_idx - 1, len(thresholds) - 1)
        best_t = float(thresholds[t_idx])

    return best_t, float(f1[best_idx])

def evaluate_binary_classifier(y_true, y_prob, threshold: float | None = None) -> EvalResult:
    """
    Evaluate imbalanced binary classifier. Uses PR-AUC (Average Precision).
    If threshold is None -> chooses best threshold by F1 on PR curve.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    pr_auc = float(average_precision_score(y_true, y_prob))

    if threshold is None:
        threshold, best_f1 = find_best_threshold_by_f1(y_true, y_prob)
    else:
        # compute F1 for given threshold
        y_pred_tmp = (y_prob >= threshold).astype(int)
        tp = ((y_pred_tmp == 1) & (y_true == 1)).sum()
        fp = ((y_pred_tmp == 1) & (y_true == 0)).sum()
        fn = ((y_pred_tmp == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        best_f1 = float((2 * precision * recall) / (precision + recall + 1e-12))

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    return EvalResult(
        pr_auc=pr_auc,
        best_threshold=float(threshold),
        best_f1=float(best_f1),
        confusion_matrix=cm,
        report=rep,
    )

def print_eval(result: EvalResult) -> None:
    print(f"PR-AUC (Average Precision): {result.pr_auc:.6f}")
    print(f"Best Threshold: {result.best_threshold:.4f}")
    print(f"Best F1: {result.best_f1:.6f}")
    print("Confusion Matrix:\n", result.confusion_matrix)
    print("Classification Report:\n", result.report)
