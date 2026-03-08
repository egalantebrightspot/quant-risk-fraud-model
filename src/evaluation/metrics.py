"""Evaluation metrics (ROC, AUC, KS, lift, calibration)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from typing import Tuple


def compute_auc_ks(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute AUC and Kolmogorov–Smirnov statistic from predicted probabilities.

    KS = max |TPR - FPR| (or equivalently max |cumulative positive rate - cumulative negative rate|).

    Returns:
        auc, ks, fpr, tpr, thresholds (from roc_curve).
    """
    y_true = np.asarray(y_true).ravel()
    # y_proba: use P(positive class); if 2d (n, 2), take second column
    proba = np.asarray(y_proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    proba = proba.ravel()

    auc = roc_auc_score(y_true, proba)
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    ks = float(np.max(np.abs(tpr - fpr)))
    return auc, ks, fpr, tpr, thresholds


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Return confusion matrix with layout [[TN, FP], [FN, TP]] for labels (0, 1)."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute a short evaluation summary: AUC, KS, accuracy, precision, recall, F1, confusion matrix."""
    auc, ks, _, _, _ = compute_auc_ks(y_true, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)
    return {
        "auc": auc,
        "ks": ks,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
    }
