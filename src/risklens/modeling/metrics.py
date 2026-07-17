"""Probability and classification metrics for credit-risk models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate ranking, probability quality, and threshold classification."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional")
    if len(targets) != len(probabilities) or len(targets) == 0:
        raise ValueError("Targets and probabilities must have equal non-zero length")
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between zero and one")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be in the interval [0, 1]")

    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(targets, probabilities)), 6),
        "average_precision": round(float(average_precision_score(targets, probabilities)), 6),
        "log_loss": round(float(log_loss(targets, probabilities, labels=[0, 1])), 6),
        "brier_score": round(float(brier_score_loss(targets, probabilities)), 6),
        "threshold": threshold,
        "precision": round(float(precision_score(targets, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(targets, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(targets, predictions, zero_division=0)), 6),
        "predicted_positive_rate": round(float(predictions.mean()), 6),
        "observed_positive_rate": round(float(targets.mean()), 6),
        "mean_predicted_probability": round(float(probabilities.mean()), 6),
    }
