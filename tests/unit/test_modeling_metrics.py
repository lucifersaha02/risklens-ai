"""Unit tests for probability-model evaluation."""

import numpy as np
import pytest

from risklens.modeling.metrics import evaluate_probabilities


def test_perfect_probabilities_score_perfectly() -> None:
    result = evaluate_probabilities(
        np.array([0, 0, 1, 1]),
        np.array([0.05, 0.10, 0.90, 0.95]),
    )
    assert result["roc_auc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["f1"] == 1.0


def test_metrics_report_probability_quality() -> None:
    result = evaluate_probabilities(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.7, 0.2, 0.8]),
    )
    assert result["brier_score"] < 0.10
    assert result["mean_predicted_probability"] == pytest.approx(0.45)


def test_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="interval"):
        evaluate_probabilities(np.array([0, 1]), np.array([0.2, 1.2]))
