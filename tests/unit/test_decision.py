"""Unit tests for cost-sensitive decision policy."""

import numpy as np
import pytest

from risklens.modeling.decision import (
    build_threshold_table,
    theoretical_cost_threshold,
    threshold_metrics,
)


def test_theoretical_threshold_uses_cost_ratio() -> None:
    assert theoretical_cost_threshold(5.0, 1.0) == pytest.approx(1 / 6)


def test_theoretical_threshold_rejects_nonpositive_costs() -> None:
    with pytest.raises(ValueError, match="positive"):
        theoretical_cost_threshold(0.0, 1.0)


def test_threshold_metrics_calculate_confusion_and_cost() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9])
    result = threshold_metrics(targets, probabilities, 0.5, 5.0, 1.0)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["total_cost_units"] == 6.0


def test_threshold_table_contains_every_candidate() -> None:
    table = build_threshold_table(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.7, 0.2, 0.8]),
        np.array([0.2, 0.5]),
        5.0,
        1.0,
    )
    assert table["threshold"].tolist() == [0.2, 0.5]
