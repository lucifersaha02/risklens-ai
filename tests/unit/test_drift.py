"""Unit tests for deterministic drift monitoring primitives."""

import numpy as np
import pytest

from risklens.monitoring.drift import (
    bin_proportions,
    drift_severity,
    population_stability_index,
    quantile_cutpoints,
)


def test_identical_distributions_have_zero_psi() -> None:
    expected = np.array([0.2, 0.3, 0.5])
    assert population_stability_index(expected, expected) == pytest.approx(0.0)


def test_shifted_distribution_has_positive_psi() -> None:
    expected = np.array([0.8, 0.1, 0.1])
    actual = np.array([0.1, 0.1, 0.8])
    assert population_stability_index(expected, actual) > 0.25


def test_binary_cutpoints_and_proportions_cover_every_row() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0, np.nan])
    cutpoints = quantile_cutpoints(values)
    assert cutpoints == [0.5]
    assert bin_proportions(values, cutpoints).sum() == pytest.approx(1.0)


def test_drift_severity_uses_documented_thresholds() -> None:
    assert drift_severity(0.05, 0.10, 0.25) == "stable"
    assert drift_severity(0.15, 0.10, 0.25) == "warning"
    assert drift_severity(0.30, 0.10, 0.25) == "critical"
