"""Unit tests for probability calibration and method selection."""

import numpy as np

from risklens.modeling.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
    probability_logit,
    select_calibration_method,
)


def test_probability_logit_is_finite_at_boundaries() -> None:
    result = probability_logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(result).all()
    assert result[1] == 0.0


def test_identity_calibrator_preserves_probabilities() -> None:
    raw = np.array([0.1, 0.5, 0.9])
    targets = np.array([0, 0, 1])
    calibrated = IdentityCalibrator().fit(raw, targets).predict(raw)
    np.testing.assert_array_equal(calibrated, raw)


def test_sigmoid_calibrator_returns_valid_probabilities() -> None:
    raw = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
    targets = np.array([0, 0, 0, 1, 1, 1])
    calibrated = SigmoidCalibrator().fit(raw, targets).predict(raw)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert np.all(np.diff(calibrated) >= 0)


def test_isotonic_calibrator_returns_monotonic_probabilities() -> None:
    raw = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
    targets = np.array([0, 0, 0, 1, 1, 1])
    calibrated = IsotonicCalibrator().fit(raw, targets).predict(raw)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert np.all(np.diff(calibrated) >= 0)


def test_method_selection_prefers_lower_brier_then_log_loss() -> None:
    metrics = {
        "sigmoid": {"brier_score": 0.07, "log_loss": 0.25},
        "isotonic": {"brier_score": 0.06, "log_loss": 0.30},
    }
    assert select_calibration_method(metrics) == "isotonic"
