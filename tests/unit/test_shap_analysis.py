"""Unit tests for SHAP output normalization and reason codes."""

import numpy as np
import pytest

from risklens.explainability.shap_analysis import (
    global_importance_table,
    normalize_shap_values,
    reason_codes,
)


def test_normalize_binary_class_shap_values() -> None:
    negative = np.zeros((2, 3))
    positive = np.ones((2, 3))
    result = normalize_shap_values([negative, positive])
    np.testing.assert_array_equal(result, positive)


def test_reason_codes_separate_risk_directions() -> None:
    result = reason_codes(
        ["numeric__income", "numeric__credit", "categorical__contract_Cash"],
        np.array([-0.4, 0.8, 0.2]),
        np.array([1.0, 2.0, 1.0]),
        top_n=1,
    )
    assert result["risk_increasing"][0]["feature"] == "credit"
    assert result["risk_reducing"][0]["feature"] == "income"


def test_reason_codes_require_aligned_features() -> None:
    with pytest.raises(ValueError, match="must align"):
        reason_codes(["one"], np.array([0.1, 0.2]), np.array([1.0]))


def test_global_importance_ranks_mean_absolute_contribution() -> None:
    table = global_importance_table(
        ["numeric__small", "numeric__large"],
        np.array([[0.1, 0.8], [-0.1, -0.6]]),
    )
    assert table.iloc[0]["feature"] == "large"
    assert table.iloc[0]["mean_absolute_shap"] == pytest.approx(0.7)
