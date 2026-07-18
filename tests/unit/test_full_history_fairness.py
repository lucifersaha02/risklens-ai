"""Tests for comparing full-history subgroup disparity diagnostics."""

import pytest

from risklens.fairness.full_history import compare_disparity_gaps


def diagnostic(gap: float | None) -> dict:
    """Return a minimal responsible-AI report."""
    return {
        "diagnostics": {
            "AGE_BAND": {
                "gaps": {
                    "recall_max_min_gap": gap,
                    "roc_auc_max_min_gap": None,
                }
            }
        }
    }


def test_gap_comparison_reports_signed_change() -> None:
    comparison = compare_disparity_gaps(diagnostic(0.40), diagnostic(0.35))
    assert comparison["AGE_BAND"]["recall_max_min_gap"] == pytest.approx(-0.05)


def test_gap_comparison_preserves_unavailable_metrics() -> None:
    comparison = compare_disparity_gaps(diagnostic(None), diagnostic(None))
    assert comparison["AGE_BAND"]["roc_auc_max_min_gap"] is None
