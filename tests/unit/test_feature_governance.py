"""Tests for enforceable model-input feature governance."""

import pandas as pd
import pytest

from risklens.features.governance import (
    FeatureGovernanceSelector,
    apply_feature_governance,
)


def governance_frame() -> pd.DataFrame:
    """Return a frame containing model and audit-only attributes."""
    return pd.DataFrame(
        {
            "CODE_GENDER": ["F", "M"],
            "DAYS_BIRTH": [-12000, -15000],
            "AGE_YEARS": [32.9, 41.1],
            "NAME_FAMILY_STATUS": ["Married", "Single"],
            "AMT_CREDIT": [100000.0, 200000.0],
        }
    )


def test_governance_removes_sensitive_features_without_mutating_input() -> None:
    frame = governance_frame()
    excluded = ["CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS", "NAME_FAMILY_STATUS"]
    governed = apply_feature_governance(frame, excluded)
    assert governed.columns.tolist() == ["AMT_CREDIT"]
    assert "CODE_GENDER" in frame.columns


def test_governance_rejects_misspelled_or_missing_exclusions() -> None:
    with pytest.raises(ValueError, match="missing from the frame"):
        apply_feature_governance(governance_frame(), ["CODE_GENDR"])


def test_governance_selector_is_sklearn_compatible() -> None:
    selector = FeatureGovernanceSelector(["CODE_GENDER"]).fit(governance_frame())
    transformed = selector.transform(governance_frame())
    assert "CODE_GENDER" not in transformed.columns
    assert selector.get_params()["excluded_features"] == ["CODE_GENDER"]
