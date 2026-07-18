"""Feature-governance controls for decision-model inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def apply_feature_governance(
    frame: pd.DataFrame, excluded_features: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Remove decision-inappropriate attributes while preserving the input frame."""
    excluded = list(dict.fromkeys(str(feature) for feature in excluded_features))
    if not excluded:
        return frame.copy()
    missing = sorted(set(excluded) - set(frame.columns))
    if missing:
        raise ValueError(f"Governance exclusions are missing from the frame: {missing}")
    return frame.drop(columns=excluded).copy()


class FeatureGovernanceSelector(BaseEstimator, TransformerMixin):
    """Sklearn transformer enforcing an explicit feature-exclusion policy."""

    def __init__(self, excluded_features: list[str] | tuple[str, ...]) -> None:
        self.excluded_features = excluded_features

    def fit(
        self, frame: pd.DataFrame, target: pd.Series | None = None
    ) -> FeatureGovernanceSelector:
        """Validate exclusions without learning dataset statistics."""
        del target
        apply_feature_governance(frame, self.excluded_features)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove governed attributes from model inputs."""
        return apply_feature_governance(frame, self.excluded_features)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return parameters for sklearn cloning and persistence."""
        del deep
        return {"excluded_features": self.excluded_features}
