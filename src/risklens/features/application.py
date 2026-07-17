"""Deterministic application-time feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

REQUIRED_APPLICATION_COLUMNS = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while converting zero denominators and infinities to null."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_application_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features available at application-scoring time without fitted state."""
    missing = REQUIRED_APPLICATION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Application frame is missing columns: {sorted(missing)}")

    result = frame.copy()
    result["DAYS_EMPLOYED_ANOMALOUS"] = (result["DAYS_EMPLOYED"] == 365243).astype("int8")
    employed_days = result["DAYS_EMPLOYED"].replace(365243, np.nan)
    result["DAYS_EMPLOYED"] = employed_days

    result["AGE_YEARS"] = (-result["DAYS_BIRTH"] / 365.25).clip(18, 100)
    result["EMPLOYMENT_YEARS"] = (-employed_days / 365.25).clip(lower=0)
    result["CREDIT_INCOME_RATIO"] = safe_ratio(result["AMT_CREDIT"], result["AMT_INCOME_TOTAL"])
    result["ANNUITY_INCOME_RATIO"] = safe_ratio(result["AMT_ANNUITY"], result["AMT_INCOME_TOTAL"])
    result["CREDIT_ANNUITY_RATIO"] = safe_ratio(result["AMT_CREDIT"], result["AMT_ANNUITY"])
    result["GOODS_CREDIT_RATIO"] = safe_ratio(result["AMT_GOODS_PRICE"], result["AMT_CREDIT"])

    external_columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    external = result[external_columns]
    result["EXT_SOURCE_MEAN"] = external.mean(axis=1)
    result["EXT_SOURCE_STD"] = external.std(axis=1)
    result["EXT_SOURCE_MIN"] = external.min(axis=1)
    result["EXT_SOURCE_MAX"] = external.max(axis=1)
    result["EXT_SOURCE_COUNT"] = external.notna().sum(axis=1).astype("int8")

    if {"OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE"}.issubset(result):
        result["SOCIAL_DEFAULT_30_RATIO"] = safe_ratio(
            result["DEF_30_CNT_SOCIAL_CIRCLE"],
            result["OBS_30_CNT_SOCIAL_CIRCLE"],
        )
    if {"OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"}.issubset(result):
        result["SOCIAL_DEFAULT_60_RATIO"] = safe_ratio(
            result["DEF_60_CNT_SOCIAL_CIRCLE"],
            result["OBS_60_CNT_SOCIAL_CIRCLE"],
        )

    return result


class ApplicationFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn-compatible stateless application feature transformer."""

    def fit(
        self, frame: pd.DataFrame, target: pd.Series | None = None
    ) -> ApplicationFeatureEngineer:
        """Validate inputs; no statistics are learned."""
        del target
        missing = REQUIRED_APPLICATION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Application frame is missing columns: {sorted(missing)}")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic application feature engineering."""
        return add_application_features(frame)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return transformer parameters for scikit-learn compatibility."""
        del deep
        return {}
