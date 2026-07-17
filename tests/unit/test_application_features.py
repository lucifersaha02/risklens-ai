"""Unit tests for application feature engineering and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from risklens.features.application import (
    ApplicationFeatureEngineer,
    add_application_features,
    safe_ratio,
)
from risklens.features.preprocessing import build_preprocessor, feature_columns


def application_frame() -> pd.DataFrame:
    """Return representative application-time inputs."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003],
            "TARGET": [0, 1, 0],
            "AMT_INCOME_TOTAL": [100000.0, 0.0, 200000.0],
            "AMT_CREDIT": [200000.0, 100000.0, 300000.0],
            "AMT_ANNUITY": [20000.0, 10000.0, 0.0],
            "AMT_GOODS_PRICE": [180000.0, 90000.0, 280000.0],
            "DAYS_BIRTH": [-12000, -15000, -18000],
            "DAYS_EMPLOYED": [-1000, 365243, -2000],
            "EXT_SOURCE_1": [0.2, np.nan, 0.8],
            "EXT_SOURCE_2": [0.3, 0.4, 0.7],
            "EXT_SOURCE_3": [0.4, np.nan, 0.6],
            "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving"],
        }
    )


def test_safe_ratio_converts_zero_denominator_to_null() -> None:
    result = safe_ratio(pd.Series([4.0, 2.0]), pd.Series([2.0, 0.0]))
    assert result.iloc[0] == 2.0
    assert pd.isna(result.iloc[1])


def test_application_features_handle_employment_sentinel() -> None:
    result = add_application_features(application_frame())
    assert result.loc[1, "DAYS_EMPLOYED_ANOMALOUS"] == 1
    assert pd.isna(result.loc[1, "DAYS_EMPLOYED"])
    assert pd.isna(result.loc[1, "EMPLOYMENT_YEARS"])
    assert result.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert result.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(0.3)


def test_feature_engineer_is_stateless_and_sklearn_compatible() -> None:
    frame = application_frame()
    transformer = ApplicationFeatureEngineer().fit(frame)
    first = transformer.transform(frame)
    second = transformer.transform(frame)
    pd.testing.assert_frame_equal(first, second)


def test_feature_columns_exclude_identifiers_and_target() -> None:
    engineered = add_application_features(application_frame())
    numeric, categorical = feature_columns(engineered)
    assert "SK_ID_CURR" not in numeric
    assert "TARGET" not in numeric
    assert "NAME_CONTRACT_TYPE" in categorical


def test_preprocessor_handles_missing_and_unseen_categories() -> None:
    engineered = add_application_features(application_frame())
    train = engineered.iloc[:2]
    scoring = engineered.iloc[[2]].copy()
    scoring["NAME_CONTRACT_TYPE"] = "Unseen product"
    preprocessor = build_preprocessor(train)
    transformed_train = preprocessor.fit_transform(train)
    transformed_scoring = preprocessor.transform(scoring)
    assert transformed_train.shape[1] == transformed_scoring.shape[1]
    assert transformed_scoring.shape[0] == 1
