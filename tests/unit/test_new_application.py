"""Tests for the governed new-application simulator model."""

import pandas as pd

from risklens.modeling.new_application import (
    _internal_splits,
    add_simulator_features,
    simulator_input_columns,
)


def test_simulator_features_create_application_time_ratios() -> None:
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [600_000.0],
            "AMT_CREDIT": [900_000.0],
            "AMT_ANNUITY": [50_000.0],
            "AMT_GOODS_PRICE": [850_000.0],
            "DAYS_EMPLOYED": [-1461.0],
            "EXT_SOURCE_1": [0.52],
            "EXT_SOURCE_2": [0.61],
            "EXT_SOURCE_3": [0.49],
        }
    )
    result = add_simulator_features(frame)
    assert result.loc[0, "CREDIT_INCOME_RATIO"] == 1.5
    assert round(result.loc[0, "EMPLOYMENT_YEARS"], 2) == 4.0
    assert result.loc[0, "EXT_SOURCE_COUNT"] == 3


def test_input_allowlist_excludes_sensitive_and_outcome_columns() -> None:
    config = {
        "input_features": {
            "numeric": ["AMT_INCOME_TOTAL"],
            "categorical": ["FLAG_OWN_CAR"],
        }
    }
    columns = simulator_input_columns(config)
    assert columns == ["AMT_INCOME_TOTAL", "FLAG_OWN_CAR"]
    assert not {"TARGET", "CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"} & set(columns)


def test_internal_split_is_complete_exclusive_and_stratified() -> None:
    frame = pd.DataFrame({"row": range(1000)})
    target = pd.Series(([0] * 900) + ([1] * 100), name="TARGET")
    splits = _internal_splits(frame, target, seed=42)
    observed = set()
    for split_frame, split_target in splits.values():
        assert not observed.intersection(split_frame["row"])
        observed.update(split_frame["row"])
        assert abs(float(split_target.mean()) - 0.1) < 0.01
    assert observed == set(range(1000))
