"""Unit tests for chunked relational feature engineering."""

from pathlib import Path

import pandas as pd
import pytest

from risklens.features.history import (
    aggregate_csv_in_chunks,
    assemble_history_feature_store,
    transform_installments,
)


def test_chunked_aggregation_combines_sufficient_statistics(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "VALUE": [1.0, 3.0, 5.0, 10.0],
        }
    ).to_csv(path, index=False)
    result = aggregate_csv_in_chunks(
        path,
        prefix="TEST",
        source_columns=["VALUE"],
        numeric_columns=["VALUE"],
        chunksize=2,
    ).set_index("SK_ID_CURR")
    assert result.loc[1, "TEST_RECORD_COUNT"] == 3
    assert result.loc[1, "TEST_VALUE_SUM"] == 9.0
    assert result.loc[1, "TEST_VALUE_MEAN"] == pytest.approx(3.0)
    assert result.loc[2, "TEST_VALUE_MEAN"] == pytest.approx(10.0)


def test_installment_transform_creates_behavior_features() -> None:
    frame = pd.DataFrame(
        {
            "DAYS_INSTALMENT": [-10, -10],
            "DAYS_ENTRY_PAYMENT": [-12, -5],
            "AMT_INSTALMENT": [100.0, 100.0],
            "AMT_PAYMENT": [100.0, 70.0],
        }
    )
    result = transform_installments(frame)
    assert result["DAYS_LATE"].tolist() == [0, 5]
    assert result["PAYMENT_SHORTFALL"].tolist() == [0.0, 30.0]
    assert result["PAYMENT_RATIO"].tolist() == [1.0, 0.7]


def test_feature_store_marks_missing_history_explicitly() -> None:
    applicants = pd.DataFrame({"SK_ID_CURR": [1, 2]})
    aggregate = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "BUREAU_RECORD_COUNT": [3],
            "BUREAU_VALUE_MEAN": [2.0],
        }
    )
    result = assemble_history_feature_store(applicants, [("BUREAU", aggregate)])
    assert result.loc[0, "BUREAU_HISTORY_AVAILABLE"] == 1
    assert result.loc[1, "BUREAU_HISTORY_AVAILABLE"] == 0
    assert result.loc[1, "BUREAU_RECORD_COUNT"] == 0
    assert pd.isna(result.loc[1, "BUREAU_VALUE_MEAN"])


def test_feature_store_rejects_duplicate_applicants() -> None:
    applicants = pd.DataFrame({"SK_ID_CURR": [1, 1]})
    with pytest.raises(ValueError, match="unique"):
        assemble_history_feature_store(applicants, [])
