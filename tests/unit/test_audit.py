"""Unit tests for data-audit utilities."""

import pandas as pd
import pytest

from risklens.data.audit import application_summary, missingness_table


def sample_applications() -> pd.DataFrame:
    """Return a small representative application table."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003],
            "TARGET": [0, 1, 0],
            "DAYS_BIRTH": [-12000, -15000, -18000],
            "DAYS_EMPLOYED": [-1000, 365243, -2000],
            "AMT_INCOME_TOTAL": [100000.0, None, 200000.0],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
        }
    )


def test_missingness_table_orders_highest_first() -> None:
    result = missingness_table(sample_applications())
    assert result.iloc[0]["column"] == "AMT_INCOME_TOTAL"
    assert result.iloc[0]["missing_count"] == 1


def test_application_summary_reports_target_and_sentinel() -> None:
    result = application_summary(sample_applications())
    assert result["rows"] == 3
    assert result["duplicate_applicant_ids"] == 0
    assert result["positive_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["days_employed_365243_count"] == 1


def test_application_summary_requires_key_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        application_summary(pd.DataFrame({"value": [1, 2]}))
