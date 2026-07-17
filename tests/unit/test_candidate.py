"""Unit tests for the XGBoost candidate pipeline and CV summaries."""

import numpy as np
import pandas as pd

from risklens.modeling.candidate import (
    build_xgboost_pipeline,
    summarize_cross_validation,
)


def sample_frame(rows: int = 40) -> pd.DataFrame:
    """Create a small application-only training frame."""
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "SK_ID_CURR": 200000 + index,
            "AMT_INCOME_TOTAL": 100000.0 + index * 1000,
            "AMT_CREDIT": 200000.0 + index * 2500,
            "AMT_ANNUITY": 20000.0 + index * 100,
            "AMT_GOODS_PRICE": 180000.0 + index * 2000,
            "DAYS_BIRTH": -12000 - index * 50,
            "DAYS_EMPLOYED": -1000 - index * 10,
            "EXT_SOURCE_1": 0.2 + index / 100,
            "EXT_SOURCE_2": 0.3 + index / 100,
            "EXT_SOURCE_3": 0.4 + index / 100,
            "NAME_CONTRACT_TYPE": np.where(index % 2 == 0, "Cash", "Revolving"),
        }
    )


def test_xgboost_pipeline_returns_probabilities() -> None:
    frame = sample_frame()
    target = pd.Series([index % 2 for index in range(len(frame))])
    pipeline = build_xgboost_pipeline(
        frame,
        n_estimators=5,
        max_depth=2,
        min_child_weight=1,
    )
    pipeline.fit(frame, target)
    probabilities = pipeline.predict_proba(frame.iloc[:4])[:, 1]
    assert probabilities.shape == (4,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_cross_validation_summary_reports_mean_and_deviation() -> None:
    summary = summarize_cross_validation(
        {
            "test_roc_auc": np.array([0.70, 0.72, 0.71]),
            "test_average_precision": np.array([0.20, 0.22, 0.21]),
            "test_neg_log_loss": np.array([-0.30, -0.32, -0.31]),
        }
    )
    assert summary["roc_auc"]["mean"] == 0.71
    assert summary["average_precision"]["standard_deviation"] == 0.01
    assert summary["log_loss"]["mean"] == 0.31
