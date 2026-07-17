"""Unit tests for the application-only logistic benchmark."""

import numpy as np
import pandas as pd

from risklens.modeling.baseline import build_logistic_pipeline


def sample_frame(rows: int = 40) -> pd.DataFrame:
    """Create small application data with both target classes."""
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "SK_ID_CURR": 100000 + index,
            "AMT_INCOME_TOTAL": 100000.0 + index * 1000,
            "AMT_CREDIT": 200000.0 + index * 2000,
            "AMT_ANNUITY": 20000.0 + index * 100,
            "AMT_GOODS_PRICE": 180000.0 + index * 1500,
            "DAYS_BIRTH": -12000 - index * 50,
            "DAYS_EMPLOYED": np.where(index == 3, 365243, -1000 - index * 10),
            "EXT_SOURCE_1": 0.2 + index / 100,
            "EXT_SOURCE_2": 0.3 + index / 100,
            "EXT_SOURCE_3": 0.4 + index / 100,
            "NAME_CONTRACT_TYPE": np.where(index % 2 == 0, "Cash", "Revolving"),
        }
    )


def test_logistic_pipeline_returns_valid_probabilities() -> None:
    frame = sample_frame()
    target = pd.Series([index % 2 for index in range(len(frame))])
    pipeline = build_logistic_pipeline(frame, maximum_iterations=500)
    pipeline.fit(frame, target)
    probabilities = pipeline.predict_proba(frame.iloc[:3])[:, 1]
    assert probabilities.shape == (3,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
