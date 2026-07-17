"""Training-only preprocessing for application features."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NON_FEATURE_COLUMNS = {"SK_ID_CURR", "TARGET", "split"}


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate model inputs into numeric and categorical columns."""
    candidates = frame.drop(columns=list(NON_FEATURE_COLUMNS), errors="ignore")
    numeric = candidates.select_dtypes(include="number").columns.tolist()
    categorical = candidates.select_dtypes(exclude="number").columns.tolist()
    if not numeric and not categorical:
        raise ValueError("No model features were found")
    return numeric, categorical


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted preprocessor whose statistics must be learned on train only."""
    numeric, categorical = feature_columns(frame)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.001,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
