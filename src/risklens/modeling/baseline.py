"""Leakage-safe application-only benchmark model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from risklens.config import (
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.features.application import ApplicationFeatureEngineer, add_application_features
from risklens.features.preprocessing import build_preprocessor
from risklens.modeling.metrics import evaluate_probabilities

BASELINE_MODEL_PATH = MODEL_DIR / "application_logistic_baseline.joblib"


def build_logistic_pipeline(
    sample_frame: pd.DataFrame,
    regularization_c: float = 1.0,
    maximum_iterations: int = 1000,
    solver: str = "lbfgs",
    random_seed: int = 42,
) -> Pipeline:
    """Build an unfitted, end-to-end application-only logistic pipeline."""
    engineered_sample = add_application_features(sample_frame)
    preprocessor = build_preprocessor(engineered_sample)
    model = LogisticRegression(
        C=regularization_c,
        max_iter=maximum_iterations,
        solver=solver,
        random_state=random_seed,
    )
    return Pipeline(
        steps=[
            ("features", ApplicationFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_train_validation_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only train and validation records; calibration and holdout stay sealed."""
    assignments_path = processed_dir / "split_assignments.parquet"
    if not assignments_path.exists():
        raise FileNotFoundError("Run `risklens create-splits` before training models")

    assignments = pd.read_parquet(assignments_path)
    allowed = assignments[assignments["split"].isin(["train", "validation"])]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    merged = application.merge(
        allowed[["SK_ID_CURR", "split"]],
        on="SK_ID_CURR",
        how="inner",
        validate="one_to_one",
    )
    train = merged[merged["split"] == "train"].drop(columns="split")
    validation = merged[merged["split"] == "validation"].drop(columns="split")
    if len(train) != int((assignments["split"] == "train").sum()):
        raise ValueError("Training records do not match split assignments")
    if len(validation) != int((assignments["split"] == "validation").sum()):
        raise ValueError("Validation records do not match split assignments")
    return train, validation


def train_baselines(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Train dummy and logistic benchmarks and evaluate on validation only."""
    config = load_modeling_config(config_path)
    baseline_config = config["baseline"]
    logistic_config = baseline_config["logistic_regression"]
    threshold = float(baseline_config["decision_threshold"])
    random_seed = int(config["random_seed"])

    train, validation = load_train_validation_data()
    train_target = train.pop("TARGET").astype(int)
    validation_target = validation.pop("TARGET").astype(int)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(train), 1)), train_target)
    dummy_probabilities = dummy.predict_proba(np.zeros((len(validation), 1)))[:, 1]

    pipeline = build_logistic_pipeline(
        train.iloc[:1000],
        regularization_c=float(logistic_config["regularization_c"]),
        maximum_iterations=int(logistic_config["maximum_iterations"]),
        solver=str(logistic_config["solver"]),
        random_seed=random_seed,
    )
    pipeline.fit(train, train_target)
    logistic_probabilities = pipeline.predict_proba(validation)[:, 1]

    report = {
        "model_scope": "application_only",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "random_seed": random_seed,
        "data_policy": "train_fit_validation_evaluation_calibration_and_holdout_sealed",
        "models": {
            "dummy_prior": evaluate_probabilities(
                validation_target.to_numpy(), dummy_probabilities, threshold
            ),
            "logistic_regression": evaluate_probabilities(
                validation_target.to_numpy(), logistic_probabilities, threshold
            ),
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / BASELINE_MODEL_PATH.name)
    (metrics_dir / "application_baseline_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
