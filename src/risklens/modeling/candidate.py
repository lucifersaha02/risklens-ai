"""Cross-validated XGBoost application-only model candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from risklens.config import METRICS_DIR, MODEL_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.features.application import ApplicationFeatureEngineer, add_application_features
from risklens.features.preprocessing import build_preprocessor
from risklens.modeling.baseline import load_train_validation_data
from risklens.modeling.metrics import evaluate_probabilities

CANDIDATE_MODEL_PATH = MODEL_DIR / "application_xgboost_candidate.joblib"


def build_xgboost_pipeline(
    sample_frame: pd.DataFrame,
    *,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    min_child_weight: float = 20,
    reg_lambda: float = 2.0,
    tree_method: str = "hist",
    random_seed: int = 42,
) -> Pipeline:
    """Build an unfitted end-to-end XGBoost pipeline."""
    engineered_sample = add_application_features(sample_frame)
    preprocessor = build_preprocessor(engineered_sample)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=tree_method,
        random_state=random_seed,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("features", ApplicationFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def summarize_cross_validation(results: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize cross-validation scores as fold values, mean, and deviation."""
    metrics: dict[str, Any] = {}
    for result_name, metric_name, multiplier in (
        ("test_roc_auc", "roc_auc", 1.0),
        ("test_average_precision", "average_precision", 1.0),
        ("test_neg_log_loss", "log_loss", -1.0),
    ):
        values = np.asarray(results[result_name], dtype=float) * multiplier
        metrics[metric_name] = {
            "fold_values": [round(float(value), 6) for value in values],
            "mean": round(float(values.mean()), 6),
            "standard_deviation": round(float(values.std(ddof=1)), 6),
        }
    return metrics


def train_xgboost_candidate(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Cross-validate on train, refit on train, and evaluate once on validation."""
    config = load_modeling_config(config_path)
    model_config = config["xgboost"]
    cv_config = config["cross_validation"]
    threshold = float(config["baseline"]["decision_threshold"])
    random_seed = int(config["random_seed"])

    train, validation = load_train_validation_data()
    train_target = train.pop("TARGET").astype(int)
    validation_target = validation.pop("TARGET").astype(int)
    pipeline = build_xgboost_pipeline(
        train.iloc[:1000],
        n_estimators=int(model_config["n_estimators"]),
        max_depth=int(model_config["max_depth"]),
        learning_rate=float(model_config["learning_rate"]),
        subsample=float(model_config["subsample"]),
        colsample_bytree=float(model_config["colsample_bytree"]),
        min_child_weight=float(model_config["min_child_weight"]),
        reg_lambda=float(model_config["reg_lambda"]),
        tree_method=str(model_config["tree_method"]),
        random_seed=random_seed,
    )
    cross_validator = StratifiedKFold(
        n_splits=int(cv_config["folds"]),
        shuffle=True,
        random_state=random_seed,
    )
    scores = cross_validate(
        pipeline,
        train,
        train_target,
        cv=cross_validator,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "neg_log_loss": "neg_log_loss",
        },
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )

    pipeline.fit(train, train_target)
    validation_probabilities = pipeline.predict_proba(validation)[:, 1]
    validation_metrics = evaluate_probabilities(
        validation_target.to_numpy(), validation_probabilities, threshold
    )
    report: dict[str, Any] = {
        "model": "xgboost",
        "model_scope": "application_only",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "random_seed": random_seed,
        "cross_validation_folds": int(cv_config["folds"]),
        "cross_validation": summarize_cross_validation(scores),
        "validation": validation_metrics,
        "data_policy": "cv_on_train_refit_train_evaluate_validation_calibration_holdout_sealed",
    }

    baseline_metrics_path = metrics_dir / "application_baseline_metrics.json"
    if baseline_metrics_path.exists():
        baseline_report = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        logistic_metrics = baseline_report["models"]["logistic_regression"]
        primary_metric = str(cv_config["primary_metric"])
        report["comparison"] = {
            "primary_metric": primary_metric,
            "logistic_validation": logistic_metrics[primary_metric],
            "xgboost_validation": validation_metrics[primary_metric],
            "selected_model": (
                "xgboost"
                if validation_metrics[primary_metric] > logistic_metrics[primary_metric]
                else "logistic_regression"
            ),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / CANDIDATE_MODEL_PATH.name)
    (metrics_dir / "application_xgboost_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
