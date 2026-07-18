"""Leakage-safe training and comparison for the full-history candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from risklens.config import (
    INTERIM_DATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.candidate import build_xgboost_pipeline, summarize_cross_validation
from risklens.modeling.metrics import evaluate_probabilities

FULL_HISTORY_MODEL_PATH = MODEL_DIR / "full_history_xgboost_candidate.joblib"
FULL_HISTORY_METRICS_PATH = METRICS_DIR / "full_history_xgboost_metrics.json"
APPLICATION_METRICS_PATH = METRICS_DIR / "application_xgboost_metrics.json"


def load_full_history_train_validation_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    history_filename: str = "full_history_features.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join target-free history to only the train and validation applicants."""
    assignments_path = processed_dir / "split_assignments.parquet"
    history_path = interim_dir / history_filename
    if not assignments_path.exists():
        raise FileNotFoundError("Run `risklens create-splits` before training models")
    if not history_path.exists():
        raise FileNotFoundError("Run `risklens build-history-features` before this command")

    assignments = pd.read_parquet(assignments_path)
    allowed = assignments[assignments["split"].isin(["train", "validation"])]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    history = pd.read_parquet(history_path)

    if "TARGET" in history.columns:
        raise ValueError("History feature store must never contain TARGET")
    if history["SK_ID_CURR"].duplicated().any():
        raise ValueError("History feature store contains duplicate applicant IDs")

    merged = (
        allowed[["SK_ID_CURR", "split"]]
        .merge(application, on="SK_ID_CURR", how="left", validate="one_to_one")
        .merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    )
    if merged["TARGET"].isna().any():
        raise ValueError("Application rows are missing for assigned applicants")

    history_columns = [column for column in history.columns if column != "SK_ID_CURR"]
    if not history_columns:
        raise ValueError("History feature store has no model features")
    # Missing history is meaningful. Record counts and availability flags from the
    # feature store retain that signal; numeric values are imputed inside the pipeline.
    train = merged[merged["split"] == "train"].drop(columns="split")
    validation = merged[merged["split"] == "validation"].drop(columns="split")
    return train, validation


def _metric_comparison(
    application_metrics: dict[str, Any], full_history_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Compare candidates on the exact same validation population."""
    comparison: dict[str, Any] = {}
    for metric in ("roc_auc", "average_precision", "brier_score", "log_loss"):
        application_value = float(application_metrics[metric])
        history_value = float(full_history_metrics[metric])
        comparison[metric] = {
            "application_only": application_value,
            "full_history": history_value,
            "delta_full_history_minus_application": history_value - application_value,
        }
    comparison["selected_candidate"] = (
        "full_history_xgboost"
        if full_history_metrics["average_precision"] > application_metrics["average_precision"]
        else "application_xgboost"
    )
    return comparison


def train_full_history_candidate(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Cross-validate on train and compare with application-only on validation."""
    config = load_modeling_config(config_path)
    model_config = config["full_history_xgboost"]
    cv_config = config["cross_validation"]
    random_seed = int(config["random_seed"])
    threshold = float(config["baseline"]["decision_threshold"])
    history_filename = str(config["feature_store"]["output_file"])
    governance_config = config["feature_governance"]
    excluded_features = [
        str(feature) for feature in governance_config["excluded_decision_features"]
    ]

    train, validation = load_full_history_train_validation_data(history_filename=history_filename)
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
        excluded_features=excluded_features,
    )
    cross_validator = StratifiedKFold(
        n_splits=int(cv_config["folds"]), shuffle=True, random_state=random_seed
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
    probabilities = pipeline.predict_proba(validation)[:, 1]
    validation_metrics = evaluate_probabilities(
        validation_target.to_numpy(), probabilities, threshold
    )
    report: dict[str, Any] = {
        "model": "xgboost",
        "model_scope": "application_plus_full_history",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "random_seed": random_seed,
        "feature_governance": {
            "policy_name": str(governance_config["policy_name"]),
            "excluded_decision_features": excluded_features,
            "rationale": str(governance_config["rationale"]),
        },
        "cross_validation_folds": int(cv_config["folds"]),
        "cross_validation": summarize_cross_validation(scores),
        "validation": validation_metrics,
        "data_policy": "cv_train_refit_train_validation_only_calibration_holdout_sealed",
    }

    application_path = metrics_dir / APPLICATION_METRICS_PATH.name
    if not application_path.exists():
        raise FileNotFoundError("Run `risklens train-candidate` before comparing candidates")
    application_report = json.loads(application_path.read_text(encoding="utf-8"))
    if int(application_report["validation_rows"]) != len(validation):
        raise ValueError("Candidate validation populations do not match")
    report["comparison"] = _metric_comparison(application_report["validation"], validation_metrics)

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / FULL_HISTORY_MODEL_PATH.name)
    (metrics_dir / FULL_HISTORY_METRICS_PATH.name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
