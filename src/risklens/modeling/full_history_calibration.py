"""Probability calibration for the selected full-history candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from risklens.config import (
    INTERIM_DATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.calibration import (
    CalibratedRiskModel,
    build_calibrator,
    select_calibration_method,
)
from risklens.modeling.full_history import FULL_HISTORY_MODEL_PATH
from risklens.modeling.metrics import evaluate_probabilities

FULL_HISTORY_CALIBRATED_MODEL_PATH = MODEL_DIR / "full_history_xgboost_calibrated.joblib"
FULL_HISTORY_CALIBRATION_METRICS_PATH = METRICS_DIR / "full_history_calibration_metrics.json"


def load_full_history_calibration_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    history_filename: str = "full_history_features.parquet",
) -> pd.DataFrame:
    """Join history to only applicants assigned to the calibration split."""
    assignments_path = processed_dir / "split_assignments.parquet"
    history_path = interim_dir / history_filename
    if not assignments_path.exists():
        raise FileNotFoundError("Run `risklens create-splits` before calibration")
    if not history_path.exists():
        raise FileNotFoundError("Run `risklens build-history-features` before calibration")

    assignments = pd.read_parquet(assignments_path)
    calibration_ids = assignments.loc[assignments["split"] == "calibration", ["SK_ID_CURR"]]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    history = pd.read_parquet(history_path)

    if "TARGET" in history.columns:
        raise ValueError("History feature store must never contain TARGET")
    if history["SK_ID_CURR"].duplicated().any():
        raise ValueError("History feature store contains duplicate applicant IDs")
    if not history.columns.drop("SK_ID_CURR").tolist():
        raise ValueError("History feature store has no model features")

    calibration = calibration_ids.merge(
        application, on="SK_ID_CURR", how="left", validate="one_to_one"
    ).merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    if len(calibration) != len(calibration_ids):
        raise ValueError("Calibration records do not match split assignments")
    if calibration["TARGET"].isna().any():
        raise ValueError("Application rows are missing for calibration applicants")
    return calibration


def calibrate_full_history_candidate(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Select calibration internally, then refit it on all calibration data."""
    config = load_modeling_config(config_path)
    calibration_config = config["calibration"]
    random_seed = int(config["random_seed"])
    threshold = float(config["baseline"]["decision_threshold"])
    selection_fraction = float(calibration_config["selection_fraction"])
    history_filename = str(config["feature_store"]["output_file"])

    candidate_path = model_dir / FULL_HISTORY_MODEL_PATH.name
    if not candidate_path.exists():
        raise FileNotFoundError("Run `risklens train-full-history` before calibration")
    base_model = joblib.load(candidate_path)
    calibration = load_full_history_calibration_data(history_filename=history_filename)
    targets = calibration.pop("TARGET").astype(int)
    fit_frame, selection_frame, fit_targets, selection_targets = train_test_split(
        calibration,
        targets,
        test_size=selection_fraction,
        random_state=random_seed,
        stratify=targets,
    )

    fit_raw = base_model.predict_proba(fit_frame)[:, 1]
    selection_raw = base_model.predict_proba(selection_frame)[:, 1]
    selection_metrics: dict[str, dict[str, Any]] = {
        "uncalibrated": evaluate_probabilities(
            selection_targets.to_numpy(), selection_raw, threshold
        )
    }
    for method in calibration_config["methods"]:
        method_name = str(method)
        calibrator = build_calibrator(method_name).fit(fit_raw, fit_targets.to_numpy())
        calibrated = calibrator.predict(selection_raw)
        selection_metrics[method_name] = evaluate_probabilities(
            selection_targets.to_numpy(), calibrated, threshold
        )

    selected_method = select_calibration_method(
        selection_metrics,
        primary_metric=str(calibration_config["primary_metric"]),
        secondary_metric=str(calibration_config["secondary_metric"]),
    )
    all_raw = base_model.predict_proba(calibration)[:, 1]
    final_calibrator = build_calibrator(selected_method).fit(all_raw, targets.to_numpy())
    calibrated_model = CalibratedRiskModel(base_model, final_calibrator, selected_method)
    report = {
        "base_model": "full_history_xgboost_candidate",
        "model_scope": "application_plus_full_history",
        "calibration_rows": int(len(calibration)),
        "calibration_fit_rows": int(len(fit_frame)),
        "calibration_selection_rows": int(len(selection_frame)),
        "selection_metrics": selection_metrics,
        "selected_method": selected_method,
        "selection_primary_metric": calibration_config["primary_metric"],
        "selection_secondary_metric": calibration_config["secondary_metric"],
        "data_policy": "calibration_internal_selection_final_holdout_sealed",
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name)
    (metrics_dir / FULL_HISTORY_CALIBRATION_METRICS_PATH.name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
