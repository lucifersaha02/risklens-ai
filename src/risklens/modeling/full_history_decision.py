"""Cost-sensitive policy for the calibrated full-history risk model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risklens.config import (
    INTERIM_DATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.decision import (
    build_threshold_table,
    theoretical_cost_threshold,
    threshold_metrics,
)
from risklens.modeling.full_history_calibration import (
    FULL_HISTORY_CALIBRATED_MODEL_PATH,
)
from risklens.modeling.metrics import evaluate_probabilities

FULL_HISTORY_POLICY_PATH = METRICS_DIR / "full_history_decision_policy.json"
FULL_HISTORY_THRESHOLD_TABLE_PATH = METRICS_DIR / "full_history_threshold_sensitivity.csv"


def load_full_history_validation_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    history_filename: str = "full_history_features.parquet",
) -> pd.DataFrame:
    """Join history to validation applicants without loading holdout rows."""
    assignments_path = processed_dir / "split_assignments.parquet"
    history_path = interim_dir / history_filename
    if not assignments_path.exists():
        raise FileNotFoundError("Run `risklens create-splits` before defining policy")
    if not history_path.exists():
        raise FileNotFoundError("Run `risklens build-history-features` before policy")

    assignments = pd.read_parquet(assignments_path)
    validation_ids = assignments.loc[assignments["split"] == "validation", ["SK_ID_CURR"]]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    history = pd.read_parquet(history_path)

    if "TARGET" in history.columns:
        raise ValueError("History feature store must never contain TARGET")
    if history["SK_ID_CURR"].duplicated().any():
        raise ValueError("History feature store contains duplicate applicant IDs")
    if not history.columns.drop("SK_ID_CURR").tolist():
        raise ValueError("History feature store has no model features")

    validation = validation_ids.merge(
        application, on="SK_ID_CURR", how="left", validate="one_to_one"
    ).merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    if len(validation) != len(validation_ids):
        raise ValueError("Validation records do not match split assignments")
    if validation["TARGET"].isna().any():
        raise ValueError("Application rows are missing for validation applicants")
    return validation


def define_full_history_decision_policy(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Lock the cost-derived threshold and report validation diagnostics."""
    config = load_modeling_config(config_path)
    policy = config["decision_policy"]
    false_negative_cost = float(policy["false_negative_cost"])
    false_positive_cost = float(policy["false_positive_cost"])
    locked_threshold = theoretical_cost_threshold(false_negative_cost, false_positive_cost)
    history_filename = str(config["feature_store"]["output_file"])

    calibrated_path = model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name
    if not calibrated_path.exists():
        raise FileNotFoundError("Run `risklens calibrate-full-history` before defining policy")
    model = joblib.load(calibrated_path)
    validation = load_full_history_validation_data(history_filename=history_filename)
    targets = validation.pop("TARGET").astype(int).to_numpy()
    probabilities = model.predict_proba(validation)[:, 1]

    grid_config = policy["threshold_grid"]
    thresholds = np.arange(
        float(grid_config["minimum"]),
        float(grid_config["maximum"]) + float(grid_config["step"]) / 2,
        float(grid_config["step"]),
    )
    table = build_threshold_table(
        targets,
        probabilities,
        thresholds,
        false_negative_cost,
        false_positive_cost,
    )
    empirical_best = table.sort_values(
        ["cost_per_application", "recall", "threshold"],
        ascending=[True, False, True],
    ).iloc[0]
    locked_metrics = threshold_metrics(
        targets,
        probabilities,
        locked_threshold,
        false_negative_cost,
        false_positive_cost,
    )
    report = {
        "model": "full_history_xgboost_calibrated",
        "model_scope": "application_plus_full_history",
        "selection_rule": policy["selection_rule"],
        "cost_assumption_status": "hypothetical_portfolio_assumption",
        "false_negative_cost": false_negative_cost,
        "false_positive_cost": false_positive_cost,
        "locked_threshold": round(locked_threshold, 6),
        "locked_threshold_metrics": locked_metrics,
        "validation_probability_metrics": evaluate_probabilities(
            targets, probabilities, locked_threshold
        ),
        "empirical_best_threshold_diagnostic": {
            key: float(value) if isinstance(value, (float, np.floating)) else int(value)
            for key, value in empirical_best.to_dict().items()
        },
        "data_policy": "costs_define_threshold_validation_diagnostic_holdout_sealed",
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / FULL_HISTORY_THRESHOLD_TABLE_PATH.name, index=False)
    (metrics_dir / FULL_HISTORY_POLICY_PATH.name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
