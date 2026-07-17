"""Documented cost-sensitive decision threshold policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risklens.config import METRICS_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.calibration import CALIBRATED_MODEL_PATH
from risklens.modeling.metrics import evaluate_probabilities


def theoretical_cost_threshold(
    false_negative_cost: float,
    false_positive_cost: float,
) -> float:
    """Return the Bayes threshold for calibrated probabilities and fixed costs."""
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Misclassification costs must be positive")
    return false_positive_cost / (false_negative_cost + false_positive_cost)


def threshold_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, Any]:
    """Calculate confusion counts, operating rates, and normalized business cost."""
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    true_positive = int(((targets == 1) & (predictions == 1)).sum())
    true_negative = int(((targets == 0) & (predictions == 0)).sum())
    false_positive = int(((targets == 0) & (predictions == 1)).sum())
    false_negative = int(((targets == 1) & (predictions == 0)).sum())
    total = len(targets)
    cost = false_negative * false_negative_cost + false_positive * false_positive_cost

    return {
        "threshold": round(float(threshold), 6),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "recall": round(true_positive / max(true_positive + false_negative, 1), 6),
        "precision": round(true_positive / max(true_positive + false_positive, 1), 6),
        "specificity": round(true_negative / max(true_negative + false_positive, 1), 6),
        "approval_rate": round((true_negative + false_negative) / max(total, 1), 6),
        "review_or_decline_rate": round((true_positive + false_positive) / max(total, 1), 6),
        "total_cost_units": round(float(cost), 6),
        "cost_per_application": round(float(cost / max(total, 1)), 6),
    }


def build_threshold_table(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> pd.DataFrame:
    """Build a validation sensitivity table across possible thresholds."""
    rows = [
        threshold_metrics(
            targets,
            probabilities,
            float(threshold),
            false_negative_cost,
            false_positive_cost,
        )
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def load_validation_split(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load validation applicants without accessing calibration or holdout."""
    assignments = pd.read_parquet(processed_dir / "split_assignments.parquet")
    validation_ids = assignments.loc[assignments["split"] == "validation", ["SK_ID_CURR"]]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    validation = application.merge(
        validation_ids,
        on="SK_ID_CURR",
        how="inner",
        validate="one_to_one",
    )
    if len(validation) != len(validation_ids):
        raise ValueError("Validation records do not match split assignments")
    return validation


def define_decision_policy(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Lock a cost-derived threshold and report validation sensitivity."""
    config = load_modeling_config(config_path)
    policy = config["decision_policy"]
    false_negative_cost = float(policy["false_negative_cost"])
    false_positive_cost = float(policy["false_positive_cost"])
    locked_threshold = theoretical_cost_threshold(false_negative_cost, false_positive_cost)

    calibrated_path = model_dir / CALIBRATED_MODEL_PATH.name
    if not calibrated_path.exists():
        raise FileNotFoundError("Run `risklens calibrate-model` before defining policy")
    model = joblib.load(calibrated_path)
    validation = load_validation_split()
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
    probability_metrics = evaluate_probabilities(targets, probabilities, locked_threshold)
    report = {
        "model": "application_xgboost_calibrated",
        "selection_rule": policy["selection_rule"],
        "false_negative_cost": false_negative_cost,
        "false_positive_cost": false_positive_cost,
        "locked_threshold": round(locked_threshold, 6),
        "locked_threshold_metrics": locked_metrics,
        "validation_probability_metrics": probability_metrics,
        "empirical_best_threshold_diagnostic": {
            key: float(value) if isinstance(value, (float, np.floating)) else int(value)
            for key, value in empirical_best.to_dict().items()
        },
        "data_policy": "costs_define_threshold_validation_is_diagnostic_holdout_sealed",
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "threshold_sensitivity.csv", index=False)
    (metrics_dir / "decision_policy.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
