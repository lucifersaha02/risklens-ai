"""Responsible-AI diagnostics for the calibrated full-history model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from risklens.config import METRICS_DIR, MODEL_DIR, REPORT_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.fairness.evaluation import (
    disparity_summary,
    subgroup_table,
    write_fairness_report,
)
from risklens.modeling.decision import theoretical_cost_threshold
from risklens.modeling.full_history_calibration import (
    FULL_HISTORY_CALIBRATED_MODEL_PATH,
)
from risklens.modeling.full_history_decision import load_full_history_validation_data

FULL_HISTORY_FAIRNESS_SUMMARY_PATH = METRICS_DIR / "full_history_responsible_ai_summary.json"
FULL_HISTORY_FAIRNESS_REPORT_PATH = REPORT_DIR / "full_history_responsible_ai_report.md"


def compare_disparity_gaps(
    application: dict[str, Any], full_history: dict[str, Any]
) -> dict[str, dict[str, float | None]]:
    """Report signed changes in full-history max-minus-min subgroup gaps."""
    comparison: dict[str, dict[str, float | None]] = {}
    for group, diagnostic in full_history["diagnostics"].items():
        application_gaps = application.get("diagnostics", {}).get(group, {}).get("gaps", {})
        group_comparison: dict[str, float | None] = {}
        for metric, full_history_value in diagnostic["gaps"].items():
            application_value = application_gaps.get(metric)
            group_comparison[metric] = (
                round(float(full_history_value) - float(application_value), 6)
                if full_history_value is not None and application_value is not None
                else None
            )
        comparison[group] = group_comparison
    return comparison


def evaluate_full_history_responsible_ai(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    report_dir: Path = REPORT_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Run validation-only diagnostics for the selected full-history model."""
    config = load_modeling_config(config_path)
    policy = config["decision_policy"]
    responsible_config = config["responsible_ai"]
    threshold = theoretical_cost_threshold(
        float(policy["false_negative_cost"]),
        float(policy["false_positive_cost"]),
    )
    model_path = model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name
    if not model_path.exists():
        raise FileNotFoundError("Run `risklens calibrate-full-history` before fairness evaluation")
    model = joblib.load(model_path)
    validation = load_full_history_validation_data(
        history_filename=str(config["feature_store"]["output_file"])
    )
    scoring_frame = validation.drop(columns="TARGET")
    validation["probability"] = model.predict_proba(scoring_frame)[:, 1]
    validation["CODE_GENDER"] = validation["CODE_GENDER"].fillna("Missing")
    validation["AGE_BAND"] = pd.cut(
        -validation["DAYS_BIRTH"] / 365.25,
        bins=[float(value) for value in responsible_config["age_bins"]],
        labels=[str(value) for value in responsible_config["age_labels"]],
        include_lowest=True,
        right=False,
    )
    minimum_group_size = int(responsible_config["minimum_group_size"])
    tables = {
        group: subgroup_table(validation, group, threshold, minimum_group_size)
        for group in responsible_config["diagnostic_groups"]
    }
    report: dict[str, Any] = {
        "model": "full_history_xgboost_calibrated",
        "model_scope": "application_plus_full_history",
        "dataset_split": "validation",
        "decision_threshold": round(threshold, 6),
        "minimum_group_size": minimum_group_size,
        "diagnostics": {group: disparity_summary(table) for group, table in tables.items()},
        "interpretation": "diagnostic_not_proof_of_fairness_or_legal_compliance",
        "holdout_accessed": False,
    }

    application_path = metrics_dir / "responsible_ai_summary.json"
    if application_path.exists():
        application_report = json.loads(application_path.read_text(encoding="utf-8"))
        report["gap_change_vs_application_only"] = compare_disparity_gaps(
            application_report, report
        )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for group, table in tables.items():
        table.to_csv(metrics_dir / f"full_history_subgroup_{group.lower()}.csv", index=False)
    (metrics_dir / FULL_HISTORY_FAIRNESS_SUMMARY_PATH.name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_fairness_report(
        tables,
        report,
        report_dir / FULL_HISTORY_FAIRNESS_REPORT_PATH.name,
        title="RiskLens AI Full-History Responsible-AI Diagnostic",
    )
    return report
