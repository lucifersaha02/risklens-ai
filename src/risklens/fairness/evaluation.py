"""Validation-only responsible-AI and subgroup diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from risklens.config import METRICS_DIR, MODEL_DIR, REPORT_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.calibration import CALIBRATED_MODEL_PATH
from risklens.modeling.decision import load_validation_split, theoretical_cost_threshold


def subgroup_table(
    frame: pd.DataFrame,
    group_column: str,
    threshold: float,
    minimum_group_size: int,
    target_column: str = "TARGET",
    probability_column: str = "probability",
) -> pd.DataFrame:
    """Calculate model-quality and operating metrics for each subgroup."""
    required = {group_column, target_column, probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Subgroup frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for group, data in frame.groupby(group_column, dropna=False, observed=True):
        targets = data[target_column].astype(int).to_numpy()
        probabilities = data[probability_column].astype(float).to_numpy()
        predictions = (probabilities >= threshold).astype(int)
        positives = targets == 1
        negatives = targets == 0
        true_positive = int((predictions[positives] == 1).sum())
        false_negative = int((predictions[positives] == 0).sum())
        false_positive = int((predictions[negatives] == 1).sum())
        true_negative = int((predictions[negatives] == 0).sum())
        roc_auc = (
            float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) == 2 else None
        )
        prevalence = float(targets.mean())
        mean_probability = float(probabilities.mean())
        rows.append(
            {
                "group_column": group_column,
                "group": str(group),
                "rows": int(len(data)),
                "meets_minimum_group_size": bool(len(data) >= minimum_group_size),
                "observed_positive_rate": round(prevalence, 6),
                "mean_predicted_probability": round(mean_probability, 6),
                "absolute_calibration_gap": round(abs(mean_probability - prevalence), 6),
                "brier_score": round(float(brier_score_loss(targets, probabilities)), 6),
                "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
                "recall": round(true_positive / max(true_positive + false_negative, 1), 6),
                "false_positive_rate": round(
                    false_positive / max(false_positive + true_negative, 1), 6
                ),
                "review_or_decline_rate": round(float(predictions.mean()), 6),
                "approval_rate": round(float(1 - predictions.mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("group", ignore_index=True)


def disparity_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Report max-minus-min gaps for sufficiently large diagnostic groups."""
    eligible = table[table["meets_minimum_group_size"]]
    metrics = (
        "observed_positive_rate",
        "mean_predicted_probability",
        "absolute_calibration_gap",
        "brier_score",
        "roc_auc",
        "recall",
        "false_positive_rate",
        "review_or_decline_rate",
        "approval_rate",
    )
    gaps: dict[str, float | None] = {}
    for metric in metrics:
        values = eligible[metric].dropna().astype(float)
        gaps[f"{metric}_max_min_gap"] = (
            round(float(values.max() - values.min()), 6) if len(values) >= 2 else None
        )
    return {
        "eligible_groups": int(len(eligible)),
        "excluded_small_groups": int(len(table) - len(eligible)),
        "gaps": gaps,
    }


def write_fairness_report(
    tables: dict[str, pd.DataFrame], report: dict[str, Any], path: Path
) -> None:
    """Write a human-readable responsible-AI diagnostic report."""
    lines = [
        "# RiskLens AI Responsible-AI Diagnostic",
        "",
        f"Decision threshold: `{report['decision_threshold']:.4f}`",
        "",
        "This report evaluates model behavior across selected subgroups on validation data. "
        "It is diagnostic evidence, not proof of legal or ethical fairness.",
        "",
    ]
    for name, table in tables.items():
        lines.extend(
            [
                f"## {name}",
                "",
                (
                    "| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | "
                    "Approval | Calibration gap |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in table.to_dict(orient="records"):
            auc = "NA" if pd.isna(row["roc_auc"]) else f"{row['roc_auc']:.3f}"
            lines.append(
                f"| {row['group']} | {row['rows']:,} | "
                f"{row['observed_positive_rate']:.2%} | {auc} | "
                f"{row['recall']:.2%} | {row['false_positive_rate']:.2%} | "
                f"{row['approval_rate']:.2%} | "
                f"{row['absolute_calibration_gap']:.2%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Limitations",
            "",
            "- The dataset is historical and may encode past structural inequalities.",
            "- Gender and age diagnostics do not cover every protected or vulnerable group.",
            "- Differences in base rates complicate direct parity comparisons.",
            "- Subgroup metrics do not establish causality or legal compliance.",
            "- A real lender would require jurisdiction-specific governance and review.",
            "- Final holdout data was not accessed for this diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_responsible_ai(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    report_dir: Path = REPORT_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Run validation-only subgroup diagnostics for gender and age bands."""
    config = load_modeling_config(config_path)
    policy = config["decision_policy"]
    responsible_config = config["responsible_ai"]
    threshold = theoretical_cost_threshold(
        float(policy["false_negative_cost"]),
        float(policy["false_positive_cost"]),
    )
    model_path = model_dir / CALIBRATED_MODEL_PATH.name
    if not model_path.exists():
        raise FileNotFoundError("Run `risklens calibrate-model` before fairness evaluation")
    model = joblib.load(model_path)
    validation = load_validation_split()
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
    report = {
        "model": "application_xgboost_calibrated",
        "dataset_split": "validation",
        "decision_threshold": round(threshold, 6),
        "minimum_group_size": minimum_group_size,
        "diagnostics": {group: disparity_summary(table) for group, table in tables.items()},
        "interpretation": "diagnostic_not_proof_of_fairness_or_legal_compliance",
        "holdout_accessed": False,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for group, table in tables.items():
        table.to_csv(metrics_dir / f"subgroup_{group.lower()}.csv", index=False)
    (metrics_dir / "responsible_ai_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_fairness_report(tables, report, report_dir / "responsible_ai_report.md")
    return report
