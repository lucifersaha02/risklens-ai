"""One-time final holdout evaluation for the frozen governed model."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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
    REPORT_DIR,
)
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.fairness.evaluation import disparity_summary, subgroup_table
from risklens.governance.model_card import MODEL_FREEZE_PATH, sha256_file
from risklens.modeling.decision import threshold_metrics
from risklens.modeling.full_history_calibration import (
    FULL_HISTORY_CALIBRATED_MODEL_PATH,
)
from risklens.modeling.metrics import evaluate_probabilities

FINAL_HOLDOUT_METRICS_PATH = METRICS_DIR / "final_holdout_metrics.json"
FINAL_HOLDOUT_REPORT_PATH = REPORT_DIR / "final_holdout_report.md"
BOOTSTRAP_REPLICATES = 500
CONFIDENCE_LEVEL = 0.95


def verify_frozen_artifacts(
    report_dir: Path = REPORT_DIR,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the freeze status and every recorded artifact digest."""
    root = project_root or report_dir.parent
    freeze_path = report_dir / MODEL_FREEZE_PATH.name
    if not freeze_path.exists():
        raise FileNotFoundError("Run `risklens build-model-card` before holdout evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("holdout_accessed") is not False:
        raise RuntimeError("Final holdout has already been accessed; repeat evaluation is blocked")
    if freeze.get("release_status") != "pre_holdout_frozen_research_prototype":
        raise RuntimeError("Model is not in the required pre-holdout frozen state")
    if freeze.get("post_holdout_tuning_permitted") is not False:
        raise RuntimeError("Freeze manifest does not prohibit post-holdout tuning")

    for name, artifact in freeze["artifacts"].items():
        path = root / Path(str(artifact["path"]))
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact is missing: {name}")
        actual_hash = sha256_file(path)
        if actual_hash != artifact["sha256"]:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")
    return freeze


def load_full_history_holdout_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    history_filename: str = "full_history_features.parquet",
) -> pd.DataFrame:
    """Load exactly the applicants assigned to the final holdout split."""
    assignments_path = processed_dir / "split_assignments.parquet"
    history_path = interim_dir / history_filename
    if not assignments_path.exists() or not history_path.exists():
        raise FileNotFoundError("Required split or history artifacts are missing")
    assignments = pd.read_parquet(assignments_path)
    holdout_ids = assignments.loc[assignments["split"] == "holdout", ["SK_ID_CURR"]]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    history = pd.read_parquet(history_path)
    if "TARGET" in history.columns:
        raise ValueError("History feature store must never contain TARGET")
    if history["SK_ID_CURR"].duplicated().any():
        raise ValueError("History feature store contains duplicate applicant IDs")

    holdout = holdout_ids.merge(
        application, on="SK_ID_CURR", how="left", validate="one_to_one"
    ).merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    if len(holdout) != len(holdout_ids):
        raise ValueError("Holdout records do not match split assignments")
    if holdout["TARGET"].isna().any():
        raise ValueError("Application rows are missing for holdout applicants")
    return holdout


def bootstrap_confidence_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = CONFIDENCE_LEVEL,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Calculate deterministic non-parametric percentile confidence intervals."""
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(targets) != len(probabilities) or len(targets) == 0:
        raise ValueError("Bootstrap inputs must have equal non-zero length")
    if replicates <= 0 or not 0 < confidence_level < 1:
        raise ValueError("Bootstrap configuration is invalid")
    random = np.random.default_rng(random_seed)
    values: dict[str, list[float]] = {
        metric: []
        for metric in (
            "roc_auc",
            "average_precision",
            "brier_score",
            "log_loss",
            "recall",
            "precision",
            "approval_rate",
            "cost_per_application",
        )
    }
    for _ in range(replicates):
        indices = random.integers(0, len(targets), size=len(targets))
        sampled_targets = targets[indices]
        sampled_probabilities = probabilities[indices]
        if len(np.unique(sampled_targets)) < 2:
            continue
        probability_metrics = evaluate_probabilities(
            sampled_targets, sampled_probabilities, threshold
        )
        operating_metrics = threshold_metrics(
            sampled_targets,
            sampled_probabilities,
            threshold,
            false_negative_cost,
            false_positive_cost,
        )
        for metric in ("roc_auc", "average_precision", "brier_score", "log_loss"):
            values[metric].append(float(probability_metrics[metric]))
        for metric in ("recall", "precision", "approval_rate", "cost_per_application"):
            values[metric].append(float(operating_metrics[metric]))

    alpha = (1 - confidence_level) / 2
    intervals = {}
    for metric, samples in values.items():
        if not samples:
            raise ValueError("Bootstrap did not produce valid samples")
        intervals[metric] = {
            "lower": round(float(np.quantile(samples, alpha)), 6),
            "upper": round(float(np.quantile(samples, 1 - alpha)), 6),
        }
    return intervals


def _render_holdout_report(report: dict[str, Any]) -> str:
    """Render the immutable final evaluation evidence."""
    metrics = report["probability_metrics"]
    operating = report["locked_policy_metrics"]
    intervals = report["confidence_intervals"]
    lines = [
        "# RiskLens AI — One-Time Final Holdout Evaluation",
        "",
        "**The final holdout has been accessed. Model development is permanently closed.**",
        "",
        f"Evaluation timestamp (UTC): `{report['evaluated_at_utc']}`",
        "",
        f"Applicants: {report['holdout_rows']:,}",
        "",
        "## Frozen model performance",
        "",
        "| Metric | Point estimate | 95% bootstrap interval |",
        "|---|---:|---:|",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} | "
        f"[{intervals['roc_auc']['lower']:.4f}, {intervals['roc_auc']['upper']:.4f}] |",
        f"| PR-AUC | {metrics['average_precision']:.4f} | "
        f"[{intervals['average_precision']['lower']:.4f}, "
        f"{intervals['average_precision']['upper']:.4f}] |",
        f"| Brier score | {metrics['brier_score']:.5f} | "
        f"[{intervals['brier_score']['lower']:.5f}, "
        f"{intervals['brier_score']['upper']:.5f}] |",
        f"| Log loss | {metrics['log_loss']:.5f} | "
        f"[{intervals['log_loss']['lower']:.5f}, {intervals['log_loss']['upper']:.5f}] |",
        "",
        "## Locked policy performance",
        "",
        f"Threshold: `{report['locked_threshold']:.6f}` using the frozen hypothetical "
        "5:1 false-negative:false-positive cost assumption.",
        "",
        "| Metric | Point estimate | 95% bootstrap interval |",
        "|---|---:|---:|",
        f"| Recall | {operating['recall']:.2%} | "
        f"[{intervals['recall']['lower']:.2%}, {intervals['recall']['upper']:.2%}] |",
        f"| Precision | {operating['precision']:.2%} | "
        f"[{intervals['precision']['lower']:.2%}, {intervals['precision']['upper']:.2%}] |",
        f"| Approval rate | {operating['approval_rate']:.2%} | "
        f"[{intervals['approval_rate']['lower']:.2%}, "
        f"{intervals['approval_rate']['upper']:.2%}] |",
        f"| Cost units/application | {operating['cost_per_application']:.4f} | "
        f"[{intervals['cost_per_application']['lower']:.4f}, "
        f"{intervals['cost_per_application']['upper']:.4f}] |",
        "",
        "## Governance statement",
        "",
        "- Frozen artifact hashes were verified before loading holdout outcomes.",
        "- Calibration and threshold were not refitted or selected on holdout data.",
        "- Results must not trigger feature, hyperparameter, calibration, or threshold tuning.",
        "- Subgroup results remain diagnostic and do not establish fairness or compliance.",
        "- This remains a research prototype, not a production lending system.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_final_holdout(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    report_dir: Path = REPORT_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the freeze and evaluate the final holdout exactly once."""
    freeze = verify_frozen_artifacts(report_dir=report_dir)
    metrics_path = metrics_dir / FINAL_HOLDOUT_METRICS_PATH.name
    report_path = report_dir / FINAL_HOLDOUT_REPORT_PATH.name
    if metrics_path.exists() or report_path.exists():
        raise RuntimeError("Final holdout result already exists; repeat evaluation is blocked")

    config = load_modeling_config(config_path)
    model_path = model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name
    model = joblib.load(model_path)
    holdout = load_full_history_holdout_data(
        history_filename=str(config["feature_store"]["output_file"])
    )
    targets = holdout.pop("TARGET").astype(int).to_numpy()
    probabilities = model.predict_proba(holdout)[:, 1]
    threshold = float(freeze["locked_threshold"])
    policy = config["decision_policy"]
    false_negative_cost = float(policy["false_negative_cost"])
    false_positive_cost = float(policy["false_positive_cost"])

    probability_metrics = evaluate_probabilities(targets, probabilities, threshold)
    operating_metrics = threshold_metrics(
        targets,
        probabilities,
        threshold,
        false_negative_cost,
        false_positive_cost,
    )
    intervals = bootstrap_confidence_intervals(
        targets,
        probabilities,
        threshold,
        false_negative_cost,
        false_positive_cost,
        random_seed=int(config["random_seed"]),
    )

    audit = holdout.copy()
    audit["TARGET"] = targets
    audit["probability"] = probabilities
    audit["CODE_GENDER"] = audit["CODE_GENDER"].fillna("Missing")
    responsible = config["responsible_ai"]
    audit["AGE_BAND"] = pd.cut(
        -audit["DAYS_BIRTH"] / 365.25,
        bins=[float(value) for value in responsible["age_bins"]],
        labels=[str(value) for value in responsible["age_labels"]],
        include_lowest=True,
        right=False,
    )
    minimum_group_size = int(responsible["minimum_group_size"])
    subgroup_diagnostics = {}
    for group in responsible["diagnostic_groups"]:
        table = subgroup_table(audit, str(group), threshold, minimum_group_size)
        subgroup_diagnostics[str(group)] = {
            **disparity_summary(table),
            "groups": table.to_dict(orient="records"),
        }

    evaluated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    report: dict[str, Any] = {
        "model": freeze["model"],
        "release_status": "final_holdout_evaluated_research_prototype",
        "evaluated_at_utc": evaluated_at,
        "holdout_rows": int(len(holdout)),
        "locked_threshold": threshold,
        "calibration_method": freeze["calibration_method"],
        "probability_metrics": probability_metrics,
        "locked_policy_metrics": operating_metrics,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_intervals": intervals,
        "subgroup_diagnostics": subgroup_diagnostics,
        "frozen_artifact_hashes_verified": True,
        "post_holdout_tuning_permitted": False,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(_render_holdout_report(report), encoding="utf-8")
    freeze["release_status"] = "final_holdout_evaluated_research_prototype"
    freeze["holdout_accessed"] = True
    freeze["holdout_accessed_at_utc"] = evaluated_at
    (report_dir / MODEL_FREEZE_PATH.name).write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return report
