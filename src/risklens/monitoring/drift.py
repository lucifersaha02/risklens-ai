"""Reference-based feature and prediction drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from risklens.config import (
    CONFIG_DIR,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORT_DIR,
)
from risklens.modeling.full_history_decision import load_full_history_validation_data
from risklens.serving.inference import FrozenRiskScorer

MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring.yaml"
MONITORING_BASELINE_PATH = PROCESSED_DATA_DIR / "monitoring_baseline.json"
MONITORING_METRICS_PATH = METRICS_DIR / "test_population_monitoring.json"
MONITORING_REPORT_PATH = REPORT_DIR / "monitoring_report.md"


def load_monitoring_config(path: Path = MONITORING_CONFIG_PATH) -> dict[str, Any]:
    """Load the version-controlled monitoring policy."""
    if not path.exists():
        raise FileNotFoundError(f"Monitoring configuration not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Monitoring configuration must be a mapping")
    return config


def quantile_cutpoints(values: np.ndarray, bins: int = 10) -> list[float]:
    """Create stable internal cutpoints for continuous, binary, or constant data."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Cannot create bins from entirely non-finite values")
    unique = np.unique(finite)
    if len(unique) == 1:
        return []
    if len(unique) <= bins:
        return [float(value) for value in (unique[:-1] + unique[1:]) / 2]
    quantiles = np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])
    return [float(value) for value in np.unique(quantiles)]


def bin_proportions(values: np.ndarray, cutpoints: list[float]) -> np.ndarray:
    """Return normalized counts including tails and a separate non-finite bin."""
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    finite_bins = np.digitize(values[finite_mask], cutpoints, right=False)
    counts = np.bincount(finite_bins, minlength=len(cutpoints) + 1).astype(float)
    counts = np.append(counts, float((~finite_mask).sum()))
    return counts / max(float(len(values)), 1.0)


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI for two aligned discrete distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or expected.ndim != 1:
        raise ValueError("PSI distributions must be aligned one-dimensional arrays")
    if epsilon <= 0:
        raise ValueError("PSI epsilon must be positive")
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_severity(psi: float, warning_threshold: float, critical_threshold: float) -> str:
    """Map PSI to an operational alert level."""
    if not 0 <= warning_threshold < critical_threshold:
        raise ValueError("Drift thresholds must be ordered and non-negative")
    if psi >= critical_threshold:
        return "critical"
    if psi >= warning_threshold:
        return "warning"
    return "stable"


def _transformed_inputs(scorer: FrozenRiskScorer, frame: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Apply the exact frozen feature, governance, and preprocessing pipeline."""
    pipeline = scorer.pipeline
    engineered = pipeline.named_steps["features"].transform(frame)
    governed = pipeline.named_steps["governance"].transform(engineered)
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.transform(governed), preprocessor.get_feature_names_out()


def _column_values(matrix: Any, index: int) -> np.ndarray:
    """Extract one dense transformed feature without densifying the full matrix."""
    column = matrix[:, index]
    if hasattr(column, "toarray"):
        column = column.toarray()
    return np.asarray(column, dtype=float).reshape(-1)


def _distribution(values: np.ndarray, bins: int) -> dict[str, Any]:
    """Serialize one reference distribution."""
    cutpoints = quantile_cutpoints(values, bins=bins)
    return {
        "cutpoints": cutpoints,
        "proportions": [float(value) for value in bin_proportions(values, cutpoints)],
    }


def build_monitoring_baseline(
    baseline_path: Path = MONITORING_BASELINE_PATH,
    config_path: Path = MONITORING_CONFIG_PATH,
) -> dict[str, Any]:
    """Build a validation reference without touching or changing the model."""
    config = load_monitoring_config(config_path)
    scorer = FrozenRiskScorer()
    validation = load_full_history_validation_data().drop(columns="TARGET")
    transformed, feature_names = _transformed_inputs(scorer, validation)
    importances = np.asarray(scorer.pipeline.named_steps["model"].feature_importances_, dtype=float)
    maximum_features = min(int(config["reference"]["maximum_features"]), len(importances))
    selected_indices = np.argsort(importances)[-maximum_features:][::-1]
    bins = int(config["reference"]["quantile_bins"])
    feature_distributions = {}
    for index in selected_indices:
        feature_distributions[str(feature_names[index])] = _distribution(
            _column_values(transformed, int(index)), bins
        )
    probabilities = scorer.model.predict_proba(validation)[:, 1]
    prediction_distribution = _distribution(probabilities, bins)
    prediction_distribution["mean_probability"] = float(probabilities.mean())
    baseline = {
        "model": scorer.freeze["model"],
        "model_version": scorer.model_version,
        "reference_split": config["reference"]["dataset_split"],
        "reference_rows": int(len(validation)),
        "monitored_feature_count": int(len(feature_distributions)),
        "features": feature_distributions,
        "prediction": prediction_distribution,
        "target_used": False,
        "post_holdout_tuning_permitted": False,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def load_test_population(
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_data_dir: Path = INTERIM_DATA_DIR,
    history_filename: str = "full_history_features.parquet",
) -> pd.DataFrame:
    """Load the unlabeled competition test population with target-free history."""
    application = pd.read_csv(raw_data_dir / "application_test.csv")
    history = pd.read_parquet(interim_data_dir / history_filename)
    if "TARGET" in application.columns or "TARGET" in history.columns:
        raise ValueError("Monitoring population must not contain TARGET")
    if application["SK_ID_CURR"].duplicated().any() or history["SK_ID_CURR"].duplicated().any():
        raise ValueError("Monitoring inputs contain duplicate applicant IDs")
    population = application.merge(
        history,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    if len(population) != len(application):
        raise ValueError("Monitoring population row count changed during history join")
    return population


def _severity_rank(level: str) -> int:
    return {"stable": 0, "warning": 1, "critical": 2}[level]


def monitor_test_population(
    baseline_path: Path = MONITORING_BASELINE_PATH,
    metrics_path: Path = MONITORING_METRICS_PATH,
    report_path: Path = MONITORING_REPORT_PATH,
    config_path: Path = MONITORING_CONFIG_PATH,
) -> dict[str, Any]:
    """Compare the unlabeled test population with the frozen validation reference."""
    if not baseline_path.exists():
        raise FileNotFoundError("Run `risklens build-monitoring-baseline` first")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    config = load_monitoring_config(config_path)
    scorer = FrozenRiskScorer()
    if baseline["model_version"] != scorer.model_version:
        raise RuntimeError("Monitoring baseline model version does not match serving")
    population = load_test_population()
    transformed, feature_names = _transformed_inputs(scorer, population)
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}
    psi_config = config["population_stability_index"]
    warning_threshold = float(psi_config["warning_threshold"])
    critical_threshold = float(psi_config["critical_threshold"])
    epsilon = float(psi_config["epsilon"])

    features = []
    for name, reference in baseline["features"].items():
        if name not in name_to_index:
            raise ValueError(f"Monitored feature is missing from current pipeline: {name}")
        cutpoints = [float(value) for value in reference["cutpoints"]]
        actual = bin_proportions(_column_values(transformed, name_to_index[name]), cutpoints)
        psi = population_stability_index(
            np.asarray(reference["proportions"], dtype=float), actual, epsilon
        )
        features.append(
            {
                "feature": name.split("__", maxsplit=1)[-1],
                "psi": round(psi, 6),
                "severity": drift_severity(psi, warning_threshold, critical_threshold),
            }
        )
    features.sort(key=lambda item: item["psi"], reverse=True)

    probabilities = scorer.model.predict_proba(population)[:, 1]
    prediction_reference = baseline["prediction"]
    prediction_actual = bin_proportions(
        probabilities,
        [float(value) for value in prediction_reference["cutpoints"]],
    )
    prediction_psi = population_stability_index(
        np.asarray(prediction_reference["proportions"], dtype=float),
        prediction_actual,
        epsilon,
    )
    prediction_severity = drift_severity(prediction_psi, warning_threshold, critical_threshold)

    duplicate_rate = float(population["SK_ID_CURR"].duplicated().mean())
    quality_config = config["data_quality"]
    quality_alerts = []
    if len(population) < int(quality_config["minimum_batch_rows"]):
        quality_alerts.append("batch_below_minimum_rows")
    if duplicate_rate > float(quality_config["maximum_duplicate_id_rate"]):
        quality_alerts.append("duplicate_applicant_ids")
    if bool(quality_config["require_target_absent"]) and "TARGET" in population.columns:
        quality_alerts.append("target_leakage")

    severity_counts = {
        level: sum(item["severity"] == level for item in features)
        for level in ("stable", "warning", "critical")
    }
    overall_severity = prediction_severity
    if features:
        feature_max = max(features, key=lambda item: _severity_rank(item["severity"]))["severity"]
        if _severity_rank(feature_max) > _severity_rank(overall_severity):
            overall_severity = feature_max
    if quality_alerts:
        overall_severity = "critical"

    report = {
        "model": scorer.freeze["model"],
        "model_version": scorer.model_version,
        "reference_split": baseline["reference_split"],
        "current_population": "home_credit_application_test_unlabeled",
        "reference_rows": baseline["reference_rows"],
        "current_rows": int(len(population)),
        "overall_severity": overall_severity,
        "prediction_drift": {
            "psi": round(prediction_psi, 6),
            "severity": prediction_severity,
            "reference_mean_probability": round(float(prediction_reference["mean_probability"]), 6),
            "current_mean_probability": round(float(probabilities.mean()), 6),
        },
        "feature_severity_counts": severity_counts,
        "top_feature_drift": features[:20],
        "data_quality": {
            "duplicate_id_rate": duplicate_rate,
            "target_present": "TARGET" in population.columns,
            "alerts": quality_alerts,
        },
        "interpretation": config["interpretation"],
        "labels_available": False,
        "performance_drift_measured": False,
        "post_holdout_tuning_permitted": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_monitoring_report(report, report_path)
    return report


def _write_monitoring_report(report: dict[str, Any], path: Path) -> None:
    """Write a human-readable monitoring snapshot."""
    prediction = report["prediction_drift"]
    counts = report["feature_severity_counts"]
    lines = [
        "# RiskLens AI Monitoring Snapshot",
        "",
        f"Overall severity: **{report['overall_severity'].upper()}**",
        "",
        f"- Frozen model version: `{report['model_version']}`",
        f"- Reference: `{report['reference_split']}` ({report['reference_rows']:,} rows)",
        f"- Current population: `{report['current_population']}` ({report['current_rows']:,} rows)",
        f"- Prediction PSI: `{prediction['psi']:.4f}` ({prediction['severity']})",
        f"- Feature alerts: {counts['warning']} warning, {counts['critical']} critical",
        "- Labels available: No; performance drift was not measured.",
        "",
        "## Highest feature PSI",
        "",
        "| Feature | PSI | Severity |",
        "|---|---:|---|",
    ]
    for item in report["top_feature_drift"][:10]:
        lines.append(f"| {item['feature']} | {item['psi']:.4f} | {item['severity']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Alerts require investigation and do not authorize post-holdout model tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
