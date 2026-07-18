"""SHAP explanations for the selected calibrated full-history model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risklens.config import FIGURE_DIR, METRICS_DIR, MODEL_DIR, REPORT_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.full_history_calibration import (
    FULL_HISTORY_CALIBRATED_MODEL_PATH,
)
from risklens.modeling.full_history_decision import load_full_history_validation_data

GLOBAL_IMPORTANCE_PATH = METRICS_DIR / "full_history_shap_global_importance.csv"
LOCAL_EXPLANATIONS_PATH = METRICS_DIR / "full_history_local_explanations.json"
SHAP_SUMMARY_PATH = METRICS_DIR / "full_history_shap_summary.json"
SHAP_BAR_PLOT_PATH = FIGURE_DIR / "full_history_shap_bar.png"
SHAP_BEESWARM_PATH = FIGURE_DIR / "full_history_shap_beeswarm.png"
SHAP_REPORT_PATH = REPORT_DIR / "shap_explainability_report.md"


def normalize_shap_values(values: Any) -> np.ndarray:
    """Normalize common binary-class SHAP outputs to rows by features."""
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError("Expected binary-class SHAP values")
        values = values[1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        if array.shape[2] != 2:
            raise ValueError("Expected two output classes in SHAP values")
        array = array[:, :, 1]
    if array.ndim != 2:
        raise ValueError("SHAP values must have rows and features")
    return array


def clean_feature_name(name: str) -> str:
    """Remove sklearn transformer prefixes while preserving encoded categories."""
    return name.split("__", maxsplit=1)[-1]


def reason_codes(
    feature_names: list[str] | np.ndarray,
    shap_row: np.ndarray,
    feature_values: np.ndarray,
    top_n: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Return strongest risk-increasing and risk-reducing local contributions."""
    names = np.asarray(feature_names, dtype=str)
    contributions = np.asarray(shap_row, dtype=float)
    values = np.asarray(feature_values, dtype=float)
    if not (len(names) == len(contributions) == len(values)):
        raise ValueError("Feature names, values, and SHAP values must align")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    def records(indices: np.ndarray, direction: str) -> list[dict[str, Any]]:
        return [
            {
                "feature": clean_feature_name(str(names[index])),
                "transformed_value": round(float(values[index]), 6),
                "shap_value": round(float(contributions[index]), 6),
                "direction": direction,
            }
            for index in indices[:top_n]
        ]

    increasing = np.flatnonzero(contributions > 0)
    increasing = increasing[np.argsort(contributions[increasing])[::-1]]
    reducing = np.flatnonzero(contributions < 0)
    reducing = reducing[np.argsort(contributions[reducing])]
    return {
        "risk_increasing": records(increasing, "increases_raw_model_risk"),
        "risk_reducing": records(reducing, "reduces_raw_model_risk"),
    }


def global_importance_table(
    feature_names: list[str] | np.ndarray, shap_values: np.ndarray
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP contribution."""
    names = np.asarray(feature_names, dtype=str)
    values = normalize_shap_values(shap_values)
    if values.shape[1] != len(names):
        raise ValueError("Feature names do not match SHAP columns")
    table = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in names],
            "mean_absolute_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
        }
    )
    return table.sort_values("mean_absolute_shap", ascending=False, ignore_index=True)


def _dense_row(matrix: Any, index: int) -> np.ndarray:
    """Convert one transformed sparse or dense row to a flat dense array."""
    row = matrix[index]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row, dtype=float).reshape(-1)


def _write_plots(
    importance: pd.DataFrame,
    shap_values: np.ndarray,
    transformed: Any,
    feature_names: np.ndarray,
    top_features: int,
    figure_dir: Path,
) -> None:
    """Write a global importance bar chart and SHAP beeswarm plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    figure_dir.mkdir(parents=True, exist_ok=True)
    displayed = importance.head(top_features).sort_values("mean_absolute_shap")
    _, axis = plt.subplots(figsize=(10, 8))
    axis.barh(displayed["feature"], displayed["mean_absolute_shap"])
    axis.set_xlabel("Mean absolute SHAP value (raw XGBoost margin)")
    axis.set_title("RiskLens AI — Global Feature Importance")
    plt.tight_layout()
    plt.savefig(figure_dir / SHAP_BAR_PLOT_PATH.name, dpi=160, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=[clean_feature_name(name) for name in feature_names],
        max_display=top_features,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / SHAP_BEESWARM_PATH.name, dpi=160, bbox_inches="tight")
    plt.close()


def _write_report(report: dict[str, Any], importance: pd.DataFrame, path: Path) -> None:
    """Write a concise model-explanation report with governance limitations."""
    lines = [
        "# RiskLens AI SHAP Explainability Report",
        "",
        "## Scope",
        "",
        f"- Model: `{report['model']}`",
        f"- Dataset split: `{report['dataset_split']}`",
        f"- Explanation sample: {report['sample_rows']:,} applicants",
        "- Final holdout accessed: **No**",
        "",
        "SHAP values explain the underlying XGBoost raw margin. The displayed final risk "
        "probability is subsequently transformed by sigmoid calibration, so SHAP values "
        "must not be interpreted as percentage-point probability changes.",
        "",
        "## Leading global drivers",
        "",
        "| Rank | Feature | Mean absolute SHAP | Mean signed SHAP |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(importance.head(15).itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.feature} | {row.mean_absolute_shap:.6f} | "
            f"{row.mean_signed_shap:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- SHAP describes this model's behavior; it does not establish causality.",
            "- Correlated features can share or redistribute attribution.",
            "- One-hot and engineered features require domain-aware interpretation.",
            "- Reason codes support human review and are not autonomous lending decisions.",
            "- Protected attributes require governance even when they are not top drivers.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_full_history_shap_explanations(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    figure_dir: Path = FIGURE_DIR,
    report_dir: Path = REPORT_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Create global and local SHAP evidence without accessing final holdout."""
    import shap

    config = load_modeling_config(config_path)
    explainability = config["explainability"]
    random_seed = int(config["random_seed"])
    model_path = model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name
    if not model_path.exists():
        raise FileNotFoundError("Run `risklens calibrate-full-history` before SHAP analysis")
    calibrated_model = joblib.load(model_path)
    pipeline = calibrated_model.base_model
    validation = load_full_history_validation_data(
        history_filename=str(config["feature_store"]["output_file"])
    )
    sample_size = min(int(explainability["validation_sample_size"]), len(validation))
    sample = validation.sample(n=sample_size, random_state=random_seed).copy()
    targets = sample.pop("TARGET").astype(int)

    engineered = pipeline.named_steps["features"].transform(sample)
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(engineered)
    feature_names = preprocessor.get_feature_names_out()
    xgboost_model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(xgboost_model)
    values = normalize_shap_values(explainer.shap_values(transformed))
    importance = global_importance_table(feature_names, values)

    raw_probabilities = pipeline.predict_proba(sample)[:, 1]
    calibrated_probabilities = calibrated_model.predict_proba(sample)[:, 1]
    raw_margins = xgboost_model.predict(transformed, output_margin=True)
    expected_value = np.asarray(explainer.expected_value, dtype=float).reshape(-1)[-1]
    reconstructed_margins = expected_value + values.sum(axis=1)
    maximum_additivity_error = float(
        np.max(np.abs(np.asarray(raw_margins) - reconstructed_margins))
    )

    local_count = min(int(explainability["local_applicants"]), sample_size)
    selected_indices = np.argsort(calibrated_probabilities)[-local_count:][::-1]
    local_explanations = []
    reason_count = int(explainability["local_reason_codes_per_direction"])
    for index in selected_indices:
        explanation = {
            "SK_ID_CURR": int(sample.iloc[index]["SK_ID_CURR"]),
            "observed_target": int(targets.iloc[index]),
            "raw_model_probability": round(float(raw_probabilities[index]), 6),
            "calibrated_probability": round(float(calibrated_probabilities[index]), 6),
            "raw_model_margin": round(float(raw_margins[index]), 6),
            "reason_codes": reason_codes(
                feature_names,
                values[index],
                _dense_row(transformed, int(index)),
                top_n=reason_count,
            ),
        }
        local_explanations.append(explanation)

    report: dict[str, Any] = {
        "model": "full_history_xgboost_calibrated",
        "dataset_split": "validation",
        "sample_rows": sample_size,
        "transformed_feature_count": int(len(feature_names)),
        "local_explanations": int(len(local_explanations)),
        "calibration_method": calibrated_model.method,
        "shap_output_space": "raw_xgboost_margin_before_probability_calibration",
        "maximum_shap_additivity_error": maximum_additivity_error,
        "holdout_accessed": False,
        "interpretation": "model_behavior_not_causality_or_credit_adverse_action_notice",
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(metrics_dir / GLOBAL_IMPORTANCE_PATH.name, index=False)
    (metrics_dir / LOCAL_EXPLANATIONS_PATH.name).write_text(
        json.dumps(local_explanations, indent=2), encoding="utf-8"
    )
    (metrics_dir / SHAP_SUMMARY_PATH.name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_plots(
        importance,
        values,
        transformed,
        feature_names,
        int(explainability["global_top_features"]),
        figure_dir,
    )
    _write_report(report, importance, report_dir / SHAP_REPORT_PATH.name)
    return report
