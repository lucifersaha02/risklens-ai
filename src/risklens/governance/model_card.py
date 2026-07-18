"""Reproducible model card and pre-holdout governance freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from risklens.config import METRICS_DIR, MODEL_DIR, REPORT_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.full_history import FULL_HISTORY_MODEL_PATH
from risklens.modeling.full_history_calibration import (
    FULL_HISTORY_CALIBRATED_MODEL_PATH,
)

MODEL_CARD_PATH = REPORT_DIR / "model_card.md"
MODEL_FREEZE_PATH = REPORT_DIR / "model_governance_freeze.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a potentially large artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_json(path: Path) -> dict[str, Any]:
    """Load a required JSON governance input with an actionable error."""
    if not path.exists():
        raise FileNotFoundError(f"Required governance evidence is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def render_model_card(context: dict[str, Any]) -> str:
    """Render the governed model evidence as a portfolio-ready model card."""
    candidate = context["candidate"]
    calibration = context["calibration"]
    policy = context["policy"]
    fairness = context["fairness"]
    shap_summary = context["shap_summary"]
    top_features = context["top_features"]
    governance = candidate["feature_governance"]
    validation = candidate["validation"]
    policy_metrics = policy["locked_threshold_metrics"]
    gender_gaps = fairness["diagnostics"]["CODE_GENDER"]["gaps"]
    age_gaps = fairness["diagnostics"]["AGE_BAND"]["gaps"]

    lines = [
        "# RiskLens AI — Governed Full-History Model Card",
        "",
        "## Release status",
        "",
        "**Pre-holdout frozen research prototype. Not approved for production lending.**",
        "",
        "This model estimates Home Credit default risk for portfolio demonstration and "
        "human decision-support research. It must not independently approve, decline, "
        "price, or otherwise determine access to credit.",
        "",
        "## Model overview",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Algorithm | XGBoost binary classifier |",
        "| Inputs | Application data plus aggregated credit history |",
        f"| Training applicants | {candidate['training_rows']:,} |",
        f"| Validation applicants | {candidate['validation_rows']:,} |",
        f"| Calibration applicants | {calibration['calibration_rows']:,} |",
        f"| Calibration method | {calibration['selected_method']} |",
        f"| Locked policy threshold | {policy['locked_threshold']:.6f} |",
        "| Final holdout | Sealed at model-card freeze |",
        "",
        "## Intended use",
        "",
        "- Portfolio demonstration of reproducible credit-risk modeling.",
        "- Analyst decision support with documented probability and reason codes.",
        "- Model-risk, calibration, drift, and subgroup diagnostic exercises.",
        "",
        "## Prohibited use",
        "",
        "- Autonomous credit approval, decline, pricing, or limit assignment.",
        "- Use as a legally sufficient adverse-action notice.",
        "- Deployment in a jurisdiction without legal, compliance, privacy, and model-risk review.",
        "- Inferring causality from SHAP values or subgroup associations.",
        "- Retraining or threshold tuning after viewing the final holdout.",
        "",
        "## Data and evaluation design",
        "",
        "- Source: Kaggle Home Credit Default Risk competition dataset.",
        "- Applicant-level deterministic stratified split: 70% train, 10% validation, "
        "10% calibration, and 10% final holdout.",
        "- Relational history aggregates are target-free and joined by `SK_ID_CURR`.",
        "- Preprocessing statistics are learned inside the training pipeline.",
        "- Final holdout was not accessed during feature development, selection, "
        "calibration, threshold definition, fairness analysis, or SHAP analysis.",
        "",
        "## Validation performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| ROC-AUC | {validation['roc_auc']:.4f} |",
        f"| PR-AUC | {validation['average_precision']:.4f} |",
        f"| Brier score | {validation['brier_score']:.5f} |",
        f"| Log loss | {validation['log_loss']:.5f} |",
        f"| Three-fold CV ROC-AUC | {candidate['cross_validation']['roc_auc']['mean']:.4f} "
        f"± {candidate['cross_validation']['roc_auc']['standard_deviation']:.4f} |",
        f"| Three-fold CV PR-AUC | "
        f"{candidate['cross_validation']['average_precision']['mean']:.4f} "
        f"± {candidate['cross_validation']['average_precision']['standard_deviation']:.4f} |",
        "",
        "## Calibration and decision policy",
        "",
        f"Sigmoid calibration was selected using {calibration['calibration_selection_rows']:,} "
        "internally reserved calibration records and refitted on the entire calibration split.",
        "",
        f"The locked threshold `{policy['locked_threshold']:.6f}` follows a hypothetical "
        f"false-negative:false-positive cost ratio of "
        f"{policy['false_negative_cost']:.0f}:{policy['false_positive_cost']:.0f}. "
        "These costs are portfolio assumptions, not lender estimates.",
        "",
        "| Validation operating metric | Value |",
        "|---|---:|",
        f"| Recall | {policy_metrics['recall']:.2%} |",
        f"| Precision | {policy_metrics['precision']:.2%} |",
        f"| Approval rate | {policy_metrics['approval_rate']:.2%} |",
        f"| Review/decline rate | {policy_metrics['review_or_decline_rate']:.2%} |",
        f"| Expected cost units/application | {policy_metrics['cost_per_application']:.4f} |",
        "",
        "## Feature governance",
        "",
        f"Policy: `{governance['policy_name']}`",
        "",
        governance["rationale"],
        "",
        "Excluded from all model preprocessing and scoring:",
        "",
        *[f"- `{feature}`" for feature in governance["excluded_decision_features"]],
        "",
        "The excluded source attributes remain available only for subgroup auditing. "
        "Proxy effects may remain and require monitoring.",
        "",
        "## Responsible-AI diagnostics",
        "",
        "These validation diagnostics are not proof of fairness or legal compliance.",
        "",
        "| Gap (max minus min) | Value |",
        "|---|---:|",
        f"| Gender recall | {gender_gaps['recall_max_min_gap']:.2%} |",
        f"| Gender false-positive rate | {gender_gaps['false_positive_rate_max_min_gap']:.2%} |",
        f"| Age-band recall | {age_gaps['recall_max_min_gap']:.2%} |",
        f"| Age-band false-positive rate | {age_gaps['false_positive_rate_max_min_gap']:.2%} |",
        "",
        "Age-band gaps remain material even after direct age exclusion, consistent with "
        "different base rates and possible proxy information. Human governance and ongoing "
        "subgroup monitoring are required.",
        "",
        "## Explainability",
        "",
        f"SHAP analysis used {shap_summary['sample_rows']:,} validation applicants and "
        f"{shap_summary['transformed_feature_count']:,} transformed features. SHAP values "
        "explain the raw XGBoost margin before sigmoid calibration and are not probability "
        "percentage-point changes.",
        "",
        "Leading global features:",
        "",
        *[f"- `{feature}`" for feature in top_features],
        "",
        "## Known limitations",
        "",
        "- Competition data is historical, anonymized, and not representative of every market.",
        "- Historical outcomes may encode structural inequities and policy effects.",
        "- External-score fields are opaque and would require vendor governance in production.",
        "- Missingness and dataset shift may materially affect predictions.",
        "- SHAP attribution is descriptive, not causal.",
        "- Subgroup diagnostics cover gender and age bands only.",
        "- Hypothetical costs do not establish real business value.",
        "",
        "## Human oversight and controls",
        "",
        "- A qualified analyst must review model output and reason codes.",
        "- Users must be able to escalate, override, and document decisions.",
        "- Input validation, access control, audit logging, monitoring, and incident response "
        "are required before any production consideration.",
        "- Performance, calibration, drift, and subgroup behavior must be monitored.",
        "- Final holdout results must be reported once and must not trigger model tuning.",
        "",
        "## Reproducibility freeze",
        "",
        "The accompanying `model_governance_freeze.json` records SHA-256 hashes for the "
        "candidate model, calibrated model, and modeling configuration. Holdout evaluation "
        "must fail if those hashes no longer match.",
    ]
    return "\n".join(lines) + "\n"


def build_model_card(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    report_dir: Path = REPORT_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Build the model card and freeze selected artifacts before holdout access."""
    candidate_path = model_dir / FULL_HISTORY_MODEL_PATH.name
    calibrated_path = model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name
    for path in (candidate_path, calibrated_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Cannot freeze missing artifact: {path}")

    evidence_paths = {
        "candidate": metrics_dir / "full_history_xgboost_metrics.json",
        "calibration": metrics_dir / "full_history_calibration_metrics.json",
        "policy": metrics_dir / "full_history_decision_policy.json",
        "fairness": metrics_dir / "full_history_responsible_ai_summary.json",
        "shap_summary": metrics_dir / "full_history_shap_summary.json",
    }
    context = {name: _require_json(path) for name, path in evidence_paths.items()}
    importance_path = metrics_dir / "full_history_shap_global_importance.csv"
    if not importance_path.exists():
        raise FileNotFoundError("Run `risklens explain-full-history` before model freeze")
    importance = pd.read_csv(importance_path)
    context["top_features"] = importance.head(10)["feature"].astype(str).tolist()

    config = load_modeling_config(config_path)
    expected_exclusions = [
        str(feature) for feature in config["feature_governance"]["excluded_decision_features"]
    ]
    recorded_exclusions = context["candidate"]["feature_governance"]["excluded_decision_features"]
    if recorded_exclusions != expected_exclusions:
        raise ValueError("Candidate metrics do not match the current governance exclusions")
    if context["fairness"].get("holdout_accessed") is not False:
        raise ValueError("Fairness evidence does not confirm a sealed holdout")
    if context["shap_summary"].get("holdout_accessed") is not False:
        raise ValueError("SHAP evidence does not confirm a sealed holdout")

    freeze = {
        "release_status": "pre_holdout_frozen_research_prototype",
        "model": "full_history_xgboost_calibrated",
        "locked_threshold": context["policy"]["locked_threshold"],
        "calibration_method": context["calibration"]["selected_method"],
        "governance_policy": config["feature_governance"]["policy_name"],
        "excluded_decision_features": expected_exclusions,
        "artifacts": {
            "candidate_model": {
                "path": str(candidate_path.relative_to(report_dir.parent)),
                "sha256": sha256_file(candidate_path),
            },
            "calibrated_model": {
                "path": str(calibrated_path.relative_to(report_dir.parent)),
                "sha256": sha256_file(calibrated_path),
            },
            "modeling_config": {
                "path": str(config_path.relative_to(report_dir.parent)),
                "sha256": sha256_file(config_path),
            },
        },
        "holdout_accessed": False,
        "post_holdout_tuning_permitted": False,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / MODEL_CARD_PATH.name).write_text(render_model_card(context), encoding="utf-8")
    (report_dir / MODEL_FREEZE_PATH.name).write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return freeze
