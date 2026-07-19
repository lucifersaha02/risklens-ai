"""Hash-verified applicant scoring with calibrated probabilities and SHAP reasons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risklens.config import INTERIM_DATA_DIR, MODEL_DIR, RAW_DATA_DIR, REPORT_DIR
from risklens.explainability.shap_analysis import normalize_shap_values, reason_codes
from risklens.governance.model_card import MODEL_FREEZE_PATH, sha256_file
from risklens.modeling.full_history_calibration import (
    FULL_HISTORY_CALIBRATED_MODEL_PATH,
)
from risklens.serving.schemas import ModelInfoResponse, PredictionResponse


class ApplicantNotFoundError(LookupError):
    """Raised when an applicant identifier is absent from the demo feature store."""


def verify_serving_freeze(
    report_dir: Path = REPORT_DIR,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen artifacts for read-only inference after holdout evaluation."""
    root = project_root or report_dir.parent
    freeze_path = report_dir / MODEL_FREEZE_PATH.name
    if not freeze_path.exists():
        raise FileNotFoundError("Model governance freeze is missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    allowed_statuses = {
        "pre_holdout_frozen_research_prototype",
        "final_holdout_evaluated_research_prototype",
    }
    if freeze.get("release_status") not in allowed_statuses:
        raise RuntimeError("Model release status is not eligible for serving")
    if freeze.get("post_holdout_tuning_permitted") is not False:
        raise RuntimeError("Serving requires a freeze that prohibits model tuning")
    for name, artifact in freeze["artifacts"].items():
        path = root / Path(str(artifact["path"]))
        if not path.exists():
            raise FileNotFoundError(f"Frozen serving artifact is missing: {name}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Frozen serving artifact hash mismatch: {name}")
    return freeze


def load_applicant_feature_row(
    applicant_id: int,
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    history_filename: str = "full_history_features.parquet",
    csv_chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Load one applicant's application and target-free history features."""
    if applicant_id <= 0:
        raise ValueError("Applicant ID must be positive")
    matches: list[pd.DataFrame] = []
    matched_source = ""
    for filename in ("application_train.csv", "application_test.csv"):
        path = raw_data_dir / filename
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, chunksize=csv_chunk_size):
            selected = chunk.loc[chunk["SK_ID_CURR"] == applicant_id]
            if not selected.empty:
                matches.append(selected)
                matched_source = filename
                break
    if not matches:
        raise ApplicantNotFoundError(f"Applicant {applicant_id} was not found")
    application = pd.concat(matches, ignore_index=True)
    if len(application) != 1:
        raise ValueError(f"Applicant {applicant_id} appears more than once")
    source_columns = application.columns.tolist()
    application = application.drop(columns="TARGET", errors="ignore")

    history_path = interim_dir / history_filename
    if not history_path.exists():
        raise FileNotFoundError("Run `risklens build-history-features` before scoring")
    history = pd.read_parquet(
        history_path,
        filters=[("SK_ID_CURR", "==", applicant_id)],
    )
    if "TARGET" in history.columns:
        raise ValueError("History feature store must never contain TARGET")
    if len(history) != 1:
        raise ValueError(f"Exactly one history row is required for applicant {applicant_id}")
    combined = application.merge(history, on="SK_ID_CURR", how="inner", validate="one_to_one")
    combined.attrs["source_columns"] = source_columns
    combined.attrs["source_filename"] = matched_source
    if len(combined) != 1 or "TARGET" in combined.columns:
        raise ValueError("Serving feature assembly failed its leakage or cardinality check")
    return combined


def policy_action(probability: float, threshold: float) -> str:
    """Map frozen policy output to a human-oversight workflow action."""
    if not 0 <= probability <= 1 or not 0 < threshold < 1:
        raise ValueError("Probability or threshold is outside its valid interval")
    return (
        "enhanced_manual_review_recommended"
        if probability >= threshold
        else "standard_human_review"
    )


def _dense_row(matrix: Any) -> np.ndarray:
    """Convert one sparse or dense transformed row to a flat array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float).reshape(-1)


class FrozenRiskScorer:
    """Load and serve the immutable governed model with local explanations."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        report_dir: Path = REPORT_DIR,
    ) -> None:
        self.freeze = verify_serving_freeze(report_dir=report_dir)
        model_path = model_dir / FULL_HISTORY_CALIBRATED_MODEL_PATH.name
        self.model = joblib.load(model_path)
        self.pipeline = self.model.base_model
        self.threshold = float(self.freeze["locked_threshold"])
        self.model_version = self.freeze["artifacts"]["calibrated_model"]["sha256"][:12]
        self._explainer: Any | None = None

    def model_info(self) -> ModelInfoResponse:
        """Return frozen governance metadata without exposing filesystem details."""
        return ModelInfoResponse(
            model=str(self.freeze["model"]),
            model_version=self.model_version,
            release_status=str(self.freeze["release_status"]),
            calibration_method=str(self.freeze["calibration_method"]),
            decision_threshold=self.threshold,
            governance_policy=str(self.freeze["governance_policy"]),
            excluded_decision_features=list(self.freeze["excluded_decision_features"]),
            holdout_accessed=bool(self.freeze["holdout_accessed"]),
            post_holdout_tuning_permitted=bool(self.freeze["post_holdout_tuning_permitted"]),
        )

    def score_frame(
        self,
        frame: pd.DataFrame,
        reason_count: int = 5,
    ) -> PredictionResponse:
        """Score exactly one target-free applicant frame and explain the raw margin."""
        if len(frame) != 1:
            raise ValueError("Scoring requires exactly one applicant")
        if "TARGET" in frame.columns:
            raise ValueError("TARGET is forbidden in serving inputs")
        if reason_count <= 0 or reason_count > 20:
            raise ValueError("reason_count must be between 1 and 20")
        applicant_id = int(frame.iloc[0]["SK_ID_CURR"])
        row = frame.iloc[0]
        employed_days = row.get("DAYS_EMPLOYED")
        employment_years = None
        if pd.notna(employed_days) and float(employed_days) != 365243:
            employment_years = max(-float(employed_days) / 365.25, 0.0)
        source = (
            "Home Credit application_train"
            if "TARGET" in frame.attrs.get("source_columns", [])
            else "Home Credit application_test"
        )
        raw_probability = float(self.pipeline.predict_proba(frame)[0, 1])
        calibrated_probability = float(self.model.predict_proba(frame)[0, 1])

        engineered = self.pipeline.named_steps["features"].transform(frame)
        governed = self.pipeline.named_steps["governance"].transform(engineered)
        preprocessor = self.pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(governed)
        feature_names = preprocessor.get_feature_names_out()
        xgboost_model = self.pipeline.named_steps["model"]
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(xgboost_model)
        values = normalize_shap_values(self._explainer.shap_values(transformed))[0]
        raw_margin = float(xgboost_model.predict(transformed, output_margin=True)[0])
        expected_value = float(
            np.asarray(self._explainer.expected_value, dtype=float).reshape(-1)[-1]
        )
        additivity_error = abs(raw_margin - (expected_value + float(values.sum())))

        return PredictionResponse(
            applicant_id=applicant_id,
            assessment_mode="existing_applicant_full_history",
            application_summary={
                "applicant_id": applicant_id,
                "data_source": source,
                "contract_type": str(row["NAME_CONTRACT_TYPE"]),
                "annual_income": float(row["AMT_INCOME_TOTAL"]),
                "requested_credit": float(row["AMT_CREDIT"]),
                "loan_annuity_amount": float(row["AMT_ANNUITY"]),
                "goods_price": (
                    None if pd.isna(row.get("AMT_GOODS_PRICE")) else float(row["AMT_GOODS_PRICE"])
                ),
                "employment_years": employment_years,
                "external_signals_available": int(
                    sum(
                        pd.notna(row.get(name))
                        for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
                    )
                ),
                "full_history_features_available": True,
            },
            model=str(self.freeze["model"]),
            model_version=self.model_version,
            calibration_method=str(self.freeze["calibration_method"]),
            raw_model_probability=raw_probability,
            calibrated_default_probability=calibrated_probability,
            decision_threshold=self.threshold,
            policy_action=policy_action(calibrated_probability, self.threshold),
            reason_codes=reason_codes(
                feature_names,
                values,
                _dense_row(transformed),
                top_n=reason_count,
            ),
            shap_output_space="raw_xgboost_margin_before_probability_calibration",
            explanation_additivity_error=additivity_error,
            human_decision_required=True,
            adverse_action_notice_ready=False,
        )

    def score_applicant(
        self,
        applicant_id: int,
        raw_data_dir: Path = RAW_DATA_DIR,
        interim_dir: Path = INTERIM_DATA_DIR,
        reason_count: int = 5,
    ) -> PredictionResponse:
        """Load, score, and explain one known applicant."""
        frame = load_applicant_feature_row(
            applicant_id,
            raw_data_dir=raw_data_dir,
            interim_dir=interim_dir,
        )
        return self.score_frame(frame, reason_count=reason_count)
