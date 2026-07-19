"""Hash-verified scoring for manually entered new applications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risklens.explainability.shap_analysis import normalize_shap_values, reason_codes
from risklens.governance.model_card import sha256_file
from risklens.modeling.new_application import (
    SIMULATOR_MODEL_PATH,
    SIMULATOR_RELEASE_PATH,
    NewApplicationArtifact,
)
from risklens.serving.inference import _dense_row, policy_action
from risklens.serving.schemas import NewApplicationRequest, NewApplicationResponse


def application_request_to_frame(request: NewApplicationRequest) -> pd.DataFrame:
    """Map business-friendly fields to the exact Home Credit training definitions."""
    return pd.DataFrame(
        [
            {
                "CNT_CHILDREN": request.children,
                "AMT_INCOME_TOTAL": request.annual_income,
                "AMT_CREDIT": request.requested_credit,
                "AMT_ANNUITY": request.annual_annuity,
                "AMT_GOODS_PRICE": request.goods_price,
                "DAYS_EMPLOYED": -request.employment_years * 365.25,
                "EXT_SOURCE_1": request.external_source_1,
                "EXT_SOURCE_2": request.external_source_2,
                "EXT_SOURCE_3": request.external_source_3,
                "NAME_CONTRACT_TYPE": request.contract_type,
                "FLAG_OWN_CAR": "Y" if request.owns_car else "N",
                "FLAG_OWN_REALTY": "Y" if request.owns_realty else "N",
                "NAME_INCOME_TYPE": request.income_type,
                "NAME_EDUCATION_TYPE": request.education_type,
                "NAME_HOUSING_TYPE": request.housing_type,
            }
        ]
    )


def simulator_risk_band(probability: float, threshold: float) -> str:
    """Return descriptive policy-relative bands without making a credit decision."""
    if probability >= threshold:
        return "elevated_risk"
    if probability >= threshold / 2:
        return "moderate_estimated_risk"
    return "lower_estimated_risk"


class NewApplicationScorer:
    """Serve the frozen application-only simulator with validation and SHAP."""

    def __init__(
        self,
        model_path: Path = SIMULATOR_MODEL_PATH,
        release_path: Path = SIMULATOR_RELEASE_PATH,
    ) -> None:
        if not model_path.exists() or not release_path.exists():
            raise FileNotFoundError("Run `risklens train-new-application-simulator` first")
        release = json.loads(release_path.read_text(encoding="utf-8"))
        digest = sha256_file(model_path)
        if digest != release.get("artifact_sha256"):
            raise RuntimeError("New-application simulator artifact hash mismatch")
        artifact = joblib.load(model_path)
        if not isinstance(artifact, NewApplicationArtifact):
            raise TypeError("New-application simulator artifact type is invalid")
        self.artifact = artifact
        self.model_version = digest[:12]
        self._explainer: Any | None = None

    def _quality_warnings(self, frame: pd.DataFrame) -> list[str]:
        warnings = []
        reference = self.artifact.reference
        for column, limits in reference["numeric_ranges"].items():
            value = frame.iloc[0][column]
            if pd.isna(value):
                warnings.append(f"{column} is missing and will be imputed from training data")
            elif float(value) < limits["p01"] or float(value) > limits["p99"]:
                warnings.append(f"{column} is outside the central 98% of training values")
        for column, levels in reference["categorical_levels"].items():
            if str(frame.iloc[0][column]) not in levels:
                raise ValueError(f"{column} contains a category not observed during training")
        return warnings

    def score(
        self, request: NewApplicationRequest, reason_count: int = 5
    ) -> NewApplicationResponse:
        """Validate, score, explain, and route one new application simulation."""
        if reason_count <= 0 or reason_count > 20:
            raise ValueError("reason_count must be between 1 and 20")
        frame = application_request_to_frame(request)
        if list(frame.columns) != self.artifact.input_columns:
            raise RuntimeError("New-application input contract does not match the frozen artifact")
        warnings = self._quality_warnings(frame)
        model = self.artifact.model
        pipeline = model.base_model
        probability = float(model.predict_proba(frame)[0, 1])
        engineered = pipeline.named_steps["features"].transform(frame)
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(engineered)
        xgboost = pipeline.named_steps["model"]
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(xgboost)
        values = normalize_shap_values(self._explainer.shap_values(transformed))[0]
        margin = float(xgboost.predict(transformed, output_margin=True)[0])
        expected = float(np.asarray(self._explainer.expected_value).reshape(-1)[-1])
        additivity_error = abs(margin - (expected + float(values.sum())))
        threshold = self.artifact.threshold
        external_count = sum(
            value is not None
            for value in (
                request.external_source_1,
                request.external_source_2,
                request.external_source_3,
            )
        )
        completeness = (12 + external_count) / 15
        return NewApplicationResponse(
            assessment_mode="application_only_manual_simulation",
            model=self.artifact.release_name,
            model_version=self.model_version,
            calibrated_payment_difficulty_probability=probability,
            review_threshold=threshold,
            risk_band=simulator_risk_band(probability, threshold),
            review_route=policy_action(probability, threshold),
            data_completeness=completeness,
            data_quality_warnings=warnings,
            reason_codes=reason_codes(
                preprocessor.get_feature_names_out(),
                values,
                _dense_row(transformed),
                top_n=reason_count,
            ),
            explanation_additivity_error=additivity_error,
            human_decision_required=True,
            automatic_approval_or_decline=False,
        )
