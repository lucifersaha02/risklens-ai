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

    @staticmethod
    def _range_check(
        field: str,
        label: str,
        value: float | None,
        limits: dict[str, float],
    ) -> dict[str, Any]:
        if value is None or pd.isna(value):
            status = "missing_will_be_imputed"
            interpretation = "Missing; the pipeline will use training-learned imputation."
            entered_value = None
        elif limits["p01"] <= float(value) <= limits["p99"]:
            status = "within_typical_range"
            interpretation = "Within the central 98% of simulator-training values."
            entered_value = float(value)
        elif limits["observed_min"] <= float(value) <= limits["observed_max"]:
            status = "uncommon_but_observed"
            interpretation = "Outside the typical interval, but observed in simulator training."
            entered_value = float(value)
        else:
            status = "outside_observed_training_values"
            interpretation = (
                "Outside values observed in simulator training; applicability is limited."
            )
            entered_value = float(value)
        return {
            "field": field,
            "label": label,
            "entered_value": entered_value,
            "observed_min": limits["observed_min"],
            "typical_p01": limits["p01"],
            "typical_p99": limits["p99"],
            "observed_max": limits["observed_max"],
            "status": status,
            "interpretation": interpretation,
        }

    def _range_checks(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        reference = self.artifact.reference
        engineered = self.artifact.model.base_model.named_steps["features"].transform(frame)
        raw_fields = {
            "AMT_INCOME_TOTAL": "Annual income",
            "AMT_CREDIT": "Requested credit",
            "AMT_ANNUITY": "Loan annuity amount",
            "AMT_GOODS_PRICE": "Goods price",
            "CNT_CHILDREN": "Number of children",
            "EXT_SOURCE_1": "External credit signal 1",
            "EXT_SOURCE_2": "External credit signal 2",
            "EXT_SOURCE_3": "External credit signal 3",
        }
        checks = [
            self._range_check(
                field,
                label,
                frame.iloc[0][field],
                reference["numeric_ranges"][field],
            )
            for field, label in raw_fields.items()
        ]
        checks.append(
            self._range_check(
                "EMPLOYMENT_YEARS",
                "Employment history (years)",
                float(engineered.iloc[0]["EMPLOYMENT_YEARS"]),
                reference["derived_ranges"]["EMPLOYMENT_YEARS"],
            )
        )
        return checks

    def _quality_warnings(self, frame: pd.DataFrame, checks: list[dict[str, Any]]) -> list[str]:
        warnings = []
        reference = self.artifact.reference
        for check in checks:
            if check["status"] == "uncommon_but_observed":
                warnings.append(f"{check['label']} is uncommon but observed in simulator training")
            elif check["status"] == "outside_observed_training_values":
                warnings.append(
                    f"{check['label']} is outside values observed in simulator training"
                )
            elif check["status"] == "missing_will_be_imputed":
                warnings.append(f"{check['label']} is missing and will be imputed")
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
        checks = self._range_checks(frame)
        warnings = self._quality_warnings(frame, checks)
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
        derived_metrics = [
            {
                "metric": "CREDIT_INCOME_RATIO",
                "label": "Credit-to-income ratio",
                "value": float(engineered.iloc[0]["CREDIT_INCOME_RATIO"]),
                "display_format": "ratio",
            },
            {
                "metric": "ANNUITY_INCOME_RATIO",
                "label": "Annuity-to-income ratio (dataset-scale)",
                "value": float(engineered.iloc[0]["ANNUITY_INCOME_RATIO"]),
                "display_format": "percentage",
            },
            {
                "metric": "GOODS_CREDIT_RATIO",
                "label": "Goods-price-to-credit ratio",
                "value": float(engineered.iloc[0]["GOODS_CREDIT_RATIO"]),
                "display_format": "ratio",
            },
            {
                "metric": "EXT_SOURCE_MEAN",
                "label": "Average external credit signal",
                "value": float(engineered.iloc[0]["EXT_SOURCE_MEAN"]),
                "display_format": "number",
            },
            {
                "metric": "EMPLOYMENT_YEARS",
                "label": "Employment history",
                "value": float(engineered.iloc[0]["EMPLOYMENT_YEARS"]),
                "display_format": "years",
            },
        ]
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
            derived_metrics=derived_metrics,
            input_range_checks=checks,
            assessment_coverage={
                "available_information": [
                    "Current application amounts and categories",
                    "Employment duration",
                    "Up to three anonymized external credit signals",
                    "Application-time derived ratios",
                ],
                "unavailable_full_history_information": [
                    "Detailed bureau credit records",
                    "Previous Home Credit applications",
                    "Instalment-payment history",
                    "Credit-card balance history",
                    "POS/cash-loan history",
                ],
                "comparison": (
                    "Application-only simulation uses fewer information sources than the "
                    "existing-applicant full-history assessment. Their probabilities are not "
                    "interchangeable."
                ),
            },
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
