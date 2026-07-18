"""Typed inference contracts shared by the CLI, API, and dashboard."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReasonCode(BaseModel):
    """One transformed-feature contribution to the raw XGBoost margin."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    transformed_value: float
    shap_value: float
    direction: Literal["increases_raw_model_risk", "reduces_raw_model_risk"]


class ReasonCodeSet(BaseModel):
    """Strongest local contributions in both risk directions."""

    model_config = ConfigDict(extra="forbid")

    risk_increasing: list[ReasonCode]
    risk_reducing: list[ReasonCode]


class PredictionResponse(BaseModel):
    """Governed prediction response containing no observed outcome."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: int = Field(gt=0)
    model: str
    model_version: str
    calibration_method: str
    raw_model_probability: float = Field(ge=0, le=1)
    calibrated_default_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(gt=0, lt=1)
    policy_action: Literal["standard_human_review", "enhanced_manual_review_recommended"]
    reason_codes: ReasonCodeSet
    shap_output_space: Literal["raw_xgboost_margin_before_probability_calibration"]
    explanation_additivity_error: float = Field(ge=0)
    human_decision_required: bool = True
    adverse_action_notice_ready: bool = False


class ModelInfoResponse(BaseModel):
    """Non-sensitive metadata for serving health and governance endpoints."""

    model_config = ConfigDict(extra="forbid")

    model: str
    model_version: str
    release_status: str
    calibration_method: str
    decision_threshold: float
    governance_policy: str
    excluded_decision_features: list[str]
    holdout_accessed: bool
    post_holdout_tuning_permitted: bool
