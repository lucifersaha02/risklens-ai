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


class ConfidenceInterval(BaseModel):
    """Lower and upper bounds for one bootstrap interval."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float


class PortfolioMetrics(BaseModel):
    """Frozen final holdout point estimates."""

    model_config = ConfigDict(extra="forbid")

    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss: float
    recall: float
    precision: float
    approval_rate: float
    cost_per_application: float


class SubgroupGapSummary(BaseModel):
    """Diagnostic max-minus-min gaps on final holdout."""

    model_config = ConfigDict(extra="forbid")

    gender_recall: float
    gender_false_positive_rate: float
    age_band_recall: float
    age_band_false_positive_rate: float


class PortfolioSummaryResponse(BaseModel):
    """Read-only final evidence exposed to the dashboard."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    evaluated_at_utc: str
    holdout_rows: int
    locked_threshold: float
    metrics: PortfolioMetrics
    confidence_intervals: dict[str, ConfidenceInterval]
    subgroup_gaps: SubgroupGapSummary
    post_holdout_tuning_permitted: bool


class DriftFeature(BaseModel):
    """PSI alert for one monitored transformed feature."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    psi: float = Field(ge=0)
    severity: Literal["stable", "warning", "critical"]


class PredictionDrift(BaseModel):
    """Calibrated prediction-distribution monitoring summary."""

    model_config = ConfigDict(extra="forbid")

    psi: float = Field(ge=0)
    severity: Literal["stable", "warning", "critical"]
    reference_mean_probability: float = Field(ge=0, le=1)
    current_mean_probability: float = Field(ge=0, le=1)


class FeatureSeverityCounts(BaseModel):
    """Counts of monitored features by alert level."""

    model_config = ConfigDict(extra="forbid")

    stable: int = Field(ge=0)
    warning: int = Field(ge=0)
    critical: int = Field(ge=0)


class MonitoringDataQuality(BaseModel):
    """Current-batch data-quality checks."""

    model_config = ConfigDict(extra="forbid")

    duplicate_id_rate: float = Field(ge=0, le=1)
    target_present: bool
    alerts: list[str]


class MonitoringSummaryResponse(BaseModel):
    """Read-only drift snapshot exposed to operations and the dashboard."""

    model_config = ConfigDict(extra="forbid")

    model: str
    model_version: str
    reference_split: str
    current_population: str
    reference_rows: int = Field(gt=0)
    current_rows: int = Field(gt=0)
    overall_severity: Literal["stable", "warning", "critical"]
    prediction_drift: PredictionDrift
    feature_severity_counts: FeatureSeverityCounts
    top_feature_drift: list[DriftFeature]
    data_quality: MonitoringDataQuality
    interpretation: str
    labels_available: bool
    performance_drift_measured: bool
    post_holdout_tuning_permitted: bool


class EvidencePassage(BaseModel):
    """One cited passage returned by the governance evidence assistant."""

    model_config = ConfigDict(extra="forbid")

    citation: str
    relevance_score: float = Field(ge=0)
    excerpt: str


class EvidenceAssistantResponse(BaseModel):
    """Guarded evidence response containing no autonomous lending recommendation."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer_type: Literal["grounded_evidence_briefing", "insufficient_evidence"]
    summary: str
    evidence: list[EvidencePassage]
    citations: list[str]
    disclosures: list[str]
    human_review_required: bool
    generated_by_llm: bool


class EvidenceAssistantRequest(BaseModel):
    """Bounded analyst question accepted by the evidence endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)


class NewApplicationRequest(BaseModel):
    """Business-friendly, application-time-only simulator inputs."""

    model_config = ConfigDict(extra="forbid")

    annual_income: float = Field(gt=0, le=100_000_000)
    requested_credit: float = Field(gt=0, le=100_000_000)
    annual_annuity: float = Field(gt=0, le=10_000_000)
    goods_price: float = Field(gt=0, le=100_000_000)
    employment_years: float = Field(ge=0, le=60)
    external_source_1: float | None = Field(default=None, ge=0, le=1)
    external_source_2: float | None = Field(default=None, ge=0, le=1)
    external_source_3: float | None = Field(default=None, ge=0, le=1)
    contract_type: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    owns_car: bool = False
    owns_realty: bool = True
    children: int = Field(default=0, ge=0, le=20)
    income_type: Literal[
        "Businessman",
        "Commercial associate",
        "Maternity leave",
        "Pensioner",
        "State servant",
        "Student",
        "Unemployed",
        "Working",
    ] = "Working"
    education_type: Literal[
        "Academic degree",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Secondary / secondary special",
    ] = "Secondary / secondary special"
    housing_type: Literal[
        "Co-op apartment",
        "House / apartment",
        "Municipal apartment",
        "Office apartment",
        "Rented apartment",
        "With parents",
    ] = "House / apartment"


class NewApplicationResponse(BaseModel):
    """Governed manual simulation response, explicitly not a lending decision."""

    model_config = ConfigDict(extra="forbid")

    assessment_mode: Literal["application_only_manual_simulation"]
    model: str
    model_version: str
    calibrated_payment_difficulty_probability: float = Field(ge=0, le=1)
    review_threshold: float = Field(gt=0, lt=1)
    risk_band: Literal["lower_estimated_risk", "moderate_estimated_risk", "elevated_risk"]
    review_route: Literal["standard_human_review", "enhanced_manual_review_recommended"]
    data_completeness: float = Field(ge=0, le=1)
    data_quality_warnings: list[str]
    reason_codes: ReasonCodeSet
    explanation_additivity_error: float = Field(ge=0)
    human_decision_required: bool = True
    automatic_approval_or_decline: bool = False
