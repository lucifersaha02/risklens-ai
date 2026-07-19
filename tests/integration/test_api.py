"""Integration tests for the authenticated FastAPI inference contract."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from risklens.api.main import (
    app,
    get_evidence_assistant,
    get_monitoring_summary,
    get_new_application_scorer,
    get_portfolio_summary,
    get_scorer,
)
from risklens.rag.assistant import AssistantResponse
from risklens.serving.inference import ApplicantNotFoundError
from risklens.serving.schemas import (
    ModelInfoResponse,
    MonitoringSummaryResponse,
    NewApplicationResponse,
    PortfolioSummaryResponse,
    PredictionResponse,
)


class FakeScorer:
    """Deterministic scorer used to test HTTP behavior without model artifacts."""

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model="full_history_xgboost_calibrated",
            model_version="abc123",
            release_status="final_holdout_evaluated_research_prototype",
            calibration_method="sigmoid",
            decision_threshold=1 / 6,
            governance_policy="audit_only_v1",
            excluded_decision_features=["CODE_GENDER"],
            holdout_accessed=True,
            post_holdout_tuning_permitted=False,
        )

    def score_applicant(self, applicant_id: int, reason_count: int = 5) -> PredictionResponse:
        del reason_count
        if applicant_id == 999:
            raise ApplicantNotFoundError("Applicant 999 was not found")
        return PredictionResponse(
            applicant_id=applicant_id,
            assessment_mode="existing_applicant_full_history",
            application_summary={
                "applicant_id": applicant_id,
                "data_source": "Home Credit application_test",
                "contract_type": "Cash loans",
                "annual_income": 135000,
                "requested_credit": 568800,
                "loan_annuity_amount": 20560.5,
                "goods_price": 450000,
                "employment_years": 6.38,
                "external_signals_available": 3,
                "full_history_features_available": True,
            },
            model="full_history_xgboost_calibrated",
            model_version="abc123",
            calibration_method="sigmoid",
            raw_model_probability=0.1,
            calibrated_default_probability=0.09,
            decision_threshold=1 / 6,
            policy_action="standard_human_review",
            reason_codes={"risk_increasing": [], "risk_reducing": []},
            shap_output_space="raw_xgboost_margin_before_probability_calibration",
            explanation_additivity_error=0.0,
            human_decision_required=True,
            adverse_action_notice_ready=False,
        )


class FakeNewApplicationScorer:
    """Deterministic manual simulator used for API contract tests."""

    def score(self, request: Any, reason_count: int = 5) -> NewApplicationResponse:
        del request, reason_count
        return NewApplicationResponse(
            assessment_mode="application_only_manual_simulation",
            model="new_application_simulator_v1",
            model_version="sim123",
            calibrated_payment_difficulty_probability=0.10,
            review_threshold=1 / 6,
            risk_band="moderate_estimated_risk",
            review_route="standard_human_review",
            data_completeness=1.0,
            data_quality_warnings=[],
            derived_metrics=[
                {
                    "metric": "CREDIT_INCOME_RATIO",
                    "label": "Credit-to-income ratio",
                    "value": 1.5,
                    "display_format": "ratio",
                }
            ],
            input_range_checks=[
                {
                    "field": "AMT_INCOME_TOTAL",
                    "label": "Annual income",
                    "entered_value": 600000,
                    "observed_min": 25650,
                    "typical_p01": 45000,
                    "typical_p99": 472500,
                    "observed_max": 117000000,
                    "status": "uncommon_but_observed",
                    "interpretation": "Outside typical interval, but observed in training.",
                }
            ],
            assessment_coverage={
                "available_information": ["Current application"],
                "unavailable_full_history_information": ["Bureau history"],
                "comparison": "Application-only and full-history assessments differ.",
            },
            reason_codes={"risk_increasing": [], "risk_reducing": []},
            explanation_additivity_error=0.0,
            human_decision_required=True,
            automatic_approval_or_decline=False,
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return an authenticated test client with a dependency override."""
    monkeypatch.setenv("RISKLENS_API_KEY", "test-secret")
    app.dependency_overrides[get_scorer] = lambda: FakeScorer()
    app.dependency_overrides[get_portfolio_summary] = lambda: PortfolioSummaryResponse(
        model_version="abc123",
        evaluated_at_utc="2026-07-18T00:00:00+00:00",
        holdout_rows=100,
        locked_threshold=1 / 6,
        metrics={
            "roc_auc": 0.78,
            "average_precision": 0.27,
            "brier_score": 0.06,
            "log_loss": 0.23,
            "recall": 0.42,
            "precision": 0.27,
            "approval_rate": 0.87,
            "cost_per_application": 0.32,
        },
        confidence_intervals={"roc_auc": {"lower": 0.76, "upper": 0.80}},
        subgroup_gaps={
            "gender_recall": 0.10,
            "gender_false_positive_rate": 0.04,
            "age_band_recall": 0.36,
            "age_band_false_positive_rate": 0.18,
        },
        post_holdout_tuning_permitted=False,
    )
    app.dependency_overrides[get_monitoring_summary] = lambda: MonitoringSummaryResponse(
        model="full_history_xgboost_calibrated",
        model_version="abc123",
        reference_split="validation",
        current_population="unlabeled_test",
        reference_rows=100,
        current_rows=50,
        overall_severity="critical",
        prediction_drift={
            "psi": 0.01,
            "severity": "stable",
            "reference_mean_probability": 0.08,
            "current_mean_probability": 0.079,
        },
        feature_severity_counts={"stable": 99, "warning": 0, "critical": 1},
        top_feature_drift=[
            {"feature": "CREDIT_ANNUITY_RATIO", "psi": 1.64, "severity": "critical"}
        ],
        data_quality={"duplicate_id_rate": 0.0, "target_present": False, "alerts": []},
        interpretation="Investigation heuristic only.",
        labels_available=False,
        performance_drift_measured=False,
        post_holdout_tuning_permitted=False,
    )
    app.dependency_overrides[get_evidence_assistant] = lambda: (
        lambda question: AssistantResponse(
            question=question,
            answer_type="grounded_evidence_briefing",
            summary="Retrieved trusted evidence.",
            evidence=[
                {
                    "citation": "[reports/model_card.md#Prohibited use]",
                    "relevance_score": 0.8,
                    "excerpt": "Autonomous credit approval is prohibited.",
                }
            ],
            citations=["[reports/model_card.md#Prohibited use]"],
            disclosures=["Human review required."],
            human_review_required=True,
        )
    )
    app.dependency_overrides[get_new_application_scorer] = lambda: FakeNewApplicationScorer()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_is_available_without_api_key(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "health-test"})
    assert response.status_code == 200
    assert response.json()["model_ready"] is True
    assert response.headers["X-Request-ID"] == "health-test"


def test_model_info_requires_api_key(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 401


def test_model_info_returns_governance_metadata(client: TestClient) -> None:
    response = client.get("/model-info", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200
    assert response.json()["post_holdout_tuning_permitted"] is False


def test_prediction_returns_human_review_contract(client: TestClient) -> None:
    response = client.get("/predict/100001?reason_count=3", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["applicant_id"] == 100001
    assert payload["human_decision_required"] is True
    assert "TARGET" not in payload


def test_portfolio_summary_returns_frozen_evidence(client: TestClient) -> None:
    response = client.get("/portfolio-summary", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200
    assert response.json()["metrics"]["roc_auc"] == 0.78
    assert response.json()["post_holdout_tuning_permitted"] is False


def test_monitoring_summary_distinguishes_feature_and_prediction_drift(
    client: TestClient,
) -> None:
    response = client.get("/monitoring-summary", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200
    assert response.json()["overall_severity"] == "critical"
    assert response.json()["prediction_drift"]["severity"] == "stable"
    assert response.json()["performance_drift_measured"] is False


def test_evidence_assistant_requires_api_key(client: TestClient) -> None:
    response = client.post("/evidence-assistant/query", json={"question": "What is prohibited?"})
    assert response.status_code == 401


def test_evidence_assistant_returns_cited_human_review_contract(client: TestClient) -> None:
    response = client.post(
        "/evidence-assistant/query",
        json={"question": "What is prohibited?"},
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["human_review_required"] is True
    assert payload["generated_by_llm"] is False
    assert payload["evidence"][0]["citation"] in payload["citations"]


def test_new_application_simulator_returns_non_decision_contract(client: TestClient) -> None:
    response = client.post(
        "/simulate-new-application",
        headers={"X-API-Key": "test-secret"},
        json={
            "annual_income": 600000,
            "requested_credit": 900000,
            "annual_annuity": 50000,
            "goods_price": 850000,
            "employment_years": 4,
            "external_source_1": 0.52,
            "external_source_2": 0.61,
            "external_source_3": 0.49,
            "contract_type": "Cash loans",
            "owns_car": False,
            "owns_realty": True,
            "children": 0,
            "income_type": "Working",
            "education_type": "Higher education",
            "housing_type": "House / apartment",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assessment_mode"] == "application_only_manual_simulation"
    assert payload["human_decision_required"] is True
    assert payload["automatic_approval_or_decline"] is False


def test_new_application_simulator_rejects_invalid_external_score(client: TestClient) -> None:
    response = client.post(
        "/simulate-new-application",
        headers={"X-API-Key": "test-secret"},
        json={
            "annual_income": 600000,
            "requested_credit": 900000,
            "annual_annuity": 50000,
            "goods_price": 850000,
            "employment_years": 4,
            "external_source_1": 1.5,
        },
    )
    assert response.status_code == 422


def test_unknown_applicant_returns_sanitized_404(client: TestClient) -> None:
    response = client.get("/predict/999", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 404
    assert response.json()["error"] == "applicant_not_found"
