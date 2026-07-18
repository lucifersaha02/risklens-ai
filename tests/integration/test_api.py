"""Integration tests for the authenticated FastAPI inference contract."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from risklens.api.main import app, get_scorer
from risklens.serving.inference import ApplicantNotFoundError
from risklens.serving.schemas import ModelInfoResponse, PredictionResponse


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


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return an authenticated test client with a dependency override."""
    monkeypatch.setenv("RISKLENS_API_KEY", "test-secret")
    app.dependency_overrides[get_scorer] = lambda: FakeScorer()
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


def test_unknown_applicant_returns_sanitized_404(client: TestClient) -> None:
    response = client.get("/predict/999", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 404
    assert response.json()["error"] == "applicant_not_found"
