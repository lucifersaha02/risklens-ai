"""Unit tests for the Streamlit dashboard's API-only client."""

import httpx
import pytest

from risklens.dashboard.client import RiskLensAPIClient, RiskLensAPIError


def test_dashboard_client_sends_api_key_and_prediction_parameters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "secret"
        assert request.url.params["reason_count"] == "3"
        return httpx.Response(
            200,
            json={
                "applicant_id": 100001,
                "model": "model",
                "model_version": "version",
                "calibration_method": "sigmoid",
                "raw_model_probability": 0.1,
                "calibrated_default_probability": 0.09,
                "decision_threshold": 0.166667,
                "policy_action": "standard_human_review",
                "reason_codes": {"risk_increasing": [], "risk_reducing": []},
                "shap_output_space": "raw_xgboost_margin_before_probability_calibration",
                "explanation_additivity_error": 0.0,
                "human_decision_required": True,
                "adverse_action_notice_ready": False,
            },
        )

    client = RiskLensAPIClient("http://test", "secret", transport=httpx.MockTransport(handler))
    prediction = client.predict(100001, reason_count=3)
    assert prediction.applicant_id == 100001


def test_dashboard_client_normalizes_api_errors() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, json={"error": "applicant_not_found"})
    )
    client = RiskLensAPIClient("http://test", "secret", transport=transport)
    with pytest.raises(RiskLensAPIError, match="applicant_not_found"):
        client.predict(999)


def test_dashboard_client_normalizes_transport_errors() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = RiskLensAPIClient("http://test", "secret", transport=httpx.MockTransport(unavailable))
    with pytest.raises(RiskLensAPIError, match="unavailable"):
        client.health()
