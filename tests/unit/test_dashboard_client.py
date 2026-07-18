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


def test_dashboard_client_validates_monitoring_contract() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "model": "full_history_xgboost_calibrated",
                "model_version": "abc123",
                "reference_split": "validation",
                "current_population": "test",
                "reference_rows": 100,
                "current_rows": 50,
                "overall_severity": "critical",
                "prediction_drift": {
                    "psi": 0.01,
                    "severity": "stable",
                    "reference_mean_probability": 0.08,
                    "current_mean_probability": 0.079,
                },
                "feature_severity_counts": {"stable": 99, "warning": 0, "critical": 1},
                "top_feature_drift": [
                    {"feature": "CREDIT_ANNUITY_RATIO", "psi": 1.64, "severity": "critical"}
                ],
                "data_quality": {
                    "duplicate_id_rate": 0.0,
                    "target_present": False,
                    "alerts": [],
                },
                "interpretation": "Investigation only.",
                "labels_available": False,
                "performance_drift_measured": False,
                "post_holdout_tuning_permitted": False,
            },
        )
    )
    client = RiskLensAPIClient("http://test", "secret", transport=transport)
    summary = client.monitoring_summary()
    assert summary.overall_severity == "critical"
    assert summary.prediction_drift.severity == "stable"
