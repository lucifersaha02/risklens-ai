"""Tests for new-application input mapping and policy-relative risk bands."""

from risklens.serving.new_application import (
    NewApplicationScorer,
    application_request_to_frame,
    simulator_risk_band,
)
from risklens.serving.schemas import NewApplicationRequest


def test_business_input_maps_to_home_credit_units() -> None:
    request = NewApplicationRequest(
        annual_income=600000,
        requested_credit=900000,
        annual_annuity=50000,
        goods_price=850000,
        employment_years=4,
        external_source_1=0.52,
        external_source_2=0.61,
        external_source_3=0.49,
        education_type="Higher education",
    )
    frame = application_request_to_frame(request)
    assert frame.loc[0, "AMT_INCOME_TOTAL"] == 600000
    assert frame.loc[0, "DAYS_EMPLOYED"] == -1461
    assert "TARGET" not in frame
    assert "CODE_GENDER" not in frame
    assert "DAYS_BIRTH" not in frame


def test_risk_bands_are_relative_to_locked_threshold() -> None:
    assert simulator_risk_band(0.05, 1 / 6) == "lower_estimated_risk"
    assert simulator_risk_band(0.10, 1 / 6) == "moderate_estimated_risk"
    assert simulator_risk_band(0.20, 1 / 6) == "elevated_risk"


def test_range_check_distinguishes_typical_uncommon_and_unseen() -> None:
    limits = {"observed_min": 25_650, "p01": 45_000, "p99": 472_500, "observed_max": 117_000_000}
    typical = NewApplicationScorer._range_check("income", "Income", 150_000, limits)
    uncommon = NewApplicationScorer._range_check("income", "Income", 600_000, limits)
    unseen = NewApplicationScorer._range_check("income", "Income", 200_000_000, limits)
    assert typical["status"] == "within_typical_range"
    assert uncommon["status"] == "uncommon_but_observed"
    assert unseen["status"] == "outside_observed_training_values"
