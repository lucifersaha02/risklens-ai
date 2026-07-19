"""Tests for traceable business-friendly dashboard feature labels."""

from risklens.dashboard.feature_labels import business_feature_name, display_feature_name


def test_known_feature_has_business_and_technical_names() -> None:
    assert display_feature_name("EXT_SOURCE_MEAN") == (
        "Average external credit signal (EXT_SOURCE_MEAN)"
    )


def test_bureau_timing_label_does_not_imply_applicant_age() -> None:
    assert business_feature_name("BUREAU_DAYS_CREDIT_MAX") == ("Most recent bureau credit timing")


def test_unknown_feature_retains_exact_technical_traceability() -> None:
    assert display_feature_name("SOME_NEW_FEATURE") == ("Some New Feature (SOME_NEW_FEATURE)")
