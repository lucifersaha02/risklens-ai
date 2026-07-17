"""Unit tests for deterministic applicant-level splitting."""

import pandas as pd
import pytest

from risklens.data.splitting import create_stratified_splits, split_summary


def sample_frame(rows: int = 1000) -> pd.DataFrame:
    """Create a deterministic imbalanced classification sample."""
    targets = [1 if index % 10 == 0 else 0 for index in range(rows)]
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(100000, 100000 + rows),
            "TARGET": targets,
        }
    )


def test_splits_are_complete_exclusive_and_proportional() -> None:
    assignments = create_stratified_splits(sample_frame())
    counts = assignments["split"].value_counts(normalize=True)
    assert len(assignments) == 1000
    assert assignments["SK_ID_CURR"].is_unique
    assert counts["train"] == pytest.approx(0.70, abs=0.002)
    assert counts["validation"] == pytest.approx(0.10, abs=0.002)
    assert counts["calibration"] == pytest.approx(0.10, abs=0.002)
    assert counts["holdout"] == pytest.approx(0.10, abs=0.002)


def test_splits_are_reproducible() -> None:
    first = create_stratified_splits(sample_frame(), random_seed=42)
    second = create_stratified_splits(sample_frame(), random_seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_target_rate_is_preserved() -> None:
    assignments = create_stratified_splits(sample_frame())
    rates = assignments.groupby("split")["TARGET"].mean()
    assert all(rate == pytest.approx(0.10, abs=0.002) for rate in rates)


def test_duplicate_primary_keys_are_rejected() -> None:
    frame = sample_frame(100)
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]
    with pytest.raises(ValueError, match="unique"):
        create_stratified_splits(frame)


def test_summary_contains_all_splits() -> None:
    summary = split_summary(create_stratified_splits(sample_frame()))
    assert summary["total_rows"] == 1000
    assert set(summary["splits"]) == {"train", "validation", "calibration", "holdout"}
