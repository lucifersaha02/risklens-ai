"""Tests for full-history calibration data isolation."""

from pathlib import Path

import pandas as pd
import pytest

from risklens.modeling.full_history_calibration import (
    load_full_history_calibration_data,
)


def write_calibration_inputs(root: Path, *, target_in_history: bool = False) -> None:
    """Write minimal application, assignment, and history inputs."""
    raw = root / "raw"
    interim = root / "interim"
    processed = root / "processed"
    raw.mkdir()
    interim.mkdir()
    processed.mkdir()
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "AMT_CREDIT": [10, 20, 30, 40],
        }
    ).to_csv(raw / "application_train.csv", index=False)
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "split": ["train", "calibration", "holdout", "calibration"],
        }
    ).to_parquet(processed / "split_assignments.parquet", index=False)
    history = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "BUREAU_RECORD_COUNT": [1, 2, 3, 4],
        }
    )
    if target_in_history:
        history["TARGET"] = [0, 1, 0, 1]
    history.to_parquet(interim / "full_history_features.parquet", index=False)


def test_loader_uses_only_calibration_applicants(tmp_path: Path) -> None:
    write_calibration_inputs(tmp_path)
    calibration = load_full_history_calibration_data(
        raw_data_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    assert calibration["SK_ID_CURR"].tolist() == [2, 4]
    assert calibration["BUREAU_RECORD_COUNT"].tolist() == [2, 4]
    assert 3 not in set(calibration["SK_ID_CURR"])


def test_loader_rejects_history_target_leakage(tmp_path: Path) -> None:
    write_calibration_inputs(tmp_path, target_in_history=True)
    with pytest.raises(ValueError, match="must never contain TARGET"):
        load_full_history_calibration_data(
            raw_data_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )
