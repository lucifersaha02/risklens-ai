"""Tests for isolation of full-history policy validation data."""

from pathlib import Path

import pandas as pd
import pytest

from risklens.modeling.full_history_decision import (
    load_full_history_validation_data,
)


def write_policy_inputs(root: Path, *, duplicate_history: bool = False) -> None:
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
            "split": ["train", "validation", "holdout", "validation"],
        }
    ).to_parquet(processed / "split_assignments.parquet", index=False)
    history = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "BUREAU_RECORD_COUNT": [1, 2, 3, 4],
        }
    )
    if duplicate_history:
        history = pd.concat([history, history.iloc[[1]]], ignore_index=True)
    history.to_parquet(interim / "full_history_features.parquet", index=False)


def test_loader_uses_only_validation_applicants(tmp_path: Path) -> None:
    write_policy_inputs(tmp_path)
    validation = load_full_history_validation_data(
        raw_data_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    assert validation["SK_ID_CURR"].tolist() == [2, 4]
    assert 3 not in set(validation["SK_ID_CURR"])


def test_loader_rejects_duplicate_history_ids(tmp_path: Path) -> None:
    write_policy_inputs(tmp_path, duplicate_history=True)
    with pytest.raises(ValueError, match="duplicate applicant IDs"):
        load_full_history_validation_data(
            raw_data_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )
