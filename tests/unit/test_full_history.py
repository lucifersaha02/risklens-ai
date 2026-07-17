"""Tests for full-history joins and candidate comparisons."""

from pathlib import Path

import pandas as pd
import pytest

from risklens.modeling.full_history import (
    _metric_comparison,
    load_full_history_train_validation_data,
)


def write_inputs(root: Path, *, include_target_in_history: bool = False) -> None:
    """Write a minimal application, split, and history dataset."""
    raw = root / "raw"
    interim = root / "interim"
    processed = root / "processed"
    raw.mkdir()
    interim.mkdir()
    processed.mkdir()
    pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "AMT_CREDIT": [10, 20, 30]}).to_csv(
        raw / "application_train.csv", index=False
    )
    pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "split": ["train", "validation", "holdout"]}).to_parquet(
        processed / "split_assignments.parquet", index=False
    )
    history = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "BUREAU_RECORD_COUNT": [2, 0, 4]})
    if include_target_in_history:
        history["TARGET"] = [0, 1, 0]
    history.to_parquet(interim / "full_history_features.parquet", index=False)


def test_loader_joins_history_and_keeps_holdout_sealed(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    train, validation = load_full_history_train_validation_data(
        raw_data_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    assert train["SK_ID_CURR"].tolist() == [1]
    assert validation["SK_ID_CURR"].tolist() == [2]
    assert train["BUREAU_RECORD_COUNT"].tolist() == [2]
    assert 3 not in set(pd.concat([train, validation])["SK_ID_CURR"])


def test_loader_rejects_target_leakage(tmp_path: Path) -> None:
    write_inputs(tmp_path, include_target_in_history=True)
    with pytest.raises(ValueError, match="must never contain TARGET"):
        load_full_history_train_validation_data(
            raw_data_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_metric_comparison_selects_better_pr_auc() -> None:
    application = {
        "roc_auc": 0.75,
        "average_precision": 0.24,
        "brier_score": 0.068,
        "log_loss": 0.25,
    }
    history = {
        "roc_auc": 0.78,
        "average_precision": 0.28,
        "brier_score": 0.065,
        "log_loss": 0.24,
    }
    comparison = _metric_comparison(application, history)
    assert comparison["selected_candidate"] == "full_history_xgboost"
    assert comparison["average_precision"]["delta_full_history_minus_application"] == pytest.approx(
        0.04
    )
