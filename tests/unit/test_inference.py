"""Tests for frozen serving verification and target-free feature assembly."""

import json
from pathlib import Path

import pandas as pd
import pytest

from risklens.governance.model_card import sha256_file
from risklens.serving.inference import (
    load_applicant_feature_row,
    policy_action,
    verify_serving_freeze,
)
from risklens.serving.schemas import PredictionResponse


def test_serving_accepts_final_evaluated_frozen_artifact(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    reports.mkdir()
    models.mkdir()
    model = models / "model.joblib"
    model.write_bytes(b"frozen")
    freeze = {
        "release_status": "final_holdout_evaluated_research_prototype",
        "post_holdout_tuning_permitted": False,
        "artifacts": {"model": {"path": "models/model.joblib", "sha256": sha256_file(model)}},
    }
    (reports / "model_governance_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    result = verify_serving_freeze(report_dir=reports, project_root=tmp_path)
    assert result["post_holdout_tuning_permitted"] is False


def test_applicant_loader_excludes_target(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    pd.DataFrame({"SK_ID_CURR": [100001], "TARGET": [1], "AMT_CREDIT": [200000.0]}).to_csv(
        raw / "application_train.csv", index=False
    )
    pd.DataFrame({"SK_ID_CURR": [100001], "BUREAU_RECORD_COUNT": [3]}).to_parquet(
        interim / "full_history_features.parquet", index=False
    )
    frame = load_applicant_feature_row(100001, raw_data_dir=raw, interim_dir=interim)
    assert "TARGET" not in frame.columns
    assert frame.iloc[0]["BUREAU_RECORD_COUNT"] == 3


def test_applicant_loader_rejects_unknown_id(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    pd.DataFrame({"SK_ID_CURR": [1]}).to_csv(raw / "application_test.csv", index=False)
    with pytest.raises(KeyError, match="was not found"):
        load_applicant_feature_row(2, raw_data_dir=raw, interim_dir=interim)


def test_policy_action_never_returns_automated_decline() -> None:
    assert policy_action(0.10, 0.20) == "standard_human_review"
    assert policy_action(0.30, 0.20) == "enhanced_manual_review_recommended"


def test_prediction_contract_forbids_observed_target() -> None:
    assert "TARGET" not in PredictionResponse.model_fields
