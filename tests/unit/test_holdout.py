"""Tests for one-time holdout freeze verification and confidence intervals."""

import json
from pathlib import Path

import numpy as np
import pytest

from risklens.governance.holdout import (
    bootstrap_confidence_intervals,
    verify_frozen_artifacts,
)
from risklens.governance.model_card import sha256_file


def write_freeze(root: Path, *, accessed: bool = False) -> Path:
    """Write a minimal freeze manifest and hashed artifact."""
    reports = root / "reports"
    models = root / "models"
    reports.mkdir()
    models.mkdir()
    artifact = models / "model.joblib"
    artifact.write_bytes(b"frozen-model")
    freeze = {
        "release_status": "pre_holdout_frozen_research_prototype",
        "holdout_accessed": accessed,
        "post_holdout_tuning_permitted": False,
        "artifacts": {
            "model": {
                "path": "models/model.joblib",
                "sha256": sha256_file(artifact),
            }
        },
    }
    (reports / "model_governance_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    return artifact


def test_freeze_verification_accepts_matching_hash(tmp_path: Path) -> None:
    write_freeze(tmp_path)
    freeze = verify_frozen_artifacts(report_dir=tmp_path / "reports", project_root=tmp_path)
    assert freeze["holdout_accessed"] is False


def test_freeze_verification_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = write_freeze(tmp_path)
    artifact.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_frozen_artifacts(report_dir=tmp_path / "reports", project_root=tmp_path)


def test_freeze_verification_rejects_repeat_access(tmp_path: Path) -> None:
    write_freeze(tmp_path, accessed=True)
    with pytest.raises(RuntimeError, match="already been accessed"):
        verify_frozen_artifacts(report_dir=tmp_path / "reports", project_root=tmp_path)


def test_bootstrap_intervals_are_deterministic_and_bounded() -> None:
    targets = np.array([0, 1] * 100)
    probabilities = np.where(targets == 1, 0.7, 0.1)
    first = bootstrap_confidence_intervals(
        targets, probabilities, 0.5, 5.0, 1.0, replicates=20, random_seed=7
    )
    second = bootstrap_confidence_intervals(
        targets, probabilities, 0.5, 5.0, 1.0, replicates=20, random_seed=7
    )
    assert first == second
    assert first["roc_auc"]["lower"] <= first["roc_auc"]["upper"]
