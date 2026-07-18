"""Tests for model-card rendering and artifact hashing."""

from pathlib import Path

from risklens.governance.model_card import render_model_card, sha256_file


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"risklens")
    original_hash = sha256_file(artifact)
    assert original_hash == sha256_file(artifact)
    artifact.write_bytes(b"changed")
    assert original_hash != sha256_file(artifact)


def test_model_card_contains_governance_and_holdout_controls() -> None:
    context = {
        "candidate": {
            "training_rows": 100,
            "validation_rows": 20,
            "validation": {
                "roc_auc": 0.78,
                "average_precision": 0.27,
                "brier_score": 0.06,
                "log_loss": 0.23,
            },
            "cross_validation": {
                "roc_auc": {"mean": 0.77, "standard_deviation": 0.01},
                "average_precision": {"mean": 0.26, "standard_deviation": 0.02},
            },
            "feature_governance": {
                "policy_name": "audit_only_v1",
                "excluded_decision_features": ["CODE_GENDER"],
                "rationale": "Sensitive attributes are audit-only.",
            },
        },
        "calibration": {
            "calibration_rows": 20,
            "calibration_selection_rows": 10,
            "selected_method": "sigmoid",
        },
        "policy": {
            "locked_threshold": 1 / 6,
            "false_negative_cost": 5,
            "false_positive_cost": 1,
            "locked_threshold_metrics": {
                "recall": 0.42,
                "precision": 0.27,
                "approval_rate": 0.87,
                "review_or_decline_rate": 0.13,
                "cost_per_application": 0.32,
            },
        },
        "fairness": {
            "diagnostics": {
                "CODE_GENDER": {
                    "gaps": {
                        "recall_max_min_gap": 0.08,
                        "false_positive_rate_max_min_gap": 0.04,
                    }
                },
                "AGE_BAND": {
                    "gaps": {
                        "recall_max_min_gap": 0.37,
                        "false_positive_rate_max_min_gap": 0.19,
                    }
                },
            }
        },
        "shap_summary": {"sample_rows": 100, "transformed_feature_count": 50},
        "top_features": ["EXT_SOURCE_MEAN", "AMT_ANNUITY"],
    }
    card = render_model_card(context)
    assert "Not approved for production lending" in card
    assert "`CODE_GENDER`" in card
    assert "must fail if those hashes no longer match" in card
