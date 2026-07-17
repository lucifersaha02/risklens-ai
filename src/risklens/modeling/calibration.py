"""Probability calibration with an untouched final holdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from risklens.config import METRICS_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config
from risklens.modeling.candidate import CANDIDATE_MODEL_PATH
from risklens.modeling.metrics import evaluate_probabilities

CALIBRATED_MODEL_PATH = MODEL_DIR / "application_xgboost_calibrated.joblib"


def probability_logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Convert probabilities to finite log-odds for sigmoid calibration."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


class SigmoidCalibrator:
    """Platt-style sigmoid mapping fitted on base-model log-odds."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000)

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> SigmoidCalibrator:
        """Fit the sigmoid mapping."""
        self.model.fit(probability_logit(probabilities).reshape(-1, 1), targets)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities."""
        return self.model.predict_proba(probability_logit(probabilities).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Non-parametric monotonic probability mapping."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> IsotonicCalibrator:
        """Fit the monotonic mapping."""
        self.model.fit(np.asarray(probabilities, dtype=float), targets)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities."""
        return np.asarray(self.model.predict(probabilities), dtype=float)


class IdentityCalibrator:
    """No-op mapping retained when learned calibration does not improve error."""

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> IdentityCalibrator:
        """Accept the calibration interface without learning parameters."""
        del probabilities, targets
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return the original probabilities."""
        return np.asarray(probabilities, dtype=float)


def build_calibrator(
    method: str,
) -> IdentityCalibrator | SigmoidCalibrator | IsotonicCalibrator:
    """Construct one supported probability calibrator."""
    if method == "sigmoid":
        return SigmoidCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "uncalibrated":
        return IdentityCalibrator()
    raise ValueError(f"Unsupported calibration method: {method}")


class CalibratedRiskModel:
    """Serializable scoring wrapper around a fitted model and calibrator."""

    def __init__(self, base_model: Any, calibrator: Any, method: str) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return two-column calibrated class probabilities."""
        raw = self.base_model.predict_proba(frame)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, frame: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Return thresholded predictions."""
        return (self.predict_proba(frame)[:, 1] >= threshold).astype(int)


def load_calibration_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load only applicants assigned to the calibration split."""
    assignments_path = processed_dir / "split_assignments.parquet"
    if not assignments_path.exists():
        raise FileNotFoundError("Run `risklens create-splits` before calibration")
    assignments = pd.read_parquet(assignments_path)
    calibration_ids = assignments.loc[assignments["split"] == "calibration", ["SK_ID_CURR"]]
    application = pd.read_csv(raw_data_dir / "application_train.csv")
    calibration = application.merge(
        calibration_ids,
        on="SK_ID_CURR",
        how="inner",
        validate="one_to_one",
    )
    if len(calibration) != len(calibration_ids):
        raise ValueError("Calibration records do not match split assignments")
    return calibration


def select_calibration_method(
    method_metrics: dict[str, dict[str, Any]],
    primary_metric: str = "brier_score",
    secondary_metric: str = "log_loss",
) -> str:
    """Select the lowest-error calibration method with deterministic tie-breaking."""
    if not method_metrics:
        raise ValueError("No calibration metrics were provided")
    return min(
        method_metrics,
        key=lambda method: (
            method_metrics[method][primary_metric],
            method_metrics[method][secondary_metric],
            method,
        ),
    )


def calibrate_candidate(
    model_dir: Path = MODEL_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Select calibration on a subsplit, then refit it on all calibration data."""
    config = load_modeling_config(config_path)
    calibration_config = config["calibration"]
    random_seed = int(config["random_seed"])
    threshold = float(config["baseline"]["decision_threshold"])
    selection_fraction = float(calibration_config["selection_fraction"])

    candidate_path = model_dir / CANDIDATE_MODEL_PATH.name
    if not candidate_path.exists():
        raise FileNotFoundError("Run `risklens train-candidate` before calibration")
    base_model = joblib.load(candidate_path)
    calibration = load_calibration_data()
    targets = calibration.pop("TARGET").astype(int)
    fit_frame, selection_frame, fit_targets, selection_targets = train_test_split(
        calibration,
        targets,
        test_size=selection_fraction,
        random_state=random_seed,
        stratify=targets,
    )

    fit_raw = base_model.predict_proba(fit_frame)[:, 1]
    selection_raw = base_model.predict_proba(selection_frame)[:, 1]
    selection_metrics: dict[str, dict[str, Any]] = {
        "uncalibrated": evaluate_probabilities(
            selection_targets.to_numpy(), selection_raw, threshold
        )
    }
    for method in calibration_config["methods"]:
        calibrator = build_calibrator(str(method)).fit(fit_raw, fit_targets.to_numpy())
        calibrated = calibrator.predict(selection_raw)
        selection_metrics[str(method)] = evaluate_probabilities(
            selection_targets.to_numpy(), calibrated, threshold
        )

    selected_method = select_calibration_method(
        selection_metrics,
        primary_metric=str(calibration_config["primary_metric"]),
        secondary_metric=str(calibration_config["secondary_metric"]),
    )

    all_raw = base_model.predict_proba(calibration)[:, 1]
    final_calibrator = build_calibrator(selected_method).fit(all_raw, targets.to_numpy())
    calibrated_model = CalibratedRiskModel(base_model, final_calibrator, selected_method)
    report = {
        "base_model": "application_xgboost_candidate",
        "calibration_rows": int(len(calibration)),
        "calibration_fit_rows": int(len(fit_frame)),
        "calibration_selection_rows": int(len(selection_frame)),
        "selection_metrics": selection_metrics,
        "selected_method": selected_method,
        "selection_primary_metric": calibration_config["primary_metric"],
        "selection_secondary_metric": calibration_config["secondary_metric"],
        "data_policy": "calibration_internal_selection_holdout_sealed",
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, model_dir / CALIBRATED_MODEL_PATH.name)
    (metrics_dir / "application_calibration_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
