"""Governed application-only model for manual new-application simulation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from risklens.config import CONFIG_DIR, METRICS_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from risklens.features.application import safe_ratio
from risklens.modeling.calibration import CalibratedRiskModel, SigmoidCalibrator
from risklens.modeling.metrics import evaluate_probabilities

SIMULATOR_CONFIG_PATH = CONFIG_DIR / "new_application_simulator.yaml"
SIMULATOR_MODEL_PATH = MODEL_DIR / "new_application_simulator.joblib"
SIMULATOR_METRICS_PATH = METRICS_DIR / "new_application_simulator_metrics.json"
SIMULATOR_RELEASE_PATH = METRICS_DIR.parent / "new_application_simulator_release.json"
SIMULATOR_MODEL_CARD_PATH = METRICS_DIR.parent / "new_application_simulator_model_card.md"


def load_simulator_config(path: Path = SIMULATOR_CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the simulator release configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "input_features" not in config:
        raise ValueError("New-application simulator configuration is invalid")
    proportions = config["release"]["internal_splits"]
    if abs(sum(float(value) for value in proportions.values()) - 1.0) > 1e-9:
        raise ValueError("Simulator internal split proportions must sum to one")
    return config


def simulator_input_columns(config: dict[str, Any]) -> list[str]:
    """Return the ordered allowlist of raw application-time model inputs."""
    features = config["input_features"]
    return [*features["numeric"], *features["categorical"]]


def add_simulator_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic application-only features without sensitive attributes."""
    result = frame.copy()
    result["EMPLOYMENT_YEARS"] = (-result["DAYS_EMPLOYED"] / 365.25).clip(lower=0)
    result["CREDIT_INCOME_RATIO"] = safe_ratio(result["AMT_CREDIT"], result["AMT_INCOME_TOTAL"])
    result["ANNUITY_INCOME_RATIO"] = safe_ratio(result["AMT_ANNUITY"], result["AMT_INCOME_TOTAL"])
    result["CREDIT_ANNUITY_RATIO"] = safe_ratio(result["AMT_CREDIT"], result["AMT_ANNUITY"])
    result["GOODS_CREDIT_RATIO"] = safe_ratio(result["AMT_GOODS_PRICE"], result["AMT_CREDIT"])
    external = result[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    result["EXT_SOURCE_MEAN"] = external.mean(axis=1)
    result["EXT_SOURCE_MIN"] = external.min(axis=1)
    result["EXT_SOURCE_MAX"] = external.max(axis=1)
    result["EXT_SOURCE_COUNT"] = external.notna().sum(axis=1)
    return result


class SimulatorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Pickle-safe stateless simulator transformer."""

    def fit(self, frame: pd.DataFrame, target: pd.Series | None = None) -> SimulatorFeatureEngineer:
        del target
        add_simulator_features(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return add_simulator_features(frame)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {}


def _build_pipeline(sample: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    engineered = add_simulator_features(sample)
    numeric = engineered.select_dtypes(include="number").columns.tolist()
    categorical = engineered.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    values = config["model"]
    model = XGBClassifier(
        n_estimators=int(values["n_estimators"]),
        max_depth=int(values["max_depth"]),
        learning_rate=float(values["learning_rate"]),
        subsample=float(values["subsample"]),
        colsample_bytree=float(values["colsample_bytree"]),
        min_child_weight=float(values["min_child_weight"]),
        reg_lambda=float(values["reg_lambda"]),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(config["release"]["random_seed"]),
        n_jobs=-1,
    )
    return Pipeline(
        [("features", SimulatorFeatureEngineer()), ("preprocessor", preprocessor), ("model", model)]
    )


def _internal_splits(
    frame: pd.DataFrame, target: pd.Series, seed: int
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    development, test, development_y, test_y = train_test_split(
        frame, target, test_size=0.10, random_state=seed, stratify=target
    )
    train, remainder, train_y, remainder_y = train_test_split(
        development,
        development_y,
        test_size=2 / 9,
        random_state=seed,
        stratify=development_y,
    )
    validation, calibration, validation_y, calibration_y = train_test_split(
        remainder, remainder_y, test_size=0.50, random_state=seed, stratify=remainder_y
    )
    return {
        "train": (train, train_y),
        "validation": (validation, validation_y),
        "calibration": (calibration, calibration_y),
        "test": (test, test_y),
    }


def _training_reference(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    numeric = config["input_features"]["numeric"]
    categorical = config["input_features"]["categorical"]
    return {
        "numeric_ranges": {
            column: {
                "p01": float(frame[column].quantile(0.01)),
                "p99": float(frame[column].quantile(0.99)),
            }
            for column in numeric
        },
        "categorical_levels": {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in categorical
        },
    }


@dataclass
class NewApplicationArtifact:
    """Serializable simulator model plus its data and governance contract."""

    model: CalibratedRiskModel
    threshold: float
    input_columns: list[str]
    reference: dict[str, Any]
    release_name: str
    excluded_features: list[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_new_application_simulator(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    model_path: Path = SIMULATOR_MODEL_PATH,
    metrics_path: Path = SIMULATOR_METRICS_PATH,
    release_path: Path = SIMULATOR_RELEASE_PATH,
    model_card_path: Path = SIMULATOR_MODEL_CARD_PATH,
    config_path: Path = SIMULATOR_CONFIG_PATH,
) -> dict[str, Any]:
    """Train, calibrate, evaluate, and freeze a separate manual-input release."""
    config = load_simulator_config(config_path)
    columns = simulator_input_columns(config)
    assignments = pd.read_parquet(processed_data_dir / "split_assignments.parquet")
    parent_ids = assignments.loc[assignments["split"] == "train", ["SK_ID_CURR"]]
    application = pd.read_csv(
        raw_data_dir / "application_train.csv", usecols=["SK_ID_CURR", "TARGET", *columns]
    )
    parent = application.merge(parent_ids, on="SK_ID_CURR", how="inner", validate="one_to_one")
    target = parent.pop("TARGET").astype(int)
    frame = parent[columns].copy()
    seed = int(config["release"]["random_seed"])
    splits = _internal_splits(frame, target, seed)
    train, train_y = splits["train"]
    validation, validation_y = splits["validation"]
    calibration, calibration_y = splits["calibration"]
    test, test_y = splits["test"]

    pipeline = _build_pipeline(train.iloc[:1000], config)
    pipeline.fit(train, train_y)
    raw_validation = pipeline.predict_proba(validation)[:, 1]
    calibrator = SigmoidCalibrator().fit(
        pipeline.predict_proba(calibration)[:, 1], calibration_y.to_numpy()
    )
    calibrated = CalibratedRiskModel(pipeline, calibrator, "sigmoid")
    threshold = float(config["decision_policy"]["threshold"])
    test_probabilities = calibrated.predict_proba(test)[:, 1]
    report = {
        "release_name": config["release"]["name"],
        "assessment_mode": "application_only_manual_simulation",
        "parent_data_partition": "train",
        "original_validation_calibration_holdout_accessed": False,
        "rows": {name: len(values[0]) for name, values in splits.items()},
        "validation_raw": evaluate_probabilities(
            validation_y.to_numpy(), raw_validation, threshold
        ),
        "test_calibrated": evaluate_probabilities(test_y.to_numpy(), test_probabilities, threshold),
        "calibration_method": "sigmoid",
        "threshold": threshold,
        "excluded_features": config["excluded_features"],
        "limitations": [
            "Research simulator trained on the public Home Credit dataset.",
            "Application-only assessment does not use bureau or repayment-history tables.",
            "Output supports human review and is not an approval or decline decision.",
        ],
    }
    artifact = NewApplicationArtifact(
        model=calibrated,
        threshold=threshold,
        input_columns=columns,
        reference=_training_reference(train, config),
        release_name=str(config["release"]["name"]),
        excluded_features=list(config["excluded_features"]),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report["artifact_sha256"] = _sha256_file(model_path)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    release_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics = report["test_calibrated"]
    model_card_path.write_text(
        "\n".join(
            [
                "# RiskLens AI New Application Simulator",
                "",
                f"- Release: `{report['release_name']}`",
                "- Assessment mode: application-only manual simulation",
                "- Parent data: original training partition only",
                "- Original validation, calibration, and holdout accessed: **No**",
                f"- Test ROC-AUC: {metrics['roc_auc']:.4f}",
                f"- Test PR-AUC: {metrics['average_precision']:.4f}",
                f"- Test Brier score: {metrics['brier_score']:.4f}",
                f"- Review threshold: {threshold:.2%}",
                "",
                "This research simulator estimates Home Credit payment-difficulty risk. ",
                "It does not approve or decline loans, and human review is required.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report
