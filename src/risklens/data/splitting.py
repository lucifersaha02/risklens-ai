"""Deterministic, leakage-safe applicant-level dataset splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from risklens.config import CONFIG_DIR, METRICS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

MODELING_CONFIG_PATH = CONFIG_DIR / "modeling.yaml"
SPLIT_NAMES = ("train", "validation", "calibration", "holdout")


def load_modeling_config(path: Path = MODELING_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate split configuration."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid modeling configuration: {path}")
    proportions = config.get("splits", {})
    if set(proportions) != set(SPLIT_NAMES):
        raise ValueError(f"Splits must be exactly: {SPLIT_NAMES}")
    total = sum(float(value) for value in proportions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split proportions must sum to 1.0, received {total}")
    if any(float(value) <= 0 for value in proportions.values()):
        raise ValueError("Every split proportion must be positive")
    return config


def create_stratified_splits(
    frame: pd.DataFrame,
    random_seed: int = 42,
    primary_key: str = "SK_ID_CURR",
    target: str = "TARGET",
    proportions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assign each unique applicant to one deterministic stratified split."""
    proportions = proportions or {
        "train": 0.70,
        "validation": 0.10,
        "calibration": 0.10,
        "holdout": 0.10,
    }
    if set(proportions) != set(SPLIT_NAMES):
        raise ValueError(f"Splits must be exactly: {SPLIT_NAMES}")
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("Split proportions must sum to 1.0")

    required = {primary_key, target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split data is missing columns: {sorted(missing)}")
    if frame[primary_key].isna().any() or frame[target].isna().any():
        raise ValueError("Primary keys and targets must not contain null values")
    if frame[primary_key].duplicated().any():
        raise ValueError("Primary keys must be unique before splitting")

    base = frame[[primary_key, target]].copy()
    train, remainder = train_test_split(
        base,
        test_size=1.0 - proportions["train"],
        random_state=random_seed,
        stratify=base[target],
    )
    remainder_share = 1.0 - proportions["train"]
    validation_relative = proportions["validation"] / remainder_share
    validation, calibration_holdout = train_test_split(
        remainder,
        test_size=1.0 - validation_relative,
        random_state=random_seed,
        stratify=remainder[target],
    )
    final_share = proportions["calibration"] + proportions["holdout"]
    holdout_relative = proportions["holdout"] / final_share
    calibration, holdout = train_test_split(
        calibration_holdout,
        test_size=holdout_relative,
        random_state=random_seed,
        stratify=calibration_holdout[target],
    )

    parts = []
    for name, split_frame in (
        ("train", train),
        ("validation", validation),
        ("calibration", calibration),
        ("holdout", holdout),
    ):
        part = split_frame.copy()
        part["split"] = name
        parts.append(part)

    assignments = pd.concat(parts, ignore_index=True)
    assignments = assignments.sort_values(primary_key, ignore_index=True)
    validate_split_assignments(assignments, len(base), primary_key, target)
    return assignments


def validate_split_assignments(
    assignments: pd.DataFrame,
    expected_rows: int,
    primary_key: str = "SK_ID_CURR",
    target: str = "TARGET",
) -> None:
    """Assert split completeness, exclusivity, labels, and class preservation."""
    if len(assignments) != expected_rows:
        raise ValueError("Split assignments do not contain every applicant")
    if assignments[primary_key].duplicated().any():
        raise ValueError("An applicant appears in more than one split")
    observed = set(assignments["split"].unique())
    if observed != set(SPLIT_NAMES):
        raise ValueError(f"Unexpected split labels: {sorted(observed)}")

    overall_rate = float(assignments[target].mean())
    for name, group in assignments.groupby("split"):
        split_rate = float(group[target].mean())
        if abs(split_rate - overall_rate) > 0.002:
            raise ValueError(f"Target rate drift is too large in {name}")


def split_summary(assignments: pd.DataFrame) -> dict[str, Any]:
    """Create an auditable summary of split sizes and target rates."""
    total = len(assignments)
    summary: dict[str, Any] = {
        "total_rows": total,
        "overall_positive_rate": round(float(assignments["TARGET"].mean()), 6),
        "splits": {},
    }
    for name in SPLIT_NAMES:
        group = assignments[assignments["split"] == name]
        summary["splits"][name] = {
            "rows": int(len(group)),
            "proportion": round(len(group) / max(total, 1), 6),
            "positive_rate": round(float(group["TARGET"].mean()), 6),
            "positive_count": int(group["TARGET"].sum()),
        }
    return summary


def create_and_save_splits(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Create split assignments and persist reproducibility artifacts."""
    config = load_modeling_config(config_path)
    identifiers = config["identifiers"]
    primary_key = str(identifiers["primary_key"])
    target = str(identifiers["target"])
    seed = int(config["random_seed"])
    proportions = {str(name): float(value) for name, value in config["splits"].items()}

    frame = pd.read_csv(
        raw_data_dir / "application_train.csv",
        usecols=[primary_key, target],
    )
    assignments = create_stratified_splits(frame, seed, primary_key, target, proportions)
    summary = split_summary(assignments)
    summary["random_seed"] = seed
    summary["holdout_policy"] = config["policy"]["holdout_usage"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(processed_dir / "split_assignments.parquet", index=False)
    (metrics_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
