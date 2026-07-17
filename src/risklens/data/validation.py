"""Data-contract validation for the Home Credit dataset."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from risklens.config import CONFIG_DIR, RAW_DATA_DIR

CONTRACT_PATH = CONFIG_DIR / "data_contract.yaml"


class DataValidationError(Exception):
    """Raised when the raw dataset violates its contract."""


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the version-controlled dataset contract."""
    with path.open("r", encoding="utf-8") as file:
        contract = yaml.safe_load(file)

    if not isinstance(contract, dict):
        raise DataValidationError(f"Invalid data contract: {path}")

    return contract


def read_csv_header(path: Path) -> list[str]:
    """Read a CSV header, falling back for Windows-encoded source files."""
    try:
        return pd.read_csv(
            path,
            nrows=0,
            encoding="utf-8-sig",
        ).columns.tolist()
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                nrows=0,
                encoding="cp1252",
            ).columns.tolist()
        except Exception as error:
            raise DataValidationError(
                f"Could not read the header of {path.name}: {error}"
            ) from error
    except Exception as error:
        raise DataValidationError(
            f"Could not read the header of {path.name}: {error}"
        ) from error


def count_csv_rows(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> int:
    """Count CSV data rows without loading the file into memory."""
    newline_count = 0
    last_byte = b""

    try:
        with path.open("rb") as file:
            while chunk := file.read(chunk_size):
                newline_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
    except OSError as error:
        raise DataValidationError(
            f"Could not read {path.name}: {error}"
        ) from error

    if newline_count == 0:
        return 0

    # A file ending with a newline has one line per newline.
    # Otherwise, its final record must be counted separately.
    total_lines = (
        newline_count
        if last_byte == b"\n"
        else newline_count + 1
    )

    # Exclude the CSV header.
    return max(total_lines - 1, 0)


def validate_file(
    path: Path,
    minimum_rows: int,
    required_columns: list[str],
) -> dict[str, Any]:
    """Validate the existence, header and minimum row count of one CSV."""
    if not path.exists():
        raise DataValidationError(f"Missing required file: {path.name}")

    header = read_csv_header(path)
    missing_columns = sorted(set(required_columns) - set(header))

    if missing_columns:
        raise DataValidationError(
            f"{path.name} is missing required columns: {missing_columns}"
        )

    row_count = count_csv_rows(path)

    if row_count < minimum_rows:
        raise DataValidationError(
            f"{path.name} contains {row_count:,} rows; "
            f"expected at least {minimum_rows:,}"
        )

    return {
        "file": path.name,
        "rows": row_count,
        "columns": len(header),
        "size_mb": round(path.stat().st_size / (1024**2), 2),
        "status": "passed",
    }


def validate_training_target(
    path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate training primary-key uniqueness and target distribution."""
    dataset_config = contract["dataset"]
    validation_config = contract["validation"]

    primary_key = dataset_config["primary_key"]
    target = dataset_config["target"]

    frame = pd.read_csv(path, usecols=[primary_key, target])

    duplicate_count = int(frame[primary_key].duplicated().sum())
    maximum_duplicates = validation_config["maximum_duplicate_primary_keys"]

    if duplicate_count > maximum_duplicates:
        raise DataValidationError(
            f"{path.name} contains {duplicate_count:,} duplicate {primary_key} values"
        )

    observed_targets = set(frame[target].dropna().astype(int).unique().tolist())
    allowed_targets = set(validation_config["allowed_target_values"])

    if not observed_targets.issubset(allowed_targets):
        raise DataValidationError(
            f"Unexpected target values: {sorted(observed_targets - allowed_targets)}"
        )

    positive_rate = float(frame[target].mean())
    expected_rate = validation_config["expected_positive_rate"]

    if not expected_rate["minimum"] <= positive_rate <= expected_rate["maximum"]:
        raise DataValidationError(
            f"Positive target rate {positive_rate:.4f} is outside the expected range"
        )

    return {
        "primary_key": primary_key,
        "duplicate_primary_keys": duplicate_count,
        "target": target,
        "target_values": sorted(observed_targets),
        "positive_rate": round(positive_rate, 6),
        "status": "passed",
    }


def validate_raw_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate the complete raw Home Credit dataset."""
    contract = load_contract(contract_path)
    file_results: list[dict[str, Any]] = []

    for filename, rules in contract["expected_files"].items():
        result = validate_file(
            path=raw_data_dir / filename,
            minimum_rows=int(rules["minimum_rows"]),
            required_columns=list(rules["required_columns"]),
        )
        file_results.append(result)

    target_result = validate_training_target(
        raw_data_dir / "application_train.csv",
        contract,
    )

    return {
        "dataset": contract["dataset"]["name"],
        "raw_data_directory": str(raw_data_dir),
        "files_checked": len(file_results),
        "files": file_results,
        "target_validation": target_result,
        "status": "passed",
    }
