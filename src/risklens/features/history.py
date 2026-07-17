"""Chunked relational feature engineering for the full-history model."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risklens.config import INTERIM_DATA_DIR, METRICS_DIR, RAW_DATA_DIR
from risklens.data.splitting import MODELING_CONFIG_PATH, load_modeling_config

ChunkTransform = Callable[[pd.DataFrame], pd.DataFrame]


def _combine_partial_aggregates(
    current: pd.DataFrame | None, partial: pd.DataFrame
) -> pd.DataFrame:
    """Combine chunk-level sufficient statistics without averaging averages."""
    if current is None:
        return partial
    combined = pd.concat([current, partial], axis=0)
    aggregation = {
        column: (
            "min" if column.endswith("__min") else "max" if column.endswith("__max") else "sum"
        )
        for column in combined.columns
    }
    return combined.groupby(level=0, sort=False).agg(aggregation)


def aggregate_csv_in_chunks(
    path: Path,
    prefix: str,
    source_columns: list[str],
    numeric_columns: list[str],
    applicant_ids: set[int] | None = None,
    transform: ChunkTransform | None = None,
    chunksize: int = 500_000,
    key: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Aggregate a one-to-many CSV using mergeable sufficient statistics."""
    state: pd.DataFrame | None = None
    usecols = list(dict.fromkeys([key, *source_columns]))
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        if applicant_ids is not None:
            chunk = chunk[chunk[key].isin(applicant_ids)]
        if chunk.empty:
            continue
        if transform is not None:
            chunk = transform(chunk)
        missing = set(numeric_columns) - set(chunk.columns)
        if missing:
            raise ValueError(f"{path.name} transform is missing columns: {sorted(missing)}")

        grouped = chunk.groupby(key, sort=False)[numeric_columns].agg(
            ["sum", "count", "min", "max"]
        )
        grouped.columns = [f"{column}__{stat}" for column, stat in grouped.columns]
        grouped["__record_count"] = chunk.groupby(key, sort=False).size()
        state = _combine_partial_aggregates(state, grouped)

    record_column = f"{prefix}_RECORD_COUNT".upper()
    if state is None:
        return pd.DataFrame(columns=[key, record_column])

    output: dict[str, Any] = {
        key: state.index.to_numpy(),
        record_column: state["__record_count"].to_numpy(dtype="int64"),
    }
    for column in numeric_columns:
        sums = state[f"{column}__sum"]
        counts = state[f"{column}__count"]
        base = f"{prefix}_{column}".upper()
        output[f"{base}_SUM"] = sums.to_numpy()
        output[f"{base}_COUNT"] = counts.to_numpy(dtype="int64")
        output[f"{base}_MEAN"] = (sums / counts.replace(0, np.nan)).to_numpy()
        output[f"{base}_MIN"] = state[f"{column}__min"].to_numpy()
        output[f"{base}_MAX"] = state[f"{column}__max"].to_numpy()
    return pd.DataFrame(output)


def transform_bureau(chunk: pd.DataFrame) -> pd.DataFrame:
    """Add bureau-status flags available from historical credit records."""
    result = chunk.copy()
    result["ACTIVE_CREDIT_FLAG"] = (result["CREDIT_ACTIVE"] == "Active").astype("int8")
    result["CLOSED_CREDIT_FLAG"] = (result["CREDIT_ACTIVE"] == "Closed").astype("int8")
    result["OVERDUE_FLAG"] = (result["CREDIT_DAY_OVERDUE"].fillna(0) > 0).astype("int8")
    return result


def transform_previous(chunk: pd.DataFrame) -> pd.DataFrame:
    """Add previous-application status flags."""
    result = chunk.copy()
    result["APPROVED_FLAG"] = (result["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
    result["REFUSED_FLAG"] = (result["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
    return result


def transform_installments(chunk: pd.DataFrame) -> pd.DataFrame:
    """Create repayment timeliness and shortfall features."""
    result = chunk.copy()
    result["DAYS_LATE"] = (result["DAYS_ENTRY_PAYMENT"] - result["DAYS_INSTALMENT"]).clip(lower=0)
    result["PAYMENT_SHORTFALL"] = (result["AMT_INSTALMENT"] - result["AMT_PAYMENT"]).clip(lower=0)
    result["PAYMENT_RATIO"] = (
        result["AMT_PAYMENT"] / result["AMT_INSTALMENT"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    return result


def transform_credit_card(chunk: pd.DataFrame) -> pd.DataFrame:
    """Create credit-card utilization and delinquency features."""
    result = chunk.copy()
    result["UTILIZATION_RATIO"] = (
        result["AMT_BALANCE"] / result["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    result["DPD_FLAG"] = (result["SK_DPD"].fillna(0) > 0).astype("int8")
    return result


def transform_pos(chunk: pd.DataFrame) -> pd.DataFrame:
    """Create POS/CASH delinquency indicators."""
    result = chunk.copy()
    result["DPD_FLAG"] = (result["SK_DPD"].fillna(0) > 0).astype("int8")
    result["DPD_DEF_FLAG"] = (result["SK_DPD_DEF"].fillna(0) > 0).astype("int8")
    return result


def history_table_specs() -> list[dict[str, Any]]:
    """Return version-controlled relational aggregation specifications."""
    return [
        {
            "file": "bureau.csv",
            "prefix": "BUREAU",
            "source": [
                "CREDIT_ACTIVE",
                "DAYS_CREDIT",
                "CREDIT_DAY_OVERDUE",
                "DAYS_CREDIT_ENDDATE",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE",
                "CNT_CREDIT_PROLONG",
            ],
            "numeric": [
                "DAYS_CREDIT",
                "CREDIT_DAY_OVERDUE",
                "DAYS_CREDIT_ENDDATE",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE",
                "CNT_CREDIT_PROLONG",
                "ACTIVE_CREDIT_FLAG",
                "CLOSED_CREDIT_FLAG",
                "OVERDUE_FLAG",
            ],
            "transform": transform_bureau,
        },
        {
            "file": "previous_application.csv",
            "prefix": "PREVIOUS",
            "source": [
                "NAME_CONTRACT_STATUS",
                "AMT_ANNUITY",
                "AMT_APPLICATION",
                "AMT_CREDIT",
                "AMT_DOWN_PAYMENT",
                "DAYS_DECISION",
                "CNT_PAYMENT",
            ],
            "numeric": [
                "AMT_ANNUITY",
                "AMT_APPLICATION",
                "AMT_CREDIT",
                "AMT_DOWN_PAYMENT",
                "DAYS_DECISION",
                "CNT_PAYMENT",
                "APPROVED_FLAG",
                "REFUSED_FLAG",
            ],
            "transform": transform_previous,
        },
        {
            "file": "installments_payments.csv",
            "prefix": "INSTALLMENTS",
            "source": [
                "DAYS_INSTALMENT",
                "DAYS_ENTRY_PAYMENT",
                "AMT_INSTALMENT",
                "AMT_PAYMENT",
            ],
            "numeric": [
                "DAYS_LATE",
                "PAYMENT_SHORTFALL",
                "PAYMENT_RATIO",
                "AMT_INSTALMENT",
                "AMT_PAYMENT",
            ],
            "transform": transform_installments,
        },
        {
            "file": "credit_card_balance.csv",
            "prefix": "CREDIT_CARD",
            "source": [
                "AMT_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL",
                "AMT_DRAWINGS_CURRENT",
                "AMT_PAYMENT_CURRENT",
                "SK_DPD",
            ],
            "numeric": [
                "AMT_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL",
                "AMT_DRAWINGS_CURRENT",
                "AMT_PAYMENT_CURRENT",
                "SK_DPD",
                "UTILIZATION_RATIO",
                "DPD_FLAG",
            ],
            "transform": transform_credit_card,
        },
        {
            "file": "POS_CASH_balance.csv",
            "prefix": "POS",
            "source": [
                "MONTHS_BALANCE",
                "SK_DPD",
                "SK_DPD_DEF",
                "CNT_INSTALMENT",
                "CNT_INSTALMENT_FUTURE",
            ],
            "numeric": [
                "MONTHS_BALANCE",
                "SK_DPD",
                "SK_DPD_DEF",
                "CNT_INSTALMENT",
                "CNT_INSTALMENT_FUTURE",
                "DPD_FLAG",
                "DPD_DEF_FLAG",
            ],
            "transform": transform_pos,
        },
    ]


def assemble_history_feature_store(
    applicants: pd.DataFrame,
    aggregate_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Left-join history aggregates and add explicit availability indicators."""
    if "SK_ID_CURR" not in applicants or applicants["SK_ID_CURR"].duplicated().any():
        raise ValueError("Applicants must contain unique SK_ID_CURR values")
    result = applicants[["SK_ID_CURR"]].copy()
    presence_columns = []
    for prefix, aggregate in aggregate_tables:
        result = result.merge(
            aggregate,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )
        record_column = f"{prefix}_RECORD_COUNT".upper()
        presence_column = f"{prefix}_HISTORY_AVAILABLE".upper()
        result[record_column] = result[record_column].fillna(0).astype("int64")
        result[presence_column] = (result[record_column] > 0).astype("int8")
        presence_columns.append(presence_column)
    result["HISTORY_TABLE_COUNT"] = result[presence_columns].sum(axis=1).astype("int8")
    return result


def build_history_feature_store(
    raw_data_dir: Path = RAW_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    metrics_dir: Path = METRICS_DIR,
    config_path: Path = MODELING_CONFIG_PATH,
) -> dict[str, Any]:
    """Build and persist target-free history features for train and test applicants."""
    config = load_modeling_config(config_path)
    feature_config = config["feature_store"]
    chunksize = int(feature_config["chunk_size"])
    train_ids = pd.read_csv(raw_data_dir / "application_train.csv", usecols=["SK_ID_CURR"])
    test_ids = pd.read_csv(raw_data_dir / "application_test.csv", usecols=["SK_ID_CURR"])
    applicants = pd.concat([train_ids, test_ids], ignore_index=True)
    if applicants["SK_ID_CURR"].duplicated().any():
        raise ValueError("Application train and test IDs must be globally unique")
    applicant_ids = set(applicants["SK_ID_CURR"].astype(int))

    aggregates: list[tuple[str, pd.DataFrame]] = []
    coverage: dict[str, float] = {}
    for spec in history_table_specs():
        prefix = str(spec["prefix"])
        aggregate = aggregate_csv_in_chunks(
            raw_data_dir / str(spec["file"]),
            prefix=prefix,
            source_columns=list(spec["source"]),
            numeric_columns=list(spec["numeric"]),
            applicant_ids=applicant_ids,
            transform=spec["transform"],
            chunksize=chunksize,
        )
        aggregates.append((prefix, aggregate))
        coverage[prefix] = round(len(aggregate) / len(applicants), 6)

    feature_store = assemble_history_feature_store(applicants, aggregates)
    output_path = interim_dir / str(feature_config["output_file"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(output_path, index=False)
    report = {
        "rows": int(len(feature_store)),
        "feature_columns": int(feature_store.shape[1] - 1),
        "target_columns": 0,
        "table_coverage": coverage,
        "output_path": str(output_path),
        "data_policy": "target_free_features_with_explicit_history_availability",
    }
    (metrics_dir / "history_feature_store_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
