"""Reproducible data-audit utilities for the Home Credit dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from risklens.config import METRICS_DIR, RAW_DATA_DIR, REPORT_DIR

APPLICATION_TRAIN = "application_train.csv"
RELATIONAL_FILES = (
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "POS_CASH_balance.csv",
)


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, descending missingness summary."""
    rows = len(frame)
    missing = frame.isna().sum()
    result = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing_count": missing.to_numpy(),
            "missing_pct": (missing.to_numpy() / max(rows, 1) * 100).round(4),
            "unique_count": [frame[column].nunique(dropna=True) for column in frame],
        }
    )
    return result.sort_values(["missing_pct", "column"], ascending=[False, True], ignore_index=True)


def application_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize the labelled application table and known data anomalies."""
    required = {"SK_ID_CURR", "TARGET"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Application data is missing columns: {sorted(missing)}")

    target_counts = frame["TARGET"].value_counts(dropna=False).sort_index()
    numeric_columns = frame.select_dtypes(include="number").columns
    categorical_columns = frame.select_dtypes(exclude="number").columns
    constant_columns = [
        column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
    ]
    missing_pct = frame.isna().mean() * 100

    days_employed_sentinel = 0
    if "DAYS_EMPLOYED" in frame:
        days_employed_sentinel = int((frame["DAYS_EMPLOYED"] == 365243).sum())

    age_min = age_max = None
    if "DAYS_BIRTH" in frame:
        ages = -frame["DAYS_BIRTH"] / 365.25
        age_min = round(float(ages.min()), 2)
        age_max = round(float(ages.max()), 2)

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "numeric_columns": int(len(numeric_columns)),
        "categorical_columns": int(len(categorical_columns)),
        "duplicate_applicant_ids": int(frame["SK_ID_CURR"].duplicated().sum()),
        "null_targets": int(frame["TARGET"].isna().sum()),
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "positive_rate": round(float(frame["TARGET"].mean()), 6),
        "constant_columns": sorted(constant_columns),
        "columns_missing_over_40_pct": int((missing_pct > 40).sum()),
        "columns_missing_over_60_pct": int((missing_pct > 60).sum()),
        "columns_missing_over_80_pct": int((missing_pct > 80).sum()),
        "columns_missing_over_95_pct": int((missing_pct > 95).sum()),
        "days_employed_365243_count": days_employed_sentinel,
        "age_years_min": age_min,
        "age_years_max": age_max,
    }


def relational_coverage(
    path: Path,
    application_ids: set[int],
    chunksize: int = 500_000,
) -> dict[str, Any]:
    """Measure applicant coverage in a large one-to-many table by streaming IDs."""
    matched_ids: set[int] = set()
    rows = 0
    for chunk in pd.read_csv(path, usecols=["SK_ID_CURR"], chunksize=chunksize):
        rows += len(chunk)
        observed = set(chunk["SK_ID_CURR"].dropna().astype(int).unique())
        matched_ids.update(observed.intersection(application_ids))

    total_applicants = len(application_ids)
    return {
        "file": path.name,
        "rows": int(rows),
        "matched_application_ids": int(len(matched_ids)),
        "application_coverage_pct": round(len(matched_ids) / max(total_applicants, 1) * 100, 4),
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    """Write a concise human-readable audit report."""
    summary = report["application_summary"]
    lines = [
        "# RiskLens AI Data Audit",
        "",
        "## Application table",
        "",
        f"- Rows: {summary['rows']:,}",
        f"- Columns: {summary['columns']:,}",
        f"- Positive target rate: {summary['positive_rate']:.2%}",
        f"- Duplicate applicant IDs: {summary['duplicate_applicant_ids']:,}",
        f"- Columns above 60% missingness: {summary['columns_missing_over_60_pct']}",
        f"- DAYS_EMPLOYED sentinel records: {summary['days_employed_365243_count']:,}",
        "",
        "## Relational coverage",
        "",
        "| Table | Rows | Applicant coverage |",
        "|---|---:|---:|",
    ]
    for item in report["relational_coverage"]:
        lines.append(
            f"| {item['file']} | {item['rows']:,} | {item['application_coverage_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The TARGET is imbalanced, so accuracy alone is not an appropriate model metric. "
            "Missing values and the DAYS_EMPLOYED sentinel must be handled inside the training "
            "pipeline using transformations learned only from training data.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_data_audit(
    raw_data_dir: Path = RAW_DATA_DIR,
    metrics_dir: Path = METRICS_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    """Run and persist the reproducible raw-data audit."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    application = pd.read_csv(raw_data_dir / APPLICATION_TRAIN)
    missingness = missingness_table(application)
    summary = application_summary(application)
    application_ids = set(application["SK_ID_CURR"].astype(int))

    coverage = [
        relational_coverage(raw_data_dir / filename, application_ids)
        for filename in RELATIONAL_FILES
    ]
    report = {
        "dataset": "Home Credit Default Risk",
        "application_summary": summary,
        "relational_coverage": coverage,
    }

    missingness.to_csv(metrics_dir / "application_missingness.csv", index=False)
    (metrics_dir / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, report_dir / "data_audit.md")
    return report
