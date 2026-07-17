"""Unit tests for raw-data validation."""

from pathlib import Path

import pytest

from risklens.data.validation import (
    DataValidationError,
    count_csv_rows,
    read_csv_header,
    validate_file,
)


def test_count_csv_rows_with_trailing_newline(tmp_path: Path) -> None:
    """Count data rows while excluding the CSV header."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "id,value\n1,alpha\n2,beta\n",
        encoding="utf-8",
    )

    assert count_csv_rows(csv_path) == 2


def test_count_csv_rows_without_trailing_newline(tmp_path: Path) -> None:
    """Count the final record when the file has no trailing newline."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "id,value\n1,alpha\n2,beta",
        encoding="utf-8",
    )

    assert count_csv_rows(csv_path) == 2


def test_read_csv_header_supports_utf8(tmp_path: Path) -> None:
    """Read a normal UTF-8 CSV header."""
    csv_path = tmp_path / "utf8.csv"
    csv_path.write_text(
        "id,description\n1,Applicant\n",
        encoding="utf-8",
    )

    assert read_csv_header(csv_path) == ["id", "description"]


def test_read_csv_header_falls_back_to_cp1252(tmp_path: Path) -> None:
    """Read source files containing Windows cp1252 characters."""
    csv_path = tmp_path / "windows_encoded.csv"
    csv_path.write_text(
        "Table,Row,Description\n"
        "application_train,AMT_CREDIT,Credit…amount\n",
        encoding="cp1252",
    )

    assert read_csv_header(csv_path) == [
        "Table",
        "Row",
        "Description",
    ]


def test_validate_file_rejects_missing_columns(tmp_path: Path) -> None:
    """Reject a CSV that does not satisfy its column contract."""
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        "id,value\n1,100\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DataValidationError,
        match="missing required columns",
    ):
        validate_file(
            path=csv_path,
            minimum_rows=1,
            required_columns=["id", "TARGET"],
        )


def test_validate_file_returns_audit_metadata(tmp_path: Path) -> None:
    """Return useful audit metadata for a valid file."""
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(
        "id,TARGET\n1,0\n2,1\n",
        encoding="utf-8",
    )

    result = validate_file(
        path=csv_path,
        minimum_rows=2,
        required_columns=["id", "TARGET"],
    )

    assert result["status"] == "passed"
    assert result["rows"] == 2
    assert result["columns"] == 2