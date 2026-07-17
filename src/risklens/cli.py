"""Command-line interface for RiskLens AI."""

import json

import typer

from risklens.config import METRICS_DIR, ensure_output_directories
from risklens.data.audit import run_data_audit
from risklens.data.splitting import create_and_save_splits
from risklens.data.validation import DataValidationError, validate_raw_dataset

app = typer.Typer(
    name="risklens",
    help="RiskLens AI project commands.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """RiskLens AI command-line interface."""


@app.command()
def version() -> None:
    """Display the installed RiskLens AI version."""
    from risklens import __version__

    typer.echo(f"RiskLens AI {__version__}")


@app.command("validate-data")
def validate_data() -> None:
    """Validate raw data against the version-controlled data contract."""
    ensure_output_directories()

    typer.echo("Validating the Home Credit dataset...")

    try:
        report = validate_raw_dataset()
    except DataValidationError as error:
        typer.secho(f"Validation failed: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    report_path = METRICS_DIR / "raw_data_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    typer.secho(
        f"Validation passed: {report['files_checked']} files checked.",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Positive target rate: {report['target_validation']['positive_rate']:.2%}")
    typer.echo(f"Report saved to: {report_path}")


@app.command("audit-data")
def audit_data() -> None:
    """Audit application quality and relational-table coverage."""
    ensure_output_directories()
    typer.echo("Auditing application data and relational coverage...")
    report = run_data_audit()
    summary = report["application_summary"]
    typer.secho("Data audit completed.", fg=typer.colors.GREEN)
    typer.echo(f"Applications: {summary['rows']:,}")
    typer.echo(f"Positive target rate: {summary['positive_rate']:.2%}")
    typer.echo("Reports saved under reports/ and reports/metrics/.")


@app.command("create-splits")
def create_splits() -> None:
    """Create deterministic train, validation, calibration, and holdout splits."""
    ensure_output_directories()
    typer.echo("Creating leakage-safe stratified applicant splits...")
    summary = create_and_save_splits()
    typer.secho("Split assignments created.", fg=typer.colors.GREEN)
    for name, values in summary["splits"].items():
        typer.echo(f"{name}: {values['rows']:,} rows, positive rate {values['positive_rate']:.2%}")
    typer.echo("The holdout split is reserved for final evaluation only.")


if __name__ == "__main__":
    app()
