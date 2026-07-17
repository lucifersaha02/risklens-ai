"""Command-line interface for RiskLens AI."""

import json

import typer

from risklens.config import METRICS_DIR, ensure_output_directories
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
    typer.echo(
        f"Positive target rate: "
        f"{report['target_validation']['positive_rate']:.2%}"
    )
    typer.echo(f"Report saved to: {report_path}")


if __name__ == "__main__":
    app()