"""Command-line interface for RiskLens AI."""

import json

import typer

from risklens.config import METRICS_DIR, ensure_output_directories
from risklens.data.audit import run_data_audit
from risklens.data.splitting import create_and_save_splits
from risklens.data.validation import DataValidationError, validate_raw_dataset
from risklens.modeling.baseline import train_baselines
from risklens.modeling.calibration import calibrate_candidate
from risklens.modeling.candidate import train_xgboost_candidate

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


@app.command("train-baselines")
def train_baseline_models() -> None:
    """Train and validate application-only dummy and logistic benchmarks."""
    ensure_output_directories()
    typer.echo("Training application-only benchmark models...")
    report = train_baselines()
    for name, metrics in report["models"].items():
        typer.echo(
            f"{name}: ROC-AUC {metrics['roc_auc']:.4f}, "
            f"PR-AUC {metrics['average_precision']:.4f}, "
            f"Brier {metrics['brier_score']:.4f}"
        )
    typer.secho("Baseline training completed.", fg=typer.colors.GREEN)
    typer.echo("Calibration and holdout splits were not accessed.")


@app.command("train-candidate")
def train_candidate_model() -> None:
    """Cross-validate and evaluate the application-only XGBoost candidate."""
    ensure_output_directories()
    typer.echo("Cross-validating the XGBoost candidate on training data...")
    report = train_xgboost_candidate()
    cv_metrics = report["cross_validation"]
    validation = report["validation"]
    typer.echo(
        f"CV ROC-AUC: {cv_metrics['roc_auc']['mean']:.4f} "
        f"+/- {cv_metrics['roc_auc']['standard_deviation']:.4f}"
    )
    typer.echo(
        f"CV PR-AUC: {cv_metrics['average_precision']['mean']:.4f} "
        f"+/- {cv_metrics['average_precision']['standard_deviation']:.4f}"
    )
    typer.echo(
        f"Validation: ROC-AUC {validation['roc_auc']:.4f}, "
        f"PR-AUC {validation['average_precision']:.4f}, "
        f"Brier {validation['brier_score']:.4f}"
    )
    if "comparison" in report:
        typer.secho(
            f"Selected candidate: {report['comparison']['selected_model']}",
            fg=typer.colors.GREEN,
        )
    typer.echo("Calibration and holdout splits were not accessed.")


@app.command("calibrate-model")
def calibrate_model() -> None:
    """Select and fit probability calibration without accessing holdout."""
    ensure_output_directories()
    typer.echo("Comparing probability calibration methods...")
    report = calibrate_candidate()
    for method, metrics in report["selection_metrics"].items():
        typer.echo(
            f"{method}: Brier {metrics['brier_score']:.5f}, "
            f"log loss {metrics['log_loss']:.5f}, "
            f"ROC-AUC {metrics['roc_auc']:.4f}"
        )
    typer.secho(
        f"Selected calibration method: {report['selected_method']}",
        fg=typer.colors.GREEN,
    )
    typer.echo("Final holdout was not accessed.")


if __name__ == "__main__":
    app()
