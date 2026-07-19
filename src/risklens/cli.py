"""Command-line interface for RiskLens AI."""

import json
from pathlib import Path

import typer

from risklens.config import METRICS_DIR, ensure_output_directories
from risklens.data.audit import run_data_audit
from risklens.data.splitting import create_and_save_splits
from risklens.data.validation import DataValidationError, validate_raw_dataset
from risklens.explainability.shap_analysis import build_full_history_shap_explanations
from risklens.fairness.evaluation import evaluate_responsible_ai
from risklens.fairness.full_history import evaluate_full_history_responsible_ai
from risklens.features.history import build_history_feature_store
from risklens.governance.holdout import evaluate_final_holdout
from risklens.governance.model_card import build_model_card
from risklens.modeling.baseline import train_baselines
from risklens.modeling.calibration import calibrate_candidate
from risklens.modeling.candidate import train_xgboost_candidate
from risklens.modeling.decision import define_decision_policy
from risklens.modeling.full_history import train_full_history_candidate
from risklens.modeling.full_history_calibration import (
    calibrate_full_history_candidate,
)
from risklens.modeling.full_history_decision import (
    define_full_history_decision_policy,
)
from risklens.modeling.new_application import train_new_application_simulator
from risklens.monitoring.drift import build_monitoring_baseline, monitor_test_population
from risklens.rag.assistant import answer_governance_question
from risklens.rag.knowledge_base import (
    build_knowledge_index,
    evaluate_knowledge_retrieval,
    load_knowledge_index,
    load_rag_config,
)
from risklens.serving.inference import FrozenRiskScorer

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


@app.command("define-policy")
def define_policy() -> None:
    """Lock the cost-derived decision threshold and report sensitivity."""
    ensure_output_directories()
    typer.echo("Defining the cost-sensitive decision policy...")
    report = define_decision_policy()
    metrics = report["locked_threshold_metrics"]
    typer.secho(
        f"Locked decision threshold: {report['locked_threshold']:.4f}",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Validation recall {metrics['recall']:.2%}, "
        f"precision {metrics['precision']:.2%}, "
        f"approval rate {metrics['approval_rate']:.2%}"
    )
    typer.echo(f"Expected validation cost: {metrics['cost_per_application']:.4f} units/application")
    typer.echo("Threshold was derived from documented costs; holdout was not accessed.")


@app.command("evaluate-fairness")
def evaluate_fairness() -> None:
    """Run validation-only responsible-AI subgroup diagnostics."""
    ensure_output_directories()
    typer.echo("Evaluating validation subgroup behavior...")
    report = evaluate_responsible_ai()
    for group, diagnostic in report["diagnostics"].items():
        gaps = diagnostic["gaps"]
        typer.echo(
            f"{group}: {diagnostic['eligible_groups']} eligible groups, "
            f"recall gap {gaps['recall_max_min_gap']:.2%}, "
            f"FPR gap {gaps['false_positive_rate_max_min_gap']:.2%}"
        )
    typer.secho("Responsible-AI diagnostic completed.", fg=typer.colors.GREEN)
    typer.echo("This diagnostic is not proof of fairness; holdout was not accessed.")


@app.command("build-history-features")
def build_history_features() -> None:
    """Build target-free, chunked relational history aggregates."""
    ensure_output_directories()
    typer.echo("Building the full-history feature store in memory-safe chunks...")
    report = build_history_feature_store()
    typer.secho("Full-history feature store completed.", fg=typer.colors.GREEN)
    typer.echo(f"Rows: {report['rows']:,}; history features: {report['feature_columns']:,}")
    for table, coverage in report["table_coverage"].items():
        typer.echo(f"{table} applicant coverage: {coverage:.2%}")
    typer.echo("No target values were used or stored in the feature store.")


@app.command("train-full-history")
def train_full_history() -> None:
    """Cross-validate and compare the full-history XGBoost candidate."""
    ensure_output_directories()
    typer.echo("Cross-validating the full-history XGBoost candidate...")
    report = train_full_history_candidate()
    cv_metrics = report["cross_validation"]
    validation = report["validation"]
    comparison = report["comparison"]
    typer.echo(
        f"CV ROC-AUC: {cv_metrics['roc_auc']['mean']:.4f} +/- "
        f"{cv_metrics['roc_auc']['standard_deviation']:.4f}"
    )
    typer.echo(
        f"CV PR-AUC: {cv_metrics['average_precision']['mean']:.4f} +/- "
        f"{cv_metrics['average_precision']['standard_deviation']:.4f}"
    )
    typer.echo(
        f"Validation: ROC-AUC {validation['roc_auc']:.4f}, "
        f"PR-AUC {validation['average_precision']:.4f}, "
        f"Brier {validation['brier_score']:.4f}"
    )
    pr_delta = comparison["average_precision"]["delta_full_history_minus_application"]
    typer.echo(f"PR-AUC change vs application-only: {pr_delta:+.4f}")
    typer.secho(
        f"Selected candidate: {comparison['selected_candidate']}",
        fg=typer.colors.GREEN,
    )
    typer.echo("Calibration and holdout splits were not accessed.")


@app.command("calibrate-full-history")
def calibrate_full_history() -> None:
    """Calibrate the selected full-history model without accessing holdout."""
    ensure_output_directories()
    typer.echo("Comparing full-history probability calibration methods...")
    report = calibrate_full_history_candidate()
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


@app.command("define-full-history-policy")
def define_full_history_policy() -> None:
    """Define the cost-sensitive policy for the calibrated full-history model."""
    ensure_output_directories()
    typer.echo("Defining the full-history cost-sensitive decision policy...")
    report = define_full_history_decision_policy()
    metrics = report["locked_threshold_metrics"]
    typer.secho(
        f"Locked decision threshold: {report['locked_threshold']:.4f}",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Validation recall {metrics['recall']:.2%}, "
        f"precision {metrics['precision']:.2%}, "
        f"approval rate {metrics['approval_rate']:.2%}"
    )
    typer.echo(f"Expected validation cost: {metrics['cost_per_application']:.4f} units/application")
    typer.echo("Costs are hypothetical portfolio assumptions, not lender estimates.")
    typer.echo("The final holdout was not accessed.")


@app.command("evaluate-full-history-fairness")
def evaluate_full_history_fairness() -> None:
    """Run validation subgroup diagnostics for the full-history model."""
    ensure_output_directories()
    typer.echo("Evaluating full-history validation subgroup behavior...")
    report = evaluate_full_history_responsible_ai()
    for group, diagnostic in report["diagnostics"].items():
        gaps = diagnostic["gaps"]
        typer.echo(
            f"{group}: {diagnostic['eligible_groups']} eligible groups, "
            f"recall gap {gaps['recall_max_min_gap']:.2%}, "
            f"FPR gap {gaps['false_positive_rate_max_min_gap']:.2%}"
        )
    typer.secho("Full-history responsible-AI diagnostic completed.", fg=typer.colors.GREEN)
    typer.echo("This diagnostic is not proof of fairness or legal compliance.")
    typer.echo("The final holdout was not accessed.")


@app.command("explain-full-history")
def explain_full_history() -> None:
    """Build global and applicant-level SHAP explanations."""
    ensure_output_directories()
    typer.echo("Building validation-only SHAP explanations...")
    report = build_full_history_shap_explanations()
    typer.secho("SHAP explainability artifacts completed.", fg=typer.colors.GREEN)
    typer.echo(
        f"Explained {report['sample_rows']:,} applicants across "
        f"{report['transformed_feature_count']:,} transformed features."
    )
    typer.echo(
        f"Maximum raw-margin additivity error: {report['maximum_shap_additivity_error']:.6g}"
    )
    typer.echo("SHAP explains model behavior, not causality.")
    typer.echo("The final holdout was not accessed.")


@app.command("build-model-card")
def create_model_card() -> None:
    """Freeze governed artifacts and build the pre-holdout model card."""
    ensure_output_directories()
    typer.echo("Freezing governed model artifacts and building the model card...")
    freeze = build_model_card()
    typer.secho("Model governance freeze completed.", fg=typer.colors.GREEN)
    typer.echo(f"Model: {freeze['model']}")
    typer.echo(f"Governance policy: {freeze['governance_policy']}")
    typer.echo("Candidate, calibrated-model, and configuration hashes recorded.")
    typer.echo("The final holdout remains sealed.")


@app.command("evaluate-final-holdout")
def evaluate_holdout(
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Confirm irreversible one-time access to the frozen final holdout.",
    ),
) -> None:
    """Evaluate the frozen model on final holdout exactly once."""
    if not confirm:
        typer.secho(
            "Holdout access blocked. Re-run with --confirm after reviewing the freeze.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    ensure_output_directories()
    typer.echo("Verifying frozen artifacts before one-time holdout access...")
    report = evaluate_final_holdout()
    metrics = report["probability_metrics"]
    policy = report["locked_policy_metrics"]
    typer.secho(
        "Final holdout evaluation completed and permanently recorded.", fg=typer.colors.GREEN
    )
    typer.echo(
        f"ROC-AUC {metrics['roc_auc']:.4f}, PR-AUC {metrics['average_precision']:.4f}, "
        f"Brier {metrics['brier_score']:.4f}"
    )
    typer.echo(
        f"Locked-policy recall {policy['recall']:.2%}, precision {policy['precision']:.2%}, "
        f"approval rate {policy['approval_rate']:.2%}"
    )
    typer.echo("Model development is now closed; post-holdout tuning is prohibited.")


@app.command("score-applicant")
def score_applicant(
    applicant_id: int = typer.Argument(..., min=1),
    reasons: int = typer.Option(5, "--reasons", min=1, max=20),
) -> None:
    """Score one applicant with the frozen governed model."""
    scorer = FrozenRiskScorer()
    response = scorer.score_applicant(applicant_id, reason_count=reasons)
    typer.echo(response.model_dump_json(indent=2))


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
) -> None:
    """Run the authenticated RiskLens AI FastAPI service."""
    import uvicorn

    uvicorn.run("risklens.api.main:app", host=host, port=port, reload=False)


@app.command("serve-dashboard")
def serve_dashboard(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8501, "--port", min=1, max=65535),
) -> None:
    """Run the Streamlit dashboard as an API-only client."""
    import sys

    from streamlit.web import cli as streamlit_cli

    dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise SystemExit(streamlit_cli.main())


@app.command("build-monitoring-baseline")
def create_monitoring_baseline() -> None:
    """Build the frozen validation reference for drift monitoring."""
    typer.echo("Building frozen-model monitoring reference...")
    baseline = build_monitoring_baseline()
    typer.secho("Monitoring reference completed.", fg=typer.colors.GREEN)
    typer.echo(
        f"Rows: {baseline['reference_rows']:,}; features: {baseline['monitored_feature_count']:,}"
    )
    typer.echo("No target values were used and the frozen model was not changed.")


@app.command("monitor-test-population")
def monitor_unlabeled_test_population() -> None:
    """Compare the unlabeled test population with the frozen reference."""
    typer.echo("Monitoring unlabeled Home Credit test-population drift...")
    report = monitor_test_population()
    prediction = report["prediction_drift"]
    counts = report["feature_severity_counts"]
    typer.secho(
        f"Overall monitoring severity: {report['overall_severity']}",
        fg=(typer.colors.GREEN if report["overall_severity"] == "stable" else typer.colors.YELLOW),
    )
    typer.echo(
        f"Prediction PSI {prediction['psi']:.4f} ({prediction['severity']}); "
        f"feature alerts: {counts['warning']} warning, {counts['critical']} critical"
    )
    typer.echo("Labels are unavailable, so performance drift was not measured.")
    typer.echo("Monitoring alerts do not authorize post-holdout model tuning.")


@app.command("build-knowledge-index")
def build_rag_knowledge_index() -> None:
    """Build the trusted local project-document knowledge index."""
    typer.echo("Building injection-scanned local knowledge index...")
    manifest = build_knowledge_index()
    typer.secho("Knowledge index completed.", fg=typer.colors.GREEN)
    typer.echo(
        f"Sources: {manifest['source_count']}; chunks: {manifest['chunk_count']}; "
        f"backend: {manifest['backend']}"
    )
    typer.echo("Applicant-specific queries are prohibited.")


@app.command("train-new-application-simulator")
def train_manual_application_simulator() -> None:
    """Train and freeze the separate application-only simulator release."""
    typer.echo("Training the governed new-application simulator...")
    report = train_new_application_simulator()
    metrics = report["test_calibrated"]
    typer.secho("New-application simulator release completed.", fg=typer.colors.GREEN)
    typer.echo(
        f"Internal test: ROC-AUC {metrics['roc_auc']:.4f}, "
        f"PR-AUC {metrics['average_precision']:.4f}, "
        f"Brier {metrics['brier_score']:.4f}"
    )
    typer.echo(f"Review threshold: {report['threshold']:.2%}")
    typer.echo("Only the original training partition was used; the frozen release is unchanged.")


@app.command("query-knowledge")
def query_knowledge(
    question: str = typer.Argument(...),
    top_k: int | None = typer.Option(None, "--top-k", min=1, max=20),
) -> None:
    """Retrieve cited project documentation without generating a decision."""
    config = load_rag_config()
    requested_top_k = top_k or int(config["index"]["default_top_k"])
    results = load_knowledge_index().search(
        question,
        top_k=requested_top_k,
        minimum_score=float(config["index"]["minimum_score"]),
    )
    if not results:
        typer.echo("No sufficiently relevant documentation was found.")
        raise typer.Exit(code=1)
    for rank, result in enumerate(results, start=1):
        typer.echo(f"{rank}. {result['citation']} score={result['score']:.4f}")
        typer.echo(f"   {result['text'][:400]}")


@app.command("evaluate-knowledge-retrieval")
def evaluate_rag_retrieval() -> None:
    """Evaluate citation-source retrieval on curated governance questions."""
    typer.echo("Evaluating local knowledge retrieval...")
    report = evaluate_knowledge_retrieval()
    typer.secho("Retrieval evaluation completed.", fg=typer.colors.GREEN)
    typer.echo(
        f"Hit rate@{report['top_k']}: {report['source_hit_rate_at_k']:.2%}; "
        f"MRR: {report['mean_reciprocal_rank']:.4f}"
    )


@app.command("ask-evidence-assistant")
def ask_evidence_assistant(question: str = typer.Argument(...)) -> None:
    """Return a guarded, citation-backed project evidence briefing."""
    response = answer_governance_question(question)
    typer.echo(json.dumps(response.as_dict(), indent=2))


if __name__ == "__main__":
    app()
