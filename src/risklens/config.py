"""Central filesystem configuration for RiskLens AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "home_credit"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

POLICY_DIR = PROJECT_ROOT / "policies"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRICS_DIR = REPORT_DIR / "metrics"


def ensure_output_directories() -> None:
    """Create directories used for generated project artifacts."""
    for directory in (
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
