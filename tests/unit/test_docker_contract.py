"""Static deployment-contract tests that do not require a Docker daemon."""

import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_api_extra_declares_dashboard_http_client() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    api_dependencies = config["project"]["optional-dependencies"]["api"]
    assert any(dependency.startswith("httpx") for dependency in api_dependencies)


def test_dockerfile_runs_as_non_root_and_uses_python_312() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12.10-slim" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "RISKLENS_PROJECT_ROOT=/app" in dockerfile
    assert "COPY models" not in dockerfile
    assert "COPY data" not in dockerfile


def test_compose_mounts_governed_artifacts_read_only() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert "./data:/app/data:ro" in api["volumes"]
    assert "./models:/app/models:ro" in api["volumes"]
    assert "./reports:/app/reports:ro" in api["volumes"]
    assert api["read_only"] is True
    assert api["cap_drop"] == ["ALL"]
    assert api["ports"] == ["127.0.0.1:8000:8000"]


def test_dashboard_uses_private_api_service_name_and_local_port() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    dashboard = compose["services"]["dashboard"]
    assert dashboard["environment"]["RISKLENS_API_URL"] == "http://api:8000"
    assert dashboard["ports"] == ["127.0.0.1:8501:8501"]
    assert dashboard["depends_on"]["api"]["condition"] == "service_healthy"


def test_secret_and_large_artifacts_are_excluded_from_build_context() -> None:
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", "data", "models", "reports", ".git", ".venv"):
        assert required in ignored
