"""Typed HTTP client for the RiskLens AI dashboard."""

from __future__ import annotations

from typing import Any

import httpx

from risklens.serving.schemas import (
    EvidenceAssistantResponse,
    ModelInfoResponse,
    MonitoringSummaryResponse,
    NewApplicationRequest,
    NewApplicationResponse,
    PortfolioSummaryResponse,
    PredictionResponse,
)


class RiskLensAPIError(RuntimeError):
    """Sanitized API or transport failure suitable for dashboard display."""


class RiskLensAPIClient:
    """Small synchronous client keeping the dashboard behind the API boundary."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an authenticated GET and normalize failures."""
        try:
            with httpx.Client(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get(path, params=params)
            response.raise_for_status()
            return dict(response.json())
        except httpx.HTTPStatusError as error:
            payload = error.response.json()
            detail = payload.get("detail", payload.get("error", "API request failed"))
            raise RiskLensAPIError(str(detail)) from error
        except (httpx.RequestError, ValueError) as error:
            raise RiskLensAPIError(
                "RiskLens API is unavailable or returned invalid data"
            ) from error

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute an authenticated JSON POST and normalize failures."""
        try:
            with httpx.Client(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(path, json=payload)
            response.raise_for_status()
            return dict(response.json())
        except httpx.HTTPStatusError as error:
            payload = error.response.json()
            detail = payload.get("detail", payload.get("error", "API request failed"))
            raise RiskLensAPIError(str(detail)) from error
        except (httpx.RequestError, ValueError) as error:
            raise RiskLensAPIError(
                "RiskLens API is unavailable or returned invalid data"
            ) from error

    def health(self) -> dict[str, Any]:
        """Return API readiness data."""
        return self._get("/health")

    def model_info(self) -> ModelInfoResponse:
        """Return frozen model governance metadata."""
        return ModelInfoResponse.model_validate(self._get("/model-info"))

    def portfolio_summary(self) -> PortfolioSummaryResponse:
        """Return final immutable portfolio evidence."""
        return PortfolioSummaryResponse.model_validate(self._get("/portfolio-summary"))

    def monitoring_summary(self) -> MonitoringSummaryResponse:
        """Return the latest drift and data-quality monitoring snapshot."""
        return MonitoringSummaryResponse.model_validate(self._get("/monitoring-summary"))

    def predict(self, applicant_id: int, reason_count: int = 5) -> PredictionResponse:
        """Return a governed prediction with local reason codes."""
        return PredictionResponse.model_validate(
            self._get(
                f"/predict/{applicant_id}",
                params={"reason_count": reason_count},
            )
        )

    def ask_evidence_assistant(self, question: str) -> EvidenceAssistantResponse:
        """Return a guarded, citation-backed governance evidence briefing."""
        return EvidenceAssistantResponse.model_validate(
            self._post("/evidence-assistant/query", {"question": question})
        )

    def simulate_new_application(
        self, application: NewApplicationRequest, reason_count: int = 5
    ) -> NewApplicationResponse:
        """Submit a manual application to the governed application-only simulator."""
        return NewApplicationResponse.model_validate(
            self._post(
                f"/simulate-new-application?reason_count={reason_count}",
                application.model_dump(),
            )
        )
