"""Authenticated FastAPI surface for frozen RiskLens AI inference."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from risklens.serving.inference import ApplicantNotFoundError, FrozenRiskScorer
from risklens.serving.schemas import ModelInfoResponse, PredictionResponse

LOGGER = logging.getLogger("risklens.api")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class HealthResponse(BaseModel):
    """Deployment readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: str
    model_ready: bool
    model_version: str


class ErrorResponse(BaseModel):
    """Sanitized API error contract."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
    request_id: str


@lru_cache(maxsize=1)
def get_scorer() -> FrozenRiskScorer:
    """Load and verify the frozen model once per API process."""
    return FrozenRiskScorer()


def require_api_key(
    supplied_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Require a configured API key for model metadata and predictions."""
    expected_key = os.getenv("RISKLENS_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if supplied_key is None or not secrets.compare_digest(supplied_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _request_id(request: Request) -> str:
    """Return the middleware-validated request identifier."""
    return str(getattr(request.state, "request_id", "unknown"))


def _error_response(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    """Build one consistent error response with request correlation."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error,
            detail=detail,
            request_id=_request_id(request),
        ).model_dump(),
        headers={"X-Request-ID": _request_id(request)},
    )


app = FastAPI(
    title="RiskLens AI API",
    version="0.1.0",
    description=(
        "Hash-verified credit-risk decision support. Human review is required; "
        "responses are not autonomous credit decisions or adverse-action notices."
    ),
)


@app.middleware("http")
async def request_context(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
) -> Any:
    """Attach request IDs and emit structured completion logs."""
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    LOGGER.info(
        json.dumps(
            {
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    )
    return response


@app.exception_handler(ApplicantNotFoundError)
async def applicant_not_found(request: Request, error: ApplicantNotFoundError) -> JSONResponse:
    """Return a stable 404 without a Python traceback."""
    return _error_response(request, 404, "applicant_not_found", str(error))


@app.exception_handler(ValueError)
async def invalid_request(request: Request, error: ValueError) -> JSONResponse:
    """Return domain validation failures as unprocessable input."""
    return _error_response(request, 422, "invalid_request", str(error))


@app.exception_handler(FileNotFoundError)
@app.exception_handler(RuntimeError)
async def unavailable_model(request: Request, error: Exception) -> JSONResponse:
    """Return sanitized frozen-artifact failures as service unavailable."""
    LOGGER.error(
        json.dumps(
            {
                "event": "model_unavailable",
                "request_id": _request_id(request),
                "error_type": type(error).__name__,
            }
        )
    )
    return _error_response(
        request,
        503,
        "model_unavailable",
        "Frozen model artifacts are unavailable or failed verification",
    )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(scorer: Annotated[FrozenRiskScorer, Depends(get_scorer)]) -> HealthResponse:
    """Verify that the frozen model is loaded and ready."""
    info = scorer.model_info()
    return HealthResponse(status="ok", model_ready=True, model_version=info.model_version)


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    dependencies=[Depends(require_api_key)],
    tags=["governance"],
)
def model_info(
    scorer: Annotated[FrozenRiskScorer, Depends(get_scorer)],
) -> ModelInfoResponse:
    """Return frozen governance metadata."""
    return scorer.model_info()


@app.get(
    "/predict/{applicant_id}",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    dependencies=[Depends(require_api_key)],
    tags=["inference"],
)
def predict(
    applicant_id: int,
    scorer: Annotated[FrozenRiskScorer, Depends(get_scorer)],
    reason_count: Annotated[int, Query(ge=1, le=20)] = 5,
) -> PredictionResponse:
    """Score one known applicant using the immutable governed model."""
    if applicant_id <= 0:
        raise ValueError("Applicant ID must be positive")
    return scorer.score_applicant(applicant_id, reason_count=reason_count)
