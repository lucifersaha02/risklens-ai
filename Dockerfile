# syntax=docker/dockerfile:1.7

FROM python:3.12.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

COPY pyproject.toml README.md ./
COPY src ./src

RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install ".[api]"


FROM python:3.12.10-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    RISKLENS_PROJECT_ROOT=/app \
    HOME=/tmp/risklens \
    NUMBA_CACHE_DIR=/tmp/risklens/numba

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 risklens \
    && useradd --system --uid 10001 --gid risklens --home-dir /tmp/risklens risklens

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=risklens:risklens configs ./configs

USER 10001:10001

EXPOSE 8000 8501

CMD ["risklens", "serve-api", "--host", "0.0.0.0", "--port", "8000"]
