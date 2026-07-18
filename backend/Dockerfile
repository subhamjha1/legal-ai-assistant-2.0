# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder - installs Python dependencies into a venv, including
# system packages needed only at build/install time (e.g. compilers for any
# package that needs to build from source). Kept separate from the runtime
# stage so none of this ends up in the final image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime - only what's needed to RUN the app. No build tools, no
# pip cache, no test files. This is what actually ships.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Tesseract + its dependencies are a genuine runtime requirement (OCR path
# in app/services/parser.py), not just a build-time one - so unlike
# build-essential, this stays in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user - a container running as root is a real, avoidable privilege
# escalation risk if the app is ever compromised.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser evaluation ./evaluation
COPY --chown=appuser:appuser gunicorn_conf.py ./gunicorn_conf.py

# Storage directories the app writes to at runtime - created here with
# correct ownership so the non-root user can actually write to them once
# volumes are mounted over them in docker-compose.
RUN mkdir -p storage/uploads storage/processed storage/qdrant_local storage/bm25_local \
    && chown -R appuser:appuser storage

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Gunicorn manages a pool of uvicorn worker processes - the production-grade
# way to run FastAPI (a single uvicorn process is fine for dev, but doesn't
# use multiple CPU cores and has no worker-restart supervision). Config
# (worker count, timeouts, logging) lives in gunicorn_conf.py so it's
# version-controlled and consistent across environments rather than being a
# long inline CMD.
CMD ["gunicorn", "app.main:app", "--config", "gunicorn_conf.py"]
