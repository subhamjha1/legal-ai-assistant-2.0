"""
Gunicorn production configuration.

Kept as a real config file (not CLI flags baked into the Dockerfile CMD) so
worker count, timeouts, and logging are all in one version-controlled place
and can be tuned per-environment via env vars without editing the
Dockerfile.
"""
import multiprocessing
import os

# Bind
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes - default to (2 * CPU cores) + 1, the standard gunicorn
# recommendation, but overridable via env for memory-constrained hosts
# (e.g. small Render/Railway instances where that formula over-provisions).
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts. Generous request timeout because LLM generation (Milestone 7)
# and OCR-heavy PDF parsing (Milestone 1) can legitimately take longer than
# gunicorn's 30s default without anything being wrong.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Restart workers periodically to bound the impact of any slow memory leak
# (e.g. from the BGE model or Qdrant client) rather than needing a full
# redeploy to recover - a small amount of jitter avoids all workers
# recycling at exactly the same moment.
max_requests = 1000
max_requests_jitter = 100

# Logging - stdout/stderr so container log drivers (Docker, Render,
# Railway) capture everything without extra configuration.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# Fail fast on boot errors instead of silently running zero workers.
preload_app = True
