"""
Rate limiting middleware (Milestone 10 production hardening).

Why this exists:
A single-process, in-memory sliding-window limiter, keyed by client IP.
This is a genuinely useful default for a single-instance deployment (the
common case for this project's target scale) and requires no additional
infrastructure - but it is NOT correct for a horizontally-scaled deployment
with multiple backend replicas, since each process would track its own
counters independently, effectively multiplying the real limit by the
replica count. See DEPLOYMENT.md's "Redis caching" / horizontal scaling
section: a Redis-backed limiter (shared counters across replicas) is the
documented upgrade path once you actually run more than one replica.

Kept as real, tested, dependency-free middleware rather than only a
documentation bullet point, because "rate limiting" as a bullet in a
production checklist with no working implementation behind it is exactly
the kind of unverified claim this project has avoided throughout.
"""
import time
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        # client_id -> deque of request timestamps within the current window.
        self._requests: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.settings.rate_limit_enabled or request.url.path in self.settings.rate_limit_exempt_paths:
            return await call_next(request)

        client_id = self._client_id(request)
        now = time.monotonic()
        window_start = now - 60.0

        timestamps = self._requests[client_id]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self.settings.rate_limit_requests_per_minute:
            retry_after = max(0.0, 60.0 - (now - timestamps[0]))
            logger.warning("Rate limit exceeded for %s on %s", client_id, request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        timestamps.append(now)
        return await call_next(request)

    @staticmethod
    def _client_id(request: Request) -> str:
        # Respect a trusted reverse proxy's forwarded header (Render/Railway/
        # nginx all set this) before falling back to the raw connection IP,
        # since in production the backend sits behind a proxy and
        # request.client.host would otherwise be the proxy's own IP for
        # every request.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
