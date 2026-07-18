"""
Application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chunks, hybrid_search, keyword_search, qa, retrieval, search, upload
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.rate_limit import RateLimitMiddleware

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered legal document search & Q&A system with citation-grounded answers.",
)

# Permissive CORS for local dev against the Next.js frontend (Milestone 8).
# Tighten `allow_origins` before deploying (Milestone 10).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting - see app/core/rate_limit.py docstring for what this does
# and does not cover (single-process only; needs Redis for multi-replica).
app.add_middleware(RateLimitMiddleware)

app.include_router(upload.router, prefix="/api/v1")
app.include_router(chunks.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(keyword_search.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(retrieval.router, prefix="/api/v1")
app.include_router(qa.router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
