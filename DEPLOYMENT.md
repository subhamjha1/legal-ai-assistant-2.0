# Deployment Guide

This document covers every way to run the Legal AI Assistant, from a local
dev environment to a fully deployed production stack, plus the operational
concerns (secrets, monitoring, scaling, backups, cost) that matter once
it's live.

**An upfront, honest note**: this project's development sandbox has no
Docker daemon/CLI and a network allow-list that excludes Docker Hub,
Elastic's registry, and HuggingFace (see `backend/README.md` for the
specifics). Every config file in this milestone — `Dockerfile`s,
`docker-compose.yml`, `render.yaml`, `railway.json`, `vercel.json`, the CI
workflow — is real and internally consistent with the application code, and
was validated as far as this sandbox allows (YAML/JSON parsing, `ruff`/
`eslint` passing, the Next.js standalone build actually producing a working
`server.js`, the rate-limiting middleware tested against a real server).
**Building and running the actual Docker images has not been verified in
this sandbox** and needs to be done once on a machine with Docker before
you rely on it — see the "Final Verification Checklist" at the end of this
document for exactly what to check.

---

## Table of Contents

1. [Local Development](#1-local-development)
2. [Docker Deployment](#2-docker-deployment)
3. [Render Deployment](#3-render-deployment)
4. [Railway Deployment](#4-railway-deployment)
5. [Vercel Deployment (Frontend Only)](#5-vercel-deployment-frontend-only)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Production Checklist](#7-production-checklist)
8. [SSL / HTTPS](#8-ssl--https)
9. [Secrets Management](#9-secrets-management)
10. [Logging](#10-logging)
11. [Monitoring & Observability](#11-monitoring--observability)
12. [Scaling Recommendations](#12-scaling-recommendations)
13. [Backup Strategy](#13-backup-strategy)
14. [Cost Estimates](#14-cost-estimates)
15. [Further Production Improvements](#15-further-production-improvements)
16. [CI/CD](#16-cicd)
17. [Final Verification Checklist](#17-final-verification-checklist)

---

## 1. Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

By default the backend runs fully standalone with no external services:
`QDRANT_MODE=local` (embedded Qdrant, file-backed) and
`KEYWORD_SEARCH_PROVIDER=bm25_local` (in-process BM25). This is genuinely
sufficient for development and even small production deployments — see
`backend/README.md`'s Milestones 3/4 for why both exist.

---

## 2. Docker Deployment

The full stack — backend, frontend, Qdrant, Elasticsearch — via
`docker-compose.yml` at the repo root. **No Postgres**: nothing in this
application uses a relational database (see the comment at the top of
`docker-compose.yml`); don't add one until a real feature needs it.

```bash
cp backend/.env.example backend/.env   # fill in ANTHROPIC_API_KEY etc.
docker compose up -d --build
curl http://localhost:8000/health
open http://localhost:3000
```

### What's in the compose file

| Service | Image | Persisted via volume | Healthcheck |
|---|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.11.3` | `qdrant_data` | TCP check on :6333 |
| `elasticsearch` | `docker.elastic.co/elasticsearch/elasticsearch:8.15.1` | `es_data` | `curl :9200/_cluster/health` |
| `backend` | built from `backend/Dockerfile` | `backend_uploads`, `backend_processed`, `model_cache` (HF weights) | `curl :8000/health` |
| `frontend` | built from `frontend/Dockerfile` | — (stateless) | `curl :3000/api/health` |

`backend` waits for `qdrant` and `elasticsearch` to report healthy
(`depends_on: condition: service_healthy`) before starting, and is
automatically configured to use them (`QDRANT_MODE=server`,
`KEYWORD_SEARCH_PROVIDER=elasticsearch`) rather than the embedded/local
fallbacks — those overrides live in `docker-compose.yml`'s `environment:`
block, layered on top of `backend/.env` via `env_file`.

### The `NEXT_PUBLIC_API_URL` build-arg gotcha

Next.js inlines `NEXT_PUBLIC_*` variables into the client JavaScript bundle
**at build time**, not read at container start. This means:
- For local `docker compose up --build`, the default
  (`http://localhost:8000`) is correct because the *browser* — not the
  frontend container — talks to the backend directly, and your browser
  really does see the backend on `localhost:8000`.
- For any real multi-host deployment, you must set
  `NEXT_PUBLIC_API_URL` to the backend's actual public URL **before
  building** the frontend image, and rebuild (not just restart) if that URL
  ever changes:
  ```bash
  NEXT_PUBLIC_API_URL=https://api.yourdomain.com docker compose up -d --build frontend
  ```

### Backend & frontend Dockerfiles

Both are multi-stage builds:
- **Backend** (`backend/Dockerfile`): `builder` stage compiles dependencies
  into a venv (needs `build-essential` for anything that compiles from
  source); `runtime` stage copies only the venv + app code, keeps
  `tesseract-ocr` (a genuine runtime dependency for Milestone 1's OCR path,
  unlike `build-essential`), runs as a non-root `appuser`, and serves via
  `gunicorn` + `uvicorn` workers (config in `backend/gunicorn_conf.py` —
  worker count, timeouts, logging, all overridable via env vars without
  touching the Dockerfile).
- **Frontend** (`frontend/Dockerfile`): `deps` → `builder` → `runner`
  stages, using Next.js's `output: "standalone"` mode (set in
  `next.config.ts`) so the final image copies a self-contained server
  bundle instead of the full `node_modules` tree. Runs as a non-root
  `nodeuser`.

---

## 3. Render Deployment

`render.yaml` (repo root) defines both services as a Render Blueprint.
Render has no managed Qdrant or Elasticsearch, so production use requires
pointing at managed cloud instances:

```bash
# In the Render dashboard, connect the repo -> "New Blueprint"
# Render reads render.yaml and creates both services.
```

After the blueprint runs, set these in the Render dashboard (marked
`sync: false` in `render.yaml` — real secrets are never committed):

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `QDRANT_HOST` | Qdrant Cloud cluster URL (qdrant.tech) |
| `ELASTICSEARCH_HOST` | Elastic Cloud deployment endpoint |

The backend service mounts a persistent disk at `/app/storage` (10GB,
configurable in `render.yaml`) for uploaded documents. The frontend's
`NEXT_PUBLIC_API_URL` build arg needs updating (and a manual redeploy) once
the backend's actual `.onrender.com` URL is known.

**Build command**: handled automatically via `runtime: docker` (Render
builds directly from each `Dockerfile`) — no separate build command needed.
**Start command**: comes from each Dockerfile's `CMD`.

---

## 4. Railway Deployment

Railway wants per-service configuration in a monorepo, so there are two
files: `backend/railway.json` and `frontend/railway.json`. Create two
Railway services from the same repo, setting each service's **root
directory** in the Railway dashboard:

```
Service 1: "legal-ai-backend"  → root directory: backend
Service 2: "legal-ai-frontend" → root directory: frontend
```

Railway reads the `railway.json` in whichever directory is configured as
that service's root. Required environment variables (set in the Railway
dashboard, per service):

**Backend service**: `ANTHROPIC_API_KEY`, `QDRANT_HOST`,
`ELASTICSEARCH_HOST` (or leave `QDRANT_MODE=local` /
`KEYWORD_SEARCH_PROVIDER=bm25_local` for a simpler single-container
deployment without external Qdrant/ES — see `backend/.env.example`).

**Frontend service**: `NEXT_PUBLIC_API_URL` set as a **build-time**
variable (Railway supports this via the dashboard's "Build" variable
scope, not just runtime) pointing at the backend service's Railway-provided
domain.

---

## 5. Vercel Deployment (Frontend Only)

Vercel is a serverless/edge platform — it's a great fit for the Next.js
frontend, but a poor fit for the FastAPI backend (a long-running process
with persistent connections to Qdrant/Elasticsearch doesn't map well onto
Vercel's serverless functions). **Deploy only the frontend to Vercel**; run
the backend on Render, Railway, or your own infrastructure.

```bash
cd frontend
vercel link
vercel env add NEXT_PUBLIC_API_URL production   # paste your backend's public URL
vercel --prod
```

`frontend/vercel.json` sets the framework preset, build/install commands,
and baseline security headers (`X-Content-Type-Options`, `X-Frame-Options`,
`Referrer-Policy`). The `NEXT_PUBLIC_API_URL` entry references a Vercel
env var/secret (`@legal_ai_backend_url`) rather than hardcoding a URL —
set the real value via `vercel env add` as shown above, or in the Vercel
dashboard under Project Settings → Environment Variables.

---

## 6. Environment Variables Reference

Full lists live in `backend/.env.example` and `frontend/.env.example` —
this is the "what matters most" summary.

### Backend — required for real functionality

| Variable | Purpose | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Real Claude calls (Milestone 7) | Required for real Q&A; app runs without it but `/query` will error |
| `EMBEDDING_PROVIDER` | `bge` (default, free, local) / `openai` / `hash` (offline fallback) | No — has a working default |
| `QDRANT_MODE` | `local` (embedded) / `server` (real Qdrant, needed for Docker/prod) | No — defaults to `local` |
| `KEYWORD_SEARCH_PROVIDER` | `bm25_local` (default) / `elasticsearch` | No — defaults to `bm25_local` |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Per-client-IP request budget | No — defaults to 60 |

### Frontend

| Variable | Purpose | Required? |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend URL, **baked in at build time** | Yes — defaults to `http://localhost:8000` if unset |

---

## 7. Production Checklist

- [ ] Real `ANTHROPIC_API_KEY` set (not the sandbox's empty default)
- [ ] `QDRANT_MODE=server` pointing at a real Qdrant instance (not `local`)
- [ ] `KEYWORD_SEARCH_PROVIDER=elasticsearch` pointing at a real cluster (not `bm25_local`) — or a deliberate, documented decision to stay with `bm25_local` for a smaller deployment
- [ ] `EMBEDDING_PROVIDER=bge` or `openai` (not `hash`, which is an offline/demo fallback only — see `backend/README.md`)
- [ ] CORS `allow_origins` in `backend/app/main.py` narrowed from `["*"]` to your actual frontend domain
- [ ] `RATE_LIMIT_REQUESTS_PER_MINUTE` tuned for expected real traffic
- [ ] HTTPS terminated in front of both services (see below)
- [ ] Persistent volumes/disks actually mounted (not ephemeral container storage) for `backend_uploads` / `backend_processed`
- [ ] Backend `.env` (or dashboard secrets) never committed to git
- [ ] Health check endpoints (`/health`, `/api/health`) wired into your platform's actual load balancer / uptime monitor, not just left as unused Dockerfile artifacts
- [ ] A real run of `python -m evaluation.cli` against your production-configured pipeline, with a real LLM key, before trusting answer quality (see `backend/README.md`'s Milestone 9 section on why the sandbox's own evaluation run doesn't reflect real answer quality)

---

## 8. SSL / HTTPS

None of the three deployment targets need you to configure TLS yourself:

- **Render / Railway**: HTTPS is automatic on the platform's provided
  domain (and on any custom domain you attach) — no config needed.
- **Vercel**: same — automatic HTTPS on `*.vercel.app` and custom domains.
- **Self-hosted Docker Compose** (e.g. on a bare VM): put a reverse proxy
  (Caddy or nginx) in front of both containers and let it handle
  certificates. Caddy is the simplest option — a two-line Caddyfile
  (`yourdomain.com { reverse_proxy backend:8000 }`) gets you automatic
  Let's Encrypt certificates with zero manual renewal.

---

## 9. Secrets Management

- **Never** commit a real `.env` file — both `backend/.gitignore` and
  `frontend/.gitignore` already exclude `.env*` except the `.example`
  templates.
- **Render / Railway**: set secrets via each platform's dashboard (or
  `railway variables set`) — they're encrypted at rest and injected as
  runtime env vars, never visible in build logs.
- **Vercel**: `vercel env add` (or the dashboard) — same guarantee.
- **Docker Compose**: `backend/.env` is read via `env_file:` — keep it out
  of version control and out of the Docker build context (`.dockerignore`
  already excludes `.env*`), so it's never baked into an image layer.
- Rotate `ANTHROPIC_API_KEY` (and any other provider key) immediately if a
  `.env` file is ever accidentally committed — treat it as compromised, not
  just "hopefully fine."

---

## 10. Logging

The backend already logs structured lines (`app/core/logging_config.py`) to
stdout/stderr with timestamps, levels, and module names — this is
deliberate, not an oversight: container log drivers (Docker, Render,
Railway) all capture stdout/stderr natively with zero extra configuration.
`gunicorn_conf.py` routes its own access/error logs the same way.

For anything beyond "read logs in the platform dashboard": ship stdout to a
log aggregator (Better Stack, Papertrail, or a self-hosted Loki instance)
by pointing the platform's log-forwarding integration at it — none of this
project's code needs to change, since it already logs to stdout in a
structured, parseable format.

---

## 11. Monitoring & Observability

Not implemented in this codebase (out of scope for this assignment), but
here's the concrete upgrade path:

- **Uptime / health**: point your platform's health check (Render/Railway
  already do this automatically via `healthCheckPath` in their configs) or
  an external service (UptimeRobot, Better Stack) at `/health` and
  `/api/health`.
- **Error tracking**: add Sentry — `pip install sentry-sdk` on the backend
  (a few lines in `app/main.py`'s startup), `@sentry/nextjs` on the
  frontend. Both have generous free tiers and are the standard choice for
  exactly this stack.
- **Metrics**: `prometheus-fastapi-instrumentator` (pip package) adds a
  `/metrics` endpoint with request latency/count histograms in ~3 lines of
  code, scrapeable by Prometheus or Grafana Cloud.
- **LLM-specific observability**: the `TokenUsage` already captured in
  every `AnswerResponse` (Milestone 9) is exactly the raw material a
  cost/usage dashboard needs — it just isn't piped anywhere yet. Logging it
  structured (`logger.info("llm_usage", extra={...})`) and shipping to
  whatever log aggregator you choose gets you cost tracking without new
  infrastructure.

---

## 12. Scaling Recommendations

- **Horizontal scaling (backend)**: the FastAPI app itself is stateless
  and safe to run as multiple replicas — `gunicorn_conf.py`'s worker count
  handles per-instance concurrency, and `numReplicas`/Render's instance
  count handles horizontal scale. **Caveat**: `RateLimitMiddleware`
  (Milestone 10) is in-memory per-process — with N replicas, the effective
  rate limit becomes N× the configured value, since each replica tracks
  its own counters. Fix: swap the in-memory `dict`/`deque` in
  `app/core/rate_limit.py` for Redis-backed counters (a `redis-py`
  `INCR` + `EXPIRE` pair per client per window) once you actually run more
  than one replica — the middleware's interface doesn't need to change,
  only its storage backend.
- **Reverse proxy**: put nginx or Caddy (or your platform's built-in load
  balancer) in front of multiple backend replicas for round-robin/least-
  connections load balancing — Render and Railway do this automatically
  when you scale instance count.
- **Redis caching**: beyond rate-limit counters, Redis is also the natural
  place to cache embedding results for frequently-repeated queries (a
  legal research tool plausibly gets the same or similar questions asked
  repeatedly) — a simple `hash(query) -> cached AnswerResponse` cache with
  a TTL would cut both latency and LLM cost on repeat queries.
- **Production Qdrant**: the embedded/local mode (`QDRANT_MODE=local`) is
  single-process and file-locked — fine for one backend instance, wrong
  for horizontal scaling. Switch to `QDRANT_MODE=server` pointing at a real
  Qdrant instance (self-hosted or Qdrant Cloud) the moment you run more
  than one backend replica, since multiple processes can't share one
  embedded instance's file lock.
- **Production Elasticsearch**: similarly, `KEYWORD_SEARCH_PROVIDER=bm25_local`'s
  corpus file has the same single-process assumption baked into
  `BM25LocalProvider` (see `backend/app/services/keyword_search.py`) —
  switch to `elasticsearch` before scaling horizontally.

---

## 13. Backup Strategy

- **Qdrant**: use Qdrant's built-in snapshot API
  (`POST /collections/{name}/snapshots`) on a schedule, stored to
  S3-compatible object storage. Qdrant Cloud handles this automatically.
- **Elasticsearch**: use the snapshot/restore module (`_snapshot` API)
  against an S3 or GCS repository — Elastic Cloud automates this;
  self-hosted needs a cron job calling the API.
- **Backend uploaded documents** (`backend_uploads`/`backend_processed`
  volumes): these are the actual source-of-truth PDFs and their parsed
  JSON — back up the volume itself (`docker run --rm -v backend_uploads:/data
  -v $(pwd):/backup busybox tar czf /backup/uploads-backup.tar.gz /data`),
  or better, treat cloud object storage (S3/GCS) as the real source of
  truth for uploads rather than a local volume at all once you're serious
  about production durability — that's a small code change to
  `app/services/storage.py`'s `DocumentRepository`, not a config one.
- **Golden dataset & evaluation results** (`evaluation/golden_dataset.json`,
  `evaluation/results/`): these are just files in the repo — version
  control already backs them up.

---

## 14. Cost Estimates

Rough monthly figures, US pricing, as of this writing — always check
current pricing before committing:

| Component | Minimal (dev/demo) | Production (moderate traffic) |
|---|---|---|
| Backend hosting (Render/Railway) | $7-25/mo (starter/hobby tier) | $50-150/mo (standard tier, 2GB+ RAM for BGE embeddings) |
| Frontend hosting (Vercel) | $0 (hobby tier) | $20/mo (Pro tier, for team features/analytics) |
| Qdrant | $0 (embedded/local mode) | $25-100+/mo (Qdrant Cloud, scales with vector count) |
| Elasticsearch | $0 (BM25 local mode) | $95+/mo (Elastic Cloud smallest production tier) |
| Anthropic API (Claude) | Pay-per-token, ~$3/million input tokens, ~$15/million output (Sonnet-class pricing) | Scales directly with query volume — the `TokenUsage` tracking in Milestone 9 is exactly what you'd use to project this from real traffic |
| **Total** | **~$7-25/mo** (embedded Qdrant/BM25, no managed search infra) | **~$200-400+/mo** (fully managed, horizontally scaled) + LLM usage |

The "minimal" column is a real, valid production configuration for a
small-scale deployment — `QDRANT_MODE=local` and
`KEYWORD_SEARCH_PROVIDER=bm25_local` are not toys, they're genuine
single-instance production options (see `backend/README.md`'s Milestone 3/4
sections) that simply don't scale horizontally. Move to the managed
services only when you actually need to.

---

## 15. Further Production Improvements

Beyond what's implemented (rate limiting — see `app/core/rate_limit.py`,
tested in `tests/test_rate_limit.py`):

- **Redis caching**: see Scaling Recommendations above — both for
  distributed rate-limit counters and for caching repeated-query answers.
- **API versioning**: routes are already under `/api/v1/` — when
  introducing breaking changes, add `/api/v2/` alongside rather than
  mutating `v1`'s contract.
- **Request validation hardening**: Pydantic already validates every
  request body; consider adding max-file-size enforcement at the reverse
  proxy layer too (defense in depth beyond `MAX_UPLOAD_SIZE_MB`).
- **Async LLM calls**: `LLMProvider.generate()` is currently synchronous
  (blocking the gunicorn worker for the duration of the LLM call) — for
  higher concurrency per replica, migrate to the Anthropic SDK's async
  client and an async route handler.

---

## 16. CI/CD

`.github/workflows/ci.yml` runs on every push/PR with five jobs:

1. **`backend-tests`** — the full pytest suite (128 passing tests as of
   this writing; see `backend/README.md` for the complete breakdown by
   milestone), with Tesseract installed for the OCR test path.
2. **`frontend-build`** — `npm ci && npm run build`, verifying the Next.js
   production build succeeds (this is exactly what was manually run in
   this sandbox to verify the frontend, so CI genuinely repeats a check
   that already passed once here).
3. **`lint`** — `ruff` (backend) and `eslint` (frontend), both **confirmed
   clean** in this sandbox before shipping (a few real `ruff` findings —
   unused imports — were caught and fixed during this milestone, not
   glossed over).
4. **`docker-build`** — builds both Docker images and smoke-tests that
   they actually start and pass their health checks. **This job could not
   be run in this development sandbox** (no Docker daemon here) — it's
   real, correctly-configured CI that will genuinely validate on GitHub's
   hosted runners (which have Docker and full internet access), but treat
   its first real run as the actual verification of both Dockerfiles.
5. **`deployment-readiness`** — validates that `docker-compose.yml`,
   `render.yaml`, both `railway.json` files, and `vercel.json` all parse
   correctly and reference a consistent set of services, and that every
   required env var is documented in the `.env.example` files. This is the
   same validation performed manually while building this milestone,
   codified so it can't silently regress.

GitHub's hosted runners have full, unrestricted internet access — unlike
this project's development sandbox, so unlike the deliberately-avoided
HuggingFace/OpenAI/Docker-registry calls in local testing, CI could
exercise those real network paths if a test ever needed to. It doesn't by
default (keeping CI fast and not dependent on a 1.3GB model download every
run), consistent with how every test in this project's suite already
injects fakes/offline providers rather than depending on network access.

---

## 17. Final Verification Checklist

Everything below marked "verified here" was actually run in this
development sandbox. Everything marked "needs verification" requires
infrastructure (Docker, a real API key, a real cloud account) this sandbox
doesn't have — do these once before trusting the deployment:

| Item | Status |
|---|---|
| `docker-compose.yml` parses as valid YAML, correct services/volumes/networks | Verified here |
| `render.yaml` parses as valid YAML, correct service names | Verified here |
| Both `railway.json` files parse as valid JSON with required sections | Verified here |
| `vercel.json` parses as valid JSON | Verified here |
| `.github/workflows/ci.yml` parses as valid YAML | Verified here |
| Backend `ruff check` passes cleanly | Verified here (3 real issues found & fixed) |
| Frontend `eslint`/`npm run build` pass cleanly | Verified here |
| Frontend Next.js `output: standalone` build produces a working `server.js` | Verified here |
| Rate limiting middleware behaves correctly (limit, exemption, per-client isolation, disable flag) | Verified here (5 tests, real FastAPI TestClient) |
| Full backend test suite (128 tests, retrieval/QA/evaluation pipeline) | Verified here |
| `docker build` actually succeeds for both `Dockerfile`s | Needs verification (requires Docker) |
| `docker compose up` brings up all 4 services healthy | Needs verification (requires Docker) |
| Render/Railway/Vercel deployments actually succeed | Needs verification (requires real platform accounts) |
| Real Claude answers via a real `ANTHROPIC_API_KEY` | Needs verification (requires a real key — network path is open, see `backend/README.md`) |
