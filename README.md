# Legal AI Assistant

An AI-powered legal search & Q&A system for Acts, Court Judgments, Tax
Documents, and POV Documents — every answer traced to its exact source
document and page, built to minimize hallucination at every layer of the
pipeline: parsing → OCR → chunking → embeddings → hybrid search (vector +
BM25) → MMR → re-ranking → LLM → citation formatting → frontend.

See `Architecture_Diagram.pdf` for the full visual pipeline,
`Approach_Document.pdf` for design rationale, and `Evaluation_Report.pdf`
for measured results against the Golden Set.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Backend Setup](#2-backend-setup)
3. [Frontend Setup](#3-frontend-setup)
4. [Environment Variables](#4-environment-variables)
5. [Running Locally](#5-running-locally)
6. [Running Tests](#6-running-tests)
7. [Deployment](#7-deployment)
8. [API Documentation](#8-api-documentation)
9. [Folder Structure](#9-folder-structure)
10. [Evaluation](#10-evaluation)
11. [Honest Limitations](#11-honest-limitations)

---

## 1. Installation

Prerequisites: Python 3.12+, Node.js 20+, Tesseract OCR.

```bash
# Tesseract (required for the OCR fallback path)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract

git clone <this-repo>
cd legal-ai-assistant
```

## 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env: at minimum, set ANTHROPIC_API_KEY for real Q&A generation.
```

The backend runs fully standalone with zero external services by default
(`QDRANT_MODE=local` — embedded Qdrant; `KEYWORD_SEARCH_PROVIDER=bm25_local`
— in-process BM25; `EMBEDDING_PROVIDER=bge` — local model, downloaded once
on first use). No Qdrant server, Elasticsearch cluster, or Docker is
required to run it locally.

## 3. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local
# NEXT_PUBLIC_API_URL defaults to http://localhost:8000 - change if your
# backend runs elsewhere. This is a BUILD-TIME variable for production
# builds (see DEPLOYMENT.md) but read live in `next dev`.
```

## 4. Environment Variables

Only `.env.example` files are included in this submission — no real
secrets, keys, or `.env` files are present anywhere in this repository.

**Backend** (`backend/.env.example`) — key variables:

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Real Claude calls for answer generation | empty — required for real `/query` responses |
| `LLM_PROVIDER` | `anthropic` / `openai` / `gemini` | `anthropic` |
| `EMBEDDING_PROVIDER` | `bge` / `openai` / `hash` (offline fallback) | `bge` |
| `QDRANT_MODE` | `local` (embedded) / `server` (real Qdrant) | `local` |
| `KEYWORD_SEARCH_PROVIDER` | `bm25_local` / `elasticsearch` | `bm25_local` |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Per-client-IP request budget | `60` |

Full list with every setting documented: `backend/.env.example`.

**Frontend** (`frontend/.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend URL, baked in at build time for production | `http://localhost:8000` |

## 5. Running Locally

```bash
# Terminal 1 - backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 - frontend
cd frontend
npm run dev
```

Open `http://localhost:3000`, upload a PDF (a sample is included at
`backend/sample_docs/sample_legal_doc_final.pdf`), wait for the processing
pipeline to complete, and ask a question.

## 6. Running Tests

```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

**138 tests, 137 passing, 1 auto-skipped** (the skipped test requires a
real `ANTHROPIC_API_KEY`, which is not present in this environment — see
[Honest Limitations](#11-honest-limitations)). Full breakdown by milestone
is in `backend/README.md`.

```bash
cd frontend
npm run lint    # ESLint - passes cleanly
npm run build   # Next.js production build - succeeds
```

## 7. Deployment

Full details, environment variables, and an honest verification checklist
are in **`DEPLOYMENT.md`**. Summary of what's included:

- **Docker**: `docker-compose.yml` (backend, frontend, Qdrant, Elasticsearch
  — no Postgres, since nothing in the app uses one), multi-stage
  production `Dockerfile`s for both services.
- **Render**: `render.yaml` (Blueprint for both services).
- **Railway**: `backend/railway.json`, `frontend/railway.json` (per-service
  config, matching how Railway wants monorepos configured).
- **Vercel**: `frontend/vercel.json` (frontend only — Vercel's serverless
  model is a poor fit for the long-running FastAPI backend; see
  `DEPLOYMENT.md` for why).
- **CI/CD**: `.github/workflows/ci.yml` — backend tests, frontend build,
  lint, Docker build validation, deployment-readiness config checks.

**Honestly**: this project's development sandbox has no Docker daemon and
a network policy that blocks Docker Hub, Elastic's registry, and
HuggingFace. Every config file was validated as far as this sandbox
allows (YAML/JSON parsing, lint passing, the Next.js standalone build
actually producing a working server) but **`docker build` /
`docker compose up` have not been run** — see `DEPLOYMENT.md`'s Final
Verification Checklist for the exact list of what's confirmed vs. what
needs real infrastructure to verify.

## 8. API Documentation

Interactive OpenAPI docs are auto-generated by FastAPI at
`http://localhost:8000/docs` (Swagger UI) and `/redoc` once the backend is
running. Key endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health check |
| `/api/v1/documents/upload` | POST | Upload a PDF for ingestion |
| `/api/v1/documents` | GET | List ingested documents |
| `/api/v1/documents/{id}` | GET / DELETE | Fetch or remove a document |
| `/api/v1/documents/{id}/chunks` | POST / GET | Generate or fetch chunking |
| `/api/v1/documents/{id}/index` | POST | Embed + index into vector store |
| `/api/v1/documents/{id}/keyword-index` | POST | Index into BM25/Elasticsearch |
| `/api/v1/search` | POST | Vector-only search |
| `/api/v1/keyword-search` | POST | BM25-only search |
| `/api/v1/hybrid-search` | POST | RRF-fused vector + keyword search |
| `/api/v1/retrieve` | POST | Full pipeline: hybrid search → MMR → re-rank → confidence filter |
| `/api/v1/query` | POST | Full grounded Q&A: retrieval → LLM → citations |
| `/api/v1/query/stream` | POST | Same as `/query`, streamed via Server-Sent Events |

Full request/response schemas for every endpoint: `backend/README.md` and
the live `/docs` page.

## 9. Folder Structure

```
legal-ai-assistant/
├── backend/
│   ├── app/
│   │   ├── core/           # config, logging, rate limiting
│   │   ├── schemas/        # Pydantic contracts per pipeline stage
│   │   ├── services/       # parser, chunker, embeddings, vector store,
│   │   │                   # keyword search, hybrid search, MMR, reranker,
│   │   │                   # retriever, LLM providers, prompts, citations
│   │   ├── api/routes/     # thin HTTP layer over the services
│   │   └── main.py         # FastAPI app entrypoint
│   ├── evaluation/         # golden dataset, metrics, runner, reports, CLI
│   ├── tests/               # 138 tests across every milestone
│   ├── sample_docs/         # the real sample PDF used throughout testing
│   ├── Dockerfile, railway.json, gunicorn_conf.py, requirements.txt
│   └── README.md            # detailed per-milestone backend documentation
├── frontend/
│   ├── app/                 # Next.js App Router pages + API health route
│   ├── components/          # ui/ (shadcn-style primitives) + workbench/
│   ├── lib/                 # API client, types, utils
│   ├── Dockerfile, railway.json, vercel.json, package.json
│   └── README.md            # detailed frontend documentation
├── docs/screenshots/         # real, organized demo screenshots
├── .github/workflows/ci.yml  # CI: tests, build, lint, Docker, deployment readiness
├── docker-compose.yml        # full stack: backend, frontend, Qdrant, Elasticsearch
├── render.yaml
├── DEPLOYMENT.md
├── README.md                 # this file
├── Architecture_Diagram.pdf
├── Approach_Document.pdf
├── Evaluation_Report.pdf
├── Golden_Set.csv
└── FINAL_SUBMISSION_CHECKLIST.md
```

## 10. Evaluation

```bash
cd backend
python -m evaluation.cli --dataset evaluation/golden_dataset.json --output-dir evaluation/results
```

18 questions, hand-authored from the real sample document's actual
content, covering fact lookups, statutory requirements, holdings,
multi-page synthesis, and deliberate no-evidence checks. See
`Golden_Set.csv` for the dataset and `Evaluation_Report.pdf` for full
results — including an explicit statement of what was actually measured
and why the answer-quality numbers from this sandbox's run should not be
read as a verdict on real answer quality (no real LLM key was available
here; retrieval-side metrics are fully genuine).

## 11. Honest Limitations

This project's development sandbox has:
- **No Docker daemon or CLI**, and a network policy blocking Docker Hub
  and Elastic's container registry — Docker builds and Elasticsearch could
  not be run end-to-end here.
- **No network path to huggingface.co or api.openai.com** — the BGE
  embedding model, BGE cross-encoder reranker, and OpenAI provider could
  not be exercised with real network calls here.
- **No configured `ANTHROPIC_API_KEY`** — real Claude-generated answers
  could not be verified here, though the network path to Anthropic *is*
  open (a credentials gap, not a network gap).
- **No Gemini API key** — the Gemini provider is complete code, untested
  here for the same reason as OpenAI (network + no key).

Every one of these gaps is documented at the exact point in the code and
READMEs where it matters, with a real offline/local alternative built and
tested wherever possible (in-process BM25, a dependency-free reranker, a
hashing-trick embedding fallback) rather than mocked away. See
`backend/README.md`'s per-milestone sections, `DEPLOYMENT.md`'s Final
Verification Checklist, and `Approach_Document.pdf`'s Limitations section
for the complete, itemized account.
