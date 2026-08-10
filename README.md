# Legal & Tax RAG System

A production-structured Retrieval-Augmented Generation system for US legal
and tax documents (Acts, court judgments, legal commentary, IRS
regulations), built around **hybrid retrieval** (FAISS + BM25 +
Elasticsearch fused with Reciprocal Rank Fusion), a **cross-encoder
reranker**, **GraphRAG** over Neo4j, and **citation-grounded answer
generation** that always names the document, page, and section it's
drawing from.

See [`docs/architecture.md`](docs/architecture.md) for Mermaid diagrams of
the ingestion pipeline, the retrieval pipeline, and the deployment
topology.

## What's real vs. what needs infrastructure you provide

Everything in this repo is real, working code — but "production-ready"
here means *correctly engineered*, not *pre-verified against live
Elasticsearch/Neo4j clusters*, which this sandbox doesn't have. Concretely:

| Component | Status |
|---|---|
| PDF parsing (page/section/heading preservation) | Verified end-to-end against a synthetic legal PDF in this session |
| Intelligent chunking (heading-aware + recursive fallback + parent/child) | Verified end-to-end; a real chunk_id collision bug between parent/child chunks was found and fixed during testing |
| Citation extraction (case law, U.S.C., C.F.R., IRC) | Unit-tested, passing |
| Reciprocal Rank Fusion | Unit-tested, passing |
| Retrieval evaluation metrics (Recall@K, Precision@K, MRR, NDCG) | Unit-tested, passing |
| FAISS / BM25 / embeddings / cross-encoder | Correct code against documented APIs; not executed here — model downloads (HuggingFace) aren't reachable from this sandbox's network allowlist |
| Elasticsearch / Neo4j / PostgreSQL | Correct client code against documented APIs; no live cluster available in this sandbox — every client degrades gracefully (logs a warning, returns empty results) if its backing service is unreachable, so the system still runs with FAISS+BM25-only retrieval if ES/Neo4j are down |
| RAGAS / DeepEval generation metrics | Correct integration code; requires a real LLM API key to execute (LLM-as-judge) |

**Before you rely on this in production:** run `docker compose up`, point
`GEMINI_API_KEY` or `LLM_API_KEY` at a real provider, ingest a handful of real PDFs, and check
retrieval quality against your own judgment before trusting the golden-set
numbers.

## Quickstart

```bash
cp .env.example .env
# edit .env: set GEMINI_API_KEY at minimum

cd docker
docker compose up --build
```

- Backend API: http://localhost:8000/docs (Swagger UI)
- Frontend: http://localhost:8501
- Neo4j browser: http://localhost:7474

Then ingest your corpus:

```bash
# put ~100 PDFs under data/raw/{acts,judgments,commentaries,irs}/
docker compose exec backend python scripts/ingest_corpus.py
```

Or via the API:

```bash
curl -X POST http://localhost:8000/api/v1/index
```

### Local dev (no Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# start postgres/elasticsearch/neo4j yourself, or accept degraded mode
uvicorn backend.api.main:app --host 0.0.0.0 --port 8004 --reload
```

In a second terminal:

```bash
streamlit run frontend/app.py
```

For local dev, the backend defaults to http://localhost:8004. If you keep
that port, start Streamlit with:

```bash
BACKEND_URL=http://localhost:8004 streamlit run frontend/app.py
```

Gemini is configured through the OpenAI-compatible endpoint:

```env
LLM_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_API_KEY=your-real-key
LLM_MODEL_NAME=gemini-3.5-flash
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/upload` | Upload + ingest + index a single PDF |
| `POST /api/v1/index` | Batch ingest everything under `data/raw/` |
| `POST /api/v1/query` | Full RAG: retrieve, rerank, generate a cited answer |
| `POST /api/v1/query/stream` | Same, with a streamed answer |
| `POST /api/v1/retrieve` | Retrieval only (no LLM) — inspect what would be retrieved |
| `POST /api/v1/summarize` | Map-reduce summary of a full document |
| `GET /api/v1/documents` | List indexed documents |
| `POST /api/v1/evaluate` | Run the golden-dataset evaluation, generate report + charts |
| `GET /api/v1/health` | Liveness + index sizes + which backends are enabled |
| `POST /api/v1/feedback` | Record thumbs up/down feedback |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What qualifies as a capital asset under the Internal Revenue Code?"}'
```

## Evaluation

Fill in `data/golden/golden_dataset.csv` (see
`data/golden/golden_dataset.example.csv` for the required columns:
`query, ground_truth, document, page`), then:

```bash
python scripts/run_evaluation.py
```

This computes Recall@5, Recall@10, Precision@K, MRR, NDCG (retrieval),
Faithfulness / Answer Relevancy / Context Precision / Context Recall via
RAGAS (generation), a document-type confusion matrix, and writes a
markdown report with charts to `docs/eval_reports/`.

## Key design decisions

- **RRF over weighted-average fusion**: FAISS cosine similarity, BM25
  scores, and Elasticsearch's BM25-variant scores live on incomparable
  scales. RRF fuses by rank position, not raw score, which is the
  standard fix (`backend/retrieval/rrf.py`).
- **Repository pattern for Postgres**: retrieval/API code never writes raw
  SQL — everything goes through `ChunkRepository` / `DocumentRepository`
  (`backend/database/repository.py`), so the persistence layer can be
  swapped or mocked in tests.
- **Every store degrades gracefully**: Elasticsearch and Neo4j clients
  check connectivity at startup and silently no-op (with a logged warning)
  if unreachable, rather than crashing the whole retrieval pipeline. FAISS
  + BM25 + Postgres are the required core; ES + Neo4j are enhancements.
- **Non-LLM retrieval confidence score**: the API returns
  `retrieval_confidence`, computed from the reranker's score distribution
  rather than asking the LLM to self-report confidence, which tends to be
  poorly calibrated.
- **Citation-critical fields are non-nullable** on `Chunk`
  (`backend/schemas/document.py`): `page_start`, `page_end`, `filename`
  are required and validated, because losing them silently breaks every
  citation downstream.

## Project structure

```
backend/
  api/            FastAPI app, routers, request/response schemas, DI
  config/         Pydantic settings (single source of truth for config)
  database/       SQLAlchemy models + repository pattern
  ingestion/      PDF parsing, classification, chunking, citation extraction, indexing
  retrieval/      Embeddings, FAISS, BM25, Elasticsearch, RRF, query expansion, hybrid orchestrator
  reranker/       Cross-encoder reranking
  graph/          Neo4j client, relation extraction, graph-expanded retrieval
  llm/            OpenAI-compatible client, answer generation, map-reduce summarization
  schemas/        The Open Knowledge Format (Chunk, DocumentMetadata, Citation, GraphRelation)
  evaluation/     Golden dataset loader, metrics, RAGAS/DeepEval, report generator, runner
frontend/         Streamlit UI
tests/            Pytest unit tests
scripts/          CLI: batch ingest, run evaluation
docker/           Dockerfiles + docker-compose.yml
docs/             Architecture diagrams, eval reports
data/             raw PDFs, processed indices, golden dataset
```

## Known limitations / next steps

- BM25 (`rank_bm25`) rebuilds the full corpus index on every ingest batch
  rather than updating incrementally — fine at ~100 documents, would need
  swapping to Elasticsearch-only lexical search (already implemented, just
  make it the sole lexical signal) at much larger scale.
- The LLM-based relation extraction pass for GraphRAG (AMENDS/OVERRULES/
  IMPLEMENTS beyond the deterministic keyword heuristics) is stubbed as a
  documented extension point in `relation_extractor.py`, not fully wired
  in, to bound LLM cost on a 100-document corpus by default.
- Document classification falls back to keyword regex when filenames
  aren't organized into type folders; for a production corpus, prefer
  organizing `data/raw/{acts,judgments,commentaries,irs}/` or enable
  `classify_with_llm` for higher accuracy.
