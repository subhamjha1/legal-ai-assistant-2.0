# Legal AI Assistant — Backend

## Milestone 1: Data Ingestion & Pre-processing (complete)

### What this module does

Ingests a PDF (Act, Judgment, Tax Document, or POV Document), extracts every
page's text while preserving exact page numbers, handles scanned pages via
OCR, classifies document type automatically when the user doesn't tag it,
and persists the result as structured, citation-ready JSON.

This is the foundation every later milestone depends on: if page numbers or
document identity are wrong here, no amount of good prompting later fixes a
wrong citation.

### Architecture

```
Upload (PDF)
    │
    ▼
Validation (extension, size)
    │
    ▼
Per-page extraction loop:
    ├─ Native text (PyMuPDF)          → used when text is present & no tables
    ├─ Table-aware (pdfplumber)       → used when a table is detected on the page
    └─ OCR (Tesseract, 300 DPI)       → used when native text is near-empty
    │
    ▼
Document type resolution:
    ├─ User-tagged (trusted as-is), OR
    └─ Auto-suggested:
          keyword heuristic (regex signal scoring)
          → if confidence < threshold → LLM classification (optional, graceful fallback)
    │
    ▼
Pydantic validation (ParsedDocument, page order enforced)
    │
    ▼
Persisted as JSON (storage/processed/{document_id}.json)
```

### Why these design choices

- **PyMuPDF first, not OCR-by-default**: OCR is ~10-50x slower than native
  extraction and introduces recognition errors. Running it on every page
  regardless of need would make the system both slow and less accurate than
  necessary. We only escalate to OCR when native extraction genuinely fails.
- **Table detection triggers pdfplumber, not OCR**: PyMuPDF's plain-text mode
  flattens tables into unreadable text soup. pdfplumber's table extraction
  preserves row/column structure, which matters a lot for tax documents.
- **Hybrid classification (heuristic → LLM)**: matches your instruction that
  classification should be "user can tag, system suggests if blank." A pure
  LLM call on every untagged upload is unnecessary cost/latency when a
  regex pass resolves the obvious cases (a judgment almost always says
  "IN THE ... COURT" and "PETITIONER"/"RESPONDENT" on page 1).
- **Repository pattern for storage**: Milestone 1 only needs JSON-on-disk,
  but Milestones 3+ need the same data in Qdrant/Postgres/Elasticsearch.
  Isolating storage behind `DocumentRepository` means later milestones swap
  the implementation, not every caller.

### Project structure

```
backend/
├── app/
│   ├── core/
│   │   ├── config.py           # centralized settings (env-driven)
│   │   └── logging_config.py   # structured logging setup
│   ├── schemas/
│   │   └── document.py         # Pydantic contracts (ParsedDocument, Page, etc.)
│   ├── services/
│   │   ├── parser.py           # DocumentParser: PyMuPDF + pdfplumber + Tesseract
│   │   ├── classifier.py       # DocumentClassifier: heuristic + LLM fallback
│   │   └── storage.py          # DocumentRepository: persistence layer
│   ├── api/routes/
│   │   └── upload.py           # POST /upload, GET /documents, DELETE /document
│   └── main.py                 # FastAPI app entrypoint
├── tests/
│   └── test_parser.py          # integration tests against a real synthetic PDF
├── sample_docs/                # synthetic 3-page test PDF (native + table + scanned)
├── requirements.txt
├── .env.example
└── README.md
```

### Running it

```bash
cd backend
pip install -r requirements.txt
sudo apt-get install tesseract-ocr   # if not already present
cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

Then:

```bash
# Health check
curl http://localhost:8000/health

# Upload without a type tag (auto-classified)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@sample_docs/sample_legal_doc_final.pdf"

# Upload with an explicit type tag (trusted as-is)
curl -X POST "http://localhost:8000/api/v1/documents/upload?document_type=judgment" \
  -F "file=@sample_docs/sample_legal_doc_final.pdf"

# List / fetch / delete
curl http://localhost:8000/api/v1/documents
curl http://localhost:8000/api/v1/documents/{document_id}
curl -X DELETE http://localhost:8000/api/v1/documents/{document_id}
```

### Testing

```bash
PYTHONPATH=. pytest tests/ -v
```

10 tests, all passing against a real synthetic PDF with three deliberately
different page types:
- Page 1: native digital text (judgment boilerplate)
- Page 2: native digital text (statutory reference)
- Page 3: **image-only** page (no text layer at all) — forces the real OCR
  path, not a mock. Verified OCR confidence ~95% and correct text recovery.

This was also verified end-to-end through the live HTTP API (not just unit
tests): upload → auto-classification (86% confidence, correctly identified
as "judgment") → page-level retrieval → deletion, all functioning correctly.

### API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health check |
| `/api/v1/documents/upload` | POST | Upload a PDF; optional `document_type` query param |
| `/api/v1/documents` | GET | List all ingested documents (summary view) |
| `/api/v1/documents/{document_id}` | GET | Fetch full parsed document (all pages + metadata) |
| `/api/v1/documents/{document_id}` | DELETE | Remove a document |

## Milestone 2: Structure-Aware Chunking (complete)

### What this module does

Splits each parsed document into retrieval-ready chunks, using **legal
structure first** (Section/Article/Clause/Chapter/Part headers, numbered
judgment paragraphs) rather than blind fixed-size windows. Only sections
that are still too large after structural splitting get size-capped, and
that size-cap prefers sentence boundaries and adds overlap so meaning isn't
lost across the cut. Every chunk carries exact page provenance, including
chunks that span a page boundary.

### Why structure-aware, not fixed-size

Fixed-size chunking with overlap is simple but blind to legal meaning: it
will happily cut "Section 80G(5)(iv) requires X" from "...provided that Y"
into two different chunks, so a retrieval that finds the requirement misses
its condition. In the real test document, our chunker keeps the full
"Section 80G(5)(iv) requires... shall bear the registration number granted
by the Commissioner" as **one chunk** - verified by an explicit test
(`test_section_80g_paragraph_is_a_coherent_chunk`).

Small numbered-paragraph sections (e.g. a 2-sentence paragraph) are merged
with neighbors up to `chunk_min_chars`, so the index isn't cluttered with
near-empty fragments that add noise without retrieval value. Sections that
exceed `chunk_max_chars` are still split, but using a sentence-boundary
sliding window with character overlap - so size-capping never produces a cut
mid-sentence, and context survives the cut via overlap.

### How page provenance survives chunking

The chunker concatenates all page texts into one string while recording the
exact character range each page occupies. After structural splitting (and
any size-cap sub-splitting), each resulting chunk's character range is
mapped back against that page-offset table, producing:
- `page_start` / `page_end` - the page range for quick citation
- `page_spans` - exact character counts per page, for chunks that
  legitimately straddle a page boundary

### API additions

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/documents/{document_id}/chunks` | POST | Chunk a previously-uploaded document, persist and return the result |
| `/api/v1/documents/{document_id}/chunks` | GET | Retrieve a previously-generated chunking result |

Deleting a document (`DELETE /documents/{id}`) now also deletes its
associated chunks, so re-uploading the same file later can't resurrect stale
chunk data under a reused `document_id`.

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 18 tests total, Milestones 1 + 2
```

8 new tests in `tests/test_chunker.py`, including:
- Structural marker detection (5 numbered paragraphs found in the sample doc)
- Page-span correctness for every chunk
- The Section 80G coherence test described above
- An artificially long single-paragraph document to exercise the size-cap +
  overlap path specifically (not triggered by the short sample doc)

Verified end-to-end via live HTTP calls too: upload → generate chunks →
retrieve chunks, confirming persistence and correct structural labels
(`"1."`, `"2."`, `"4."`) with accurate page attribution per chunk.

## Milestone 3: Vector Search with Qdrant (complete)

### What this module does

Embeds each chunk and indexes it into Qdrant, with every field the citation
formatter will eventually need (document, page range, structural label,
chunk text) stored as point **payload** — so a search hit becomes a citation
without a second database lookup. Supports two embedding backends behind one
interface, selected by config: **BGE** (`BAAI/bge-large-en-v1.5`, local,
free) and **OpenAI** (`text-embedding-3-large`, API-based). Default is BGE,
per your instruction.

### A note on what could and couldn't be verified in this sandbox

This development sandbox's network allow-list covers PyPI/GitHub/npm but
**not** `huggingface.co` or `api.openai.com` (confirmed: both return 403
from the egress proxy). That means:

- ✅ **Fully implemented and tested**: the Qdrant integration itself — real
  embedded Qdrant, real collection creation, real cosine-similarity search,
  real payload storage/retrieval, real filtering by document, real deletion.
  This is tested with a small deterministic `FakeEmbeddingProvider` (see
  `tests/test_vector_store.py`) so the plumbing is genuinely proven, not
  mocked away.
- ⚠️ **Implemented but not runnable here**: the actual BGE model download
  (`sentence-transformers`) and the actual OpenAI API call. Both are real,
  complete production code (`app/services/embeddings.py`) — you'll need to
  run `pytest tests/test_vector_store.py -k bge` or hit `/api/v1/search` on
  a machine with normal internet access to confirm real semantic embeddings
  end-to-end. I'd recommend doing that before treating this milestone as
  fully verified in your own environment.

### Why one interface, two providers

BGE runs locally (no API key, no per-call cost, but needs ~1.3GB of weights
downloaded once and local CPU/GPU for inference). OpenAI's
`text-embedding-3-large` needs an API key and costs per token, but requires
no local compute. Neither is universally better, so both implement
`EmbeddingProvider` (`embed_texts`, `embed_query`, `dimension`), and nothing
in the vector store or retrieval layer needs to know which is active — only
the collection's fixed vector dimension matters. Swapping is a one-line
`.env` change: `EMBEDDING_PROVIDER=bge` or `EMBEDDING_PROVIDER=openai`.

BGE's asymmetric training convention (queries need an instruction prefix,
documents don't) is handled inside `BGEEmbeddingProvider.embed_query` so
callers never have to think about it.

### A third provider, added while building Milestone 8: `hash`

Verifying the frontend's real, end-to-end user journey (upload → process →
ask → stream → cite) in this sandbox needed *some* working embedding path,
and neither BGE nor OpenAI can reach their servers here. Rather than fake
the verification, a genuinely useful production pattern was added instead:
`HashEmbeddingProvider`, a deterministic, dependency-free embedding using
the classic "hashing trick" over character trigrams — no model download,
no API key, no network at all. It's meaningfully weaker than BGE/OpenAI
(captures shared word-roots and surface overlap, not true semantic
meaning), but it's a real, fully-tested option (`EMBEDDING_PROVIDER=hash`)
useful for CI pipelines, offline demos, or any network-restricted
environment — hybrid search's BM25 side compensates for its weaker
semantic signal, since exact-term precision doesn't depend on embedding
quality at all. This is what actually powered the live frontend
verification screenshots in Milestone 8.

### Qdrant deployment modes

- `QDRANT_MODE=local` (default): embedded, file-backed Qdrant — no server
  process needed, good for local dev/demo. Single-process only (holds a file
  lock).
- `QDRANT_MODE=server`: connects to a real Qdrant instance over host:port —
  what Milestone 10's docker-compose will use.

### API additions

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/documents/{document_id}/index` | POST | Embed and index a document's chunks into Qdrant |
| `/api/v1/search` | POST | Vector similarity search; body: `{query, top_k, document_id?, document_type?}` |

Document deletion now also removes that document's vectors from Qdrant.

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 26 tests total, Milestones 1 + 2 + 3
```

8 new tests in `tests/test_vector_store.py`: indexing count correctness,
search returning accurate citation payload with the right top hit, document
filtering, deletion, vector/chunk alignment validation, and embedding
provider factory behavior (including that `OpenAIEmbeddingProvider` fails
fast with a clear error if no API key is configured).

### Running with real embeddings (do this on a machine with internet access)

```bash
# BGE (default) - first call downloads ~1.3GB of weights
EMBEDDING_PROVIDER=bge uvicorn app.main:app --port 8000

# OpenAI
EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-... uvicorn app.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/api/v1/documents/{document_id}/index
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the conditions for claiming a Section 80G deduction?", "top_k": 5}'
```

## Milestone 4: Keyword Search / BM25 (complete)

### What this module does

Indexes chunk text, section labels, and metadata for keyword (BM25) search,
so exact legal terms - section numbers ("80G"), case numbers ("1234/2024"),
statute names - are retrievable even when they're too rare or specific for
vector search to rank highly. This is what Milestone 5 will fuse with
vector search results.

### Another honest sandbox constraint, handled the same way as Milestone 3

This development sandbox has **no Docker at all** (no daemon, no CLI) and
the network allow-list doesn't include Docker Hub or Elastic's registries -
so a real Elasticsearch cluster genuinely cannot run here, not even
temporarily. Rather than write untested ES code, this milestone ships two
providers behind one `KeywordSearchProvider` interface:

- ✅ **`BM25LocalProvider`** (default, `bm25_local`): in-process BM25 Okapi
  via `rank_bm25`, persisted to disk as JSON, **no server needed at all**.
  Fully implemented and fully tested here - real scoring, real persistence
  across process restarts, real filtering, real deletion. Verified end-to-end
  through the live API too: an exact-term query for "80G" correctly ranked
  the paragraph containing it as the top hit, and a case-number query
  ("1234/2024") correctly retrieved the right chunk.
- ⚠️ **`ElasticsearchProvider`** (`elasticsearch`): complete, real production
  code (indexes Title/Body/Section/Metadata per the assignment spec, uses
  ES's default BM25 similarity) but **not runnable in this sandbox**. Spin it
  up with the included `docker-compose.yml` on a machine with Docker, point
  `KEYWORD_SEARCH_PROVIDER=elasticsearch` at it, and re-run
  `pytest tests/test_keyword_search.py` to verify it for real before
  considering this half of the milestone proven.

Both providers score with the same underlying algorithm (BM25) - the
difference is operational maturity (a real ES cluster gives you scaling and
monitoring), not retrieval theory.

### `docker-compose.yml` (new, project root)

Stands up real Qdrant and Elasticsearch containers so you can verify both
Milestone 3's OpenAI/BGE embeddings and Milestone 4's Elasticsearch provider
against real servers instead of the embedded/local dev fallbacks:

```bash
docker compose up -d
# then set QDRANT_MODE=server and KEYWORD_SEARCH_PROVIDER=elasticsearch in .env
```

### API additions

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/documents/{document_id}/keyword-index` | POST | Index a document's chunks into BM25/Elasticsearch |
| `/api/v1/keyword-search` | POST | BM25 keyword search; same request/response shape as `/search` |

Document deletion now also removes that document's entries from the
keyword index.

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 36 tests total, Milestones 1-4
```

10 new tests in `tests/test_keyword_search.py`: indexing count, exact-term
ranking correctness (the core reason BM25 exists in a hybrid system),
case-number retrieval, document filtering, deletion, re-indexing without
duplication, and corpus persistence across simulated process restarts.

## Milestone 5: Hybrid Search via Reciprocal Rank Fusion (complete)

### What this module does

Fuses vector search (Milestone 3) and keyword/BM25 search (Milestone 4)
results using **Reciprocal Rank Fusion (RRF)**, so a query benefits from
both semantic similarity and exact-term precision rather than depending on
either alone.

### Why RRF instead of a weighted score blend

Vector search produces cosine similarities (roughly 0-1); BM25 produces
unbounded scores that scale with query length and corpus statistics.
Blending them with fixed weights (e.g. `0.6*vector + 0.4*keyword`) requires
normalizing two incomparable distributions, and that normalization tends to
need re-tuning per corpus/document-type. RRF sidesteps this: it only looks
at **rank position** in each ranker's result list, not the raw score, so
it's inherently robust to two rankers scoring on completely different
scales. `test_rank_position_matters_not_raw_score_scale` proves this
directly: a vector hit with a "modest" 0.55 cosine score and a keyword hit
with a "huge" 999.0 BM25 score, both ranked #1 in their own list, get
**identical** RRF contributions.

Formula: `score(chunk) = Σ 1 / (rrf_k + rank_in_that_ranker)`, summed across
every ranker that returned the chunk (`rrf_k=60`, the standard default from
the original RRF paper). A chunk found by both rankers naturally outranks
one found by only one, without any manual weighting.

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 42 tests total, Milestones 1-5
```

11 new tests in `tests/test_hybrid_search.py`, split into two levels:
- **Unit-level** (`TestReciprocalRankFusionUnit`): the fusion math itself,
  tested against hand-constructed rank lists - formula correctness, that
  rank position (not raw score scale) drives the result, that a chunk found
  by both rankers outranks single-ranker hits, and that a chunk missing
  from one ranker is still included rather than penalized to zero.
- **Integration-level** (`TestHybridSearchIntegration`): a real embedded
  Qdrant + a real BM25LocalProvider + a deterministic fake embedder, wired
  together through the actual `HybridSearchService`, indexing a small
  corpus and confirming the fused output is correct end-to-end.

### A genuine bug found and fixed along the way

Building the integration test surfaced a real `rank_bm25` edge case: its
IDF formula gives `idf == 0` for any term appearing in exactly half the
corpus - and with a tiny 2-chunk test fixture, that zeroed out an *exact*
statutory-term match entirely. This isn't just a test artifact: a legal KB
early in its life (only 2-3 documents indexed) could hit the same thing in
production and silently drop an exact citation match, which directly
undermines this system's anti-hallucination goal. Fixed by flooring IDF at
a small positive epsilon in `BM25LocalProvider._ensure_model` (the same kind
of smoothing production search engines like Elasticsearch already apply) -
see the comment there for details.

### API additions

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/hybrid-search` | POST | RRF-fused vector + keyword search; same request shape as `/search`, response includes `rrf_score`, per-ranker rank/score, and `matched_by` for transparency |

### A note on live end-to-end testing of this endpoint

Like Milestone 3, `/api/v1/hybrid-search` depends on the real BGE/OpenAI
embedding call at query time, which this sandbox can't reach (no network
path to huggingface.co / api.openai.com). The fusion logic itself is fully
verified (above); to confirm the whole HTTP path end-to-end, run the server
on a machine with normal internet access and try:

```bash
curl -X POST http://localhost:8000/api/v1/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the conditions for claiming a Section 80G deduction?", "top_k": 5}'
```

## Milestone 6: Retriever - MMR Diversity + Re-ranking (complete)

### What this module does

The last retrieval-quality gate before generation. Takes the wide candidate
pool from Milestone 5's hybrid search and:
1. **MMR-diversifies** it (drops near-duplicate chunks that all say the same
   thing, trading a controlled amount of relevance for genuine diversity)
2. **Re-ranks** the diversified set for true query relevance
3. **Filters by confidence**, dropping anything the pipeline itself doesn't
   trust rather than passing weak context to the LLM

### Why MMR before re-ranking, not after

Re-ranking (especially a cross-encoder) is the most expensive step per
candidate. Running MMR first narrows a wide, possibly-redundant candidate
pool down to a smaller, genuinely diverse set - so re-ranking cost stays
bounded, and the LLM's eventual context window isn't spent on 5
near-paraphrases of the same holding (a real pattern in judgments that
restate a ruling multiple times).

`test_diversity_preferred_over_near_duplicate_at_low_lambda` proves the
core MMR property directly: given a most-relevant chunk, a near-duplicate,
and a still-relevant-but-distinct third chunk, MMR correctly prefers the
distinct one over the duplicate once the top pick is already selected -
using pure synthetic vectors, independent of any embedding provider.

### Re-ranking: lightweight now, cross-encoder when you have network access

Same honest pattern as Milestones 3 and 4: `BAAI/bge-reranker-large` is a
real cross-encoder (jointly scores query+passage, catching relevance a
bi-encoder or BM25 can miss) but needs a HuggingFace download this sandbox
can't reach. So two providers exist behind one `Reranker` interface:

- ✅ **`LightweightReranker`** (default): blends normalized RRF score with
  query-term Jaccard overlap. Not a cross-encoder's semantic judgment, but a
  real, defensible heuristic - fully tested here, no dependencies, no
  download. Verified: exact query-term matches score higher than
  no-overlap text at equal upstream score; upstream score still
  differentiates candidates when term overlap is tied.
- ⚠️ **`CrossEncoderReranker`**: complete production code, untested here.
  Switch via `RERANKER_PROVIDER=cross_encoder` on a machine with HuggingFace
  access, then re-run `pytest tests/test_reranker.py` to verify for real.

### API additions

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/retrieve` | POST | Full pipeline: hybrid search → MMR → re-rank → confidence filter. Response includes `candidates_considered` and `below_confidence_threshold_count` for transparency. |

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 58 tests total, Milestones 1-6
```

16 new tests:
- `tests/test_mmr.py` (8 tests): pure vector-math correctness, including
  the lambda=1.0-reduces-to-plain-top-K property and the diversity-over-
  duplicate property described above.
- `tests/test_reranker.py` (8 tests): `LightweightReranker` scoring
  correctness, plus **two full-pipeline integration tests** wiring real
  Qdrant + real BM25 + real MMR + real lightweight re-ranking together
  end-to-end (one confirming high-confidence results with complete citation
  fields, one confirming an unreachable confidence threshold correctly
  drops every result rather than returning weak context).

## Milestone 7: Prompt Engineering & Grounded Answer Generation (complete)

### What this module does

Turns the retrieval pipeline (Milestones 1-6) into an actual Q&A system.
Given a question, it retrieves confidence-filtered chunks, builds a strict
citation-grounded prompt, calls an LLM, and runs the answer through a
citation formatter that **never trusts the model's own claims about page
numbers or document names** - it only trusts our own retrieval metadata.

### The core anti-hallucination design decision

The LLM is told to cite claims using simple inline tags - `[C1]`, `[C2]` -
referring to numbered context passages it was given. The citation formatter
then does the opposite of what you might expect: it does **not** ask the
model what page or document it used. It just finds the `[Cx]` tags in the
model's text and looks up the real page/document from our own
`RetrievedChunk` data - the same page numbers Milestone 1's parser
extracted directly from the PDF. The model's only real job is to place a
tag in roughly the right spot; it never gets an opportunity to invent a
page number, because it's never asked for one.

`test_citation_page_always_comes_from_retrieval_not_model_text` proves this
concretely: even when the model's answer *prose* claims a wrong page number
("According to page 999..."), the structured citation still correctly
reports page 42, because the citation was never sourced from the model's
text in the first place.

### Two more anti-hallucination layers, both enforced in code, not just the prompt

1. **Zero-evidence short-circuit**: if retrieval returns nothing (Milestone
   6's confidence filter dropped everything), the system returns the exact
   mandated phrase - `"I could not find supporting evidence."` - **without
   calling the LLM at all**. This is deterministic and free, and removes any
   chance of the model "being helpful" and guessing when there's genuinely
   no evidence. Verified directly: `test_zero_retrieved_chunks_never_calls_llm`
   asserts the fake LLM's call count is exactly 0.
2. **Uncited claims are untrusted**: if the model answers without a single
   `[Cx]` tag (a prompt violation), the system marks
   `has_sufficient_evidence = False` rather than presenting an unsupported
   answer as trustworthy - even though the model did technically respond.

### LLM providers: one real and tested, two real but undeployable here

The spec lists GPT-4.1, Claude, and Gemini. All three are implemented
behind one `LLMProvider` interface, but they land in genuinely different
places in this sandbox:

- ✅ **`AnthropicProvider`** (default): **`api.anthropic.com` is actually
  reachable from this sandbox's network allow-list** (confirmed: a request
  returns 404, not a proxy-blocked 403) - a real, different situation from
  Milestones 3/4/6's network-blocked gaps. The only missing piece is a real
  `ANTHROPIC_API_KEY`, which isn't configured in this environment. Set one
  and `tests/test_qa_service.py::TestRealAnthropicProvider` (currently
  auto-skipped) will run for real against actual Claude.
- ⚠️ **`OpenAIProvider`**: real code, but `api.openai.com` is outside the
  network allow-list (confirmed 403 in Milestone 3).
- ⚠️ **`GeminiProvider`**: real code using Google's current `google-genai`
  SDK (the older `google-generativeai` package is deprecated - caught and
  avoided during development), but `generativelanguage.googleapis.com` is
  also outside the allow-list.

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 75 tests total (74 run + 1 auto-skipped), Milestones 1-7

# With a real Anthropic key, the skipped test runs for real:
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_qa_service.py -v
```

16 new tests:
- `tests/test_citation_formatter.py` (12 tests): the anti-hallucination
  core, fully tested with hand-written answer strings - correct tag
  extraction, deduplication, multi-tag claims, graceful handling of
  out-of-range/hallucinated tags, and the page/document-always-from-
  retrieval guarantee.
- `tests/test_qa_service.py` (4 run + 1 skipped): full pipeline with a
  `FakeLLMProvider` (real retrieval, deterministic fake LLM response) -
  correct citations on a grounded answer, correct no-evidence handling,
  the zero-call short-circuit, and uncited-claim handling. Plus one real,
  auto-skipped Anthropic integration test.

### API additions

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/query` | POST | The full grounded Q&A endpoint: retrieval → prompt → LLM → citation formatting. Response includes `answer`, `citations` (page/document from OUR data), `has_sufficient_evidence`, and `model_used`. |
| `/api/v1/query/stream` | POST | Server-Sent Events version of the same pipeline: streams `{"type":"token","text":...}` events as the LLM generates, then one final `{"type":"done","citations":[...],...}` event once the full answer is known (citations can only be finalized after the complete text is available). Built for Milestone 8's frontend, which consumes it via `fetch()` + `ReadableStream` rather than the browser's `EventSource` API, since `EventSource` only supports GET and this needs a POST body. |

### What's next (Milestone 8)

The Next.js/React/Tailwind/Shadcn frontend: upload UI, search box, streaming
answers, and expandable evidence panels showing exactly the citations this
milestone produces.

---

## Milestone 8: Frontend (complete — see `../frontend/README.md`)

A real Next.js app (not a static mockup) lives in `../frontend`: upload
UI with live pipeline progress, a query console with real token-by-token
SSE streaming from `/api/v1/query/stream`, and an "Exhibit Ledger" citation
panel that renders exactly the citation data this backend's citation
formatter (Milestone 7) produces. Built, linted, and visually verified via
Playwright screenshots of a live server in this same sandbox — full details,
design system, and honest notes on what was/wasn't verifiable here are in
the frontend's own README.








---

## Milestone 9: Evaluation (complete)

### What this module does

A complete, production-shaped evaluation harness — not a one-off script —
that runs a golden dataset through the real pipeline (retrieval + QA) and
reports on retrieval quality, answer correctness, citation accuracy,
faithfulness, latency, and token usage. Built as its own package
(`evaluation/`) decoupled from `app/` so it can evolve independently and be
CI-gated without becoming part of the production API surface.

### Architecture

```
evaluation/
├── schema.py          # GoldenQuestion, ExpectedCitation, PerQuestionResult, EvaluationSummary
├── dataset_loader.py   # loads .json (full fidelity) or .csv (flat) datasets
├── metrics.py          # pure, independently-testable metric functions
├── runner.py           # orchestrates RetrievalService + QAService per question
├── report.py           # Markdown + self-contained HTML (inline SVG charts)
├── cli.py              # single-command entrypoint, CI-gate exit codes
├── golden_dataset.json # 18 example questions (real content, see below)
└── results/            # output directory (report.md, report.html, results.json)
```

### Metrics implemented

| Metric | What it measures | Rigor note |
|---|---|---|
| **Answer correctness** | Token-F1 (SQuAD-style) vs. ground truth; binary correct-refusal for `no_evidence_check` questions | Deliberate deterministic stand-in for an LLM-judge/semantic-similarity metric — see `metrics.py` docstring |
| **Citation precision / recall** | Actual citations vs. `expected_citations`, matched by document + **page-range overlap** (not exact equality) | — |
| **Retrieval Recall@K** | Whether expected passages appear anywhere in the top-K retrieved chunks | Measures the retriever independent of the LLM |
| **MRR** | Reciprocal rank of the first relevant retrieved chunk | — |
| **nDCG@K** | Ranking quality with binary relevance, standard log-discounted formula | — |
| **Faithfulness** | Fraction of answer sentences carrying a `[Cx]` citation tag | Structural proxy, NOT RAGAS's LLM-judged entailment check — explicitly documented as such |
| **Latency** | Wall-clock, retrieval vs. generation measured separately | — |
| **Token usage** | Input/output tokens, when the LLM provider reports them | `None` for providers/fakes that don't report usage (e.g. test doubles) |

Every metric is a pure function in `metrics.py`, tested directly against
hand-constructed `PerQuestionResult` objects — 26 tests in
`tests/test_evaluation_metrics.py`, no retrieval or LLM involved at all.

### The golden dataset

18 questions (not the full 100 — see "scaling" below), hand-authored
against the **actual real content** of `sample_docs/sample_legal_doc_final.pdf`
(the same judgment used throughout this project's tests): case number,
parties, judge, the Section 80G dispute, the statutory requirement, and the
court's holding. Includes:
- **`fact_lookup`** (8) — direct facts (case number, parties, amounts)
- **`statutory_requirement`** (2) — what the law actually requires
- **`holding`** (3) — what the court decided and why
- **`synthesis`** (2) — requires drawing on multiple pages coherently
- **`no_evidence_check`** (2) — questions about things genuinely absent
  from the document (penalty amount, counsel names), specifically testing
  the anti-hallucination fallback from Milestone 7, not retrieval quality

### Scaling to 100+ questions: zero code changes

Append more objects to `golden_dataset.json`'s array (or rows to a `.csv`
file — same schema, flatter) and re-run the same CLI command. Nothing in
`dataset_loader.py`, `runner.py`, or `metrics.py` assumes a fixed dataset
size or a specific document — the harness was built and tested against a
JSON array from the start, not a hardcoded 18-question script.

### Running it

```bash
# Single command - runs the full pipeline against every question
python -m evaluation.cli --dataset evaluation/golden_dataset.json --output-dir evaluation/results

# CI usage - non-zero exit code if quality regresses below a threshold
python -m evaluation.cli --dataset evaluation/golden_dataset.json \
    --min-answer-correctness 0.6 --min-faithfulness 0.7
```

Produces `report.md`, `report.html` (self-contained, inline SVG charts, no
external image files or JS dependencies), and `results.json` (full
per-question data for further analysis).

### Testing

```bash
PYTHONPATH=. pytest tests/ -v   # 129 tests total (128 run + 1 auto-skipped), Milestones 1-9
```

47 new tests across four files:
- `test_evaluation_metrics.py` (26): every metric function, pure and isolated.
- `test_evaluation_dataset_loader.py` (8): JSON/CSV loading, round-tripping,
  error handling, and validation of the real shipped `golden_dataset.json`.
- `test_evaluation_runner.py` (9): the **real** pipeline (real parser, real
  chunker, real embedded Qdrant, real BM25, real MMR, real reranker) run
  against all 18 real golden questions, using a `PromptEchoLLM` test double
  that answers from the actual retrieved passage text — proving the harness
  computes genuine, non-trivial per-question metric variance, not constants.
- `test_evaluation_cli.py` (4): the single-command entrypoint end-to-end,
  including both CI-gate outcomes (pass and fail) and graceful handling of
  a missing dataset file.

### A real evaluation run, actually executed in this sandbox

Unlike the BGE/OpenAI/cross-encoder/Elasticsearch gaps elsewhere in this
project, **the evaluation harness itself needed no network access at all**
to prove out — it was run for real end-to-end using the offline
`EMBEDDING_PROVIDER=hash` fallback (Milestone 8) for genuine retrieval, and
a canned-but-content-aware LLM stand-in (since no real `ANTHROPIC_API_KEY`
is configured here) for answer generation. The real, inspectable output of
that run ships in `evaluation/results/` (`report.md`, `report.html`,
`results.json`). Retrieval-side metrics from that run are genuinely strong
(`avg_retrieval_recall_at_5 = 1.0`, `avg_mrr = 0.92`, `avg_ndcg_at_5 = 0.94`)
because the retrieval pipeline is fully real; answer-side metrics
(`avg_answer_correctness = 0.26`, `avg_faithfulness = 0.16`) are low **only**
because the stand-in LLM is a crude word-overlap heuristic, not a real
model — re-run with a real `ANTHROPIC_API_KEY` to get numbers that actually
reflect answer quality.

### What's next (Milestone 10 — complete, see `../DEPLOYMENT.md`)

Milestone 10 is done: multi-stage production Dockerfiles for both services
(non-root users, healthchecks, gunicorn+uvicorn workers on the backend,
Next.js standalone output on the frontend), a full `docker-compose.yml`
(backend, frontend, Qdrant, Elasticsearch — deliberately no Postgres, since
nothing in this app uses one), deployment configs for Render/Railway/Vercel,
a 5-job GitHub Actions CI pipeline, and a real, tested rate-limiting
middleware (`app/core/rate_limit.py`) as a concrete production-hardening
addition, not just a documentation bullet point. Full details, an honest
verification checklist (what was actually confirmed in this sandbox vs.
what needs a real Docker/cloud environment), and operational guidance
(SSL, secrets, monitoring, scaling, backups, cost) are in
`../DEPLOYMENT.md`.
