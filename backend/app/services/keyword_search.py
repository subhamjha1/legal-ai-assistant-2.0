"""
Keyword search providers (Milestone 4).

Why one interface, two providers:
Elasticsearch is the production target (Milestone 4's spec explicitly calls
for it, and Milestone 10's docker-compose stands up a real cluster), but
this development sandbox has no Docker and no network path to Elastic's or
Docker Hub's registries - a real ES cluster genuinely cannot run here. Rather
than write ES code that's never exercised, we also implement an in-process
BM25 provider (`rank_bm25`) that requires no server at all, so the actual
retrieval logic (indexing, scoring, filtering, deletion) is genuinely tested
in this environment. Both implement the same `KeywordSearchProvider`
interface, so Milestone 5's hybrid fusion doesn't care which one is active.

Both providers score with the *same underlying algorithm* (BM25) - the
Elasticsearch spec requirement "Implement BM25" is satisfied by ES's default
similarity, and by rank_bm25's Okapi BM25 implementation locally. This is
not a downgrade in retrieval theory, only in operational maturity (a real ES
cluster gives you monitoring, scaling, and Kibana; the local provider gives
you the same ranking algorithm without any of that infrastructure).
"""
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.chunk import ChunkingResult
from app.schemas.document import ParsedDocument

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer. Deliberately simple (no
    stemming/stopwords) so exact legal terms - section numbers, case
    citations - aren't mangled by aggressive normalization."""
    return _TOKEN_PATTERN.findall(text.lower())


class KeywordSearchProvider(ABC):
    """Common interface every keyword-search backend must implement."""

    @abstractmethod
    def index_chunks(self, document: ParsedDocument, chunking_result: ChunkingResult) -> int:
        """Index all chunks of a document. Returns count indexed."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[dict]:
        """Keyword (BM25) search. Returns payload dicts augmented with a
        `score` field, in the same shape as VectorStore.search results, so
        Milestone 5's fusion logic can treat both result lists uniformly."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove all indexed chunks for a document."""

    @abstractmethod
    def count(self) -> int:
        """Total number of indexed chunks."""


# ---------------------------------------------------------------------- #
# BM25 local provider (in-process, no server - genuinely testable here)
# ---------------------------------------------------------------------- #
@dataclass
class _IndexedChunk:
    """One entry in the BM25 corpus, carrying the same payload fields as the
    vector store so search results are shaped identically either way."""
    chunk_id: str
    document_id: str
    text: str
    tokens: list[str]
    structural_label: str | None
    page_start: int
    page_end: int
    original_filename: str
    document_type: str


class BM25LocalProvider(KeywordSearchProvider):
    """
    In-process BM25 (Okapi) via rank_bm25, persisted to a JSON corpus file so
    it survives process restarts. The BM25 model itself is rebuilt from the
    corpus lazily (on first search after a corpus change) rather than
    incrementally, since rank_bm25 has no native incremental-update API and
    legal-document corpora here are small enough that a full rebuild is fast.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.bm25_local_storage_path.mkdir(parents=True, exist_ok=True)
        self._corpus_path = self.settings.bm25_local_storage_path / "corpus.json"
        self._corpus: list[_IndexedChunk] = self._load_corpus()
        self._bm25 = None  # lazily (re)built on next search

    # -- persistence -- #
    def _load_corpus(self) -> list[_IndexedChunk]:
        if not self._corpus_path.exists():
            return []
        raw = json.loads(self._corpus_path.read_text(encoding="utf-8"))
        return [_IndexedChunk(**item) for item in raw]

    def _persist_corpus(self) -> None:
        self._corpus_path.write_text(
            json.dumps([asdict(c) for c in self._corpus], indent=2), encoding="utf-8"
        )

    def _invalidate(self) -> None:
        self._bm25 = None

    def _ensure_model(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            tokenized = [c.tokens for c in self._corpus] or [[]]
            model = BM25Okapi(tokenized)
            # Floor IDF at a small positive epsilon. The raw Robertson-Sparck
            # Jones IDF used by rank_bm25 gives idf == 0 for any term that
            # appears in exactly half the corpus (a real, not just
            # theoretical, risk early in a legal KB's life when only a
            # handful of documents are indexed) - which silently zeroes that
            # term's contribution to every score, even for an otherwise
            # exact statutory-term match. Production search engines (e.g.
            # Elasticsearch) avoid this with similar smoothing; we do the
            # same here rather than let corpus size accidentally hide exact
            # matches, which would directly undermine this system's
            # citation-accuracy goal.
            min_idf = 1e-4
            for term, idf in model.idf.items():
                if idf < min_idf:
                    model.idf[term] = min_idf
            self._bm25 = model
        return self._bm25

    # -- interface -- #
    def index_chunks(self, document: ParsedDocument, chunking_result: ChunkingResult) -> int:
        # Remove any stale entries for this document first (re-indexing case).
        self._corpus = [c for c in self._corpus if c.document_id != document.document_id]

        for chunk in chunking_result.chunks:
            self._corpus.append(
                _IndexedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=document.document_id,
                    text=chunk.text,
                    tokens=_tokenize(chunk.text),
                    structural_label=chunk.structural_label,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    original_filename=document.metadata.original_filename,
                    document_type=document.metadata.document_type.value,
                )
            )

        self._persist_corpus()
        self._invalidate()
        logger.info("BM25-indexed %d chunks for document %s", len(chunking_result.chunks), document.document_id)
        return len(chunking_result.chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[dict]:
        if not self._corpus:
            return []

        model = self._ensure_model()
        query_tokens = _tokenize(query)
        scores = model.get_scores(query_tokens)

        ranked = sorted(
            ((score, chunk) for score, chunk in zip(scores, self._corpus)),
            key=lambda pair: pair[0],
            reverse=True,
        )

        hits = []
        for score, chunk in ranked:
            if score <= 0:
                continue  # BM25 score of 0 means no query terms matched at all
            if document_id and chunk.document_id != document_id:
                continue
            if document_type and chunk.document_type != document_type:
                continue
            hit = asdict(chunk)
            hit.pop("tokens")
            hit["score"] = float(score)
            hits.append(hit)
            if len(hits) >= top_k:
                break
        return hits

    def delete_document(self, document_id: str) -> None:
        before = len(self._corpus)
        self._corpus = [c for c in self._corpus if c.document_id != document_id]
        if len(self._corpus) != before:
            self._persist_corpus()
            self._invalidate()
            logger.info("Removed BM25 entries for document %s", document_id)

    def count(self) -> int:
        return len(self._corpus)


# ---------------------------------------------------------------------- #
# Elasticsearch provider (production path; requires a real ES cluster)
# ---------------------------------------------------------------------- #
class ElasticsearchProvider(KeywordSearchProvider):
    """
    Real Elasticsearch-backed keyword search. Indexes Title (structural
    label / filename), Body (chunk text), Section (structural label), and
    Metadata (document_id, page range, document type) per the assignment
    spec. Uses ES's default BM25 similarity - no custom similarity config
    needed since that's already Elasticsearch's out-of-the-box scoring.

    NOTE: this class is complete, production-ready code, but could not be
    exercised against a real cluster in this development sandbox (no Docker,
    no network path to a registry to pull the ES image). Verify with:
        docker compose up elasticsearch
        pytest tests/test_keyword_search.py -k elasticsearch --run-es
    on a machine with Docker available.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        from elasticsearch import Elasticsearch

        self._client = Elasticsearch(self.settings.elasticsearch_host)
        self._index = self.settings.elasticsearch_index_name
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=self._index):
            return
        self._client.indices.create(
            index=self._index,
            mappings={
                "properties": {
                    "title": {"type": "text"},
                    "body": {"type": "text"},
                    "section": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                }
            },
        )

    def index_chunks(self, document: ParsedDocument, chunking_result: ChunkingResult) -> int:
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {
                    "title": document.metadata.original_filename,
                    "body": chunk.text,
                    "section": chunk.structural_label,
                    "document_id": document.document_id,
                    "document_type": document.metadata.document_type.value,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
            }
            for chunk in chunking_result.chunks
        ]
        success, _ = bulk(self._client, actions)
        return success

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[dict]:
        filters = []
        if document_id:
            filters.append({"term": {"document_id": document_id}})
        if document_type:
            filters.append({"term": {"document_type": document_type}})

        response = self._client.search(
            index=self._index,
            size=top_k,
            query={
                "bool": {
                    "must": [{"multi_match": {"query": query, "fields": ["body^2", "title", "section"]}}],
                    "filter": filters,
                }
            },
        )
        hits = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            hits.append(
                {
                    "chunk_id": hit["_id"],
                    "document_id": source["document_id"],
                    "text": source["body"],
                    "structural_label": source.get("section"),
                    "page_start": source["page_start"],
                    "page_end": source["page_end"],
                    "original_filename": source["title"],
                    "document_type": source["document_type"],
                    "score": hit["_score"],
                }
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        self._client.delete_by_query(
            index=self._index, query={"term": {"document_id": document_id}}
        )

    def count(self) -> int:
        return self._client.count(index=self._index)["count"]


def get_keyword_search_provider(provider_name: str | None = None) -> KeywordSearchProvider:
    settings = get_settings()
    name = provider_name or settings.keyword_search_provider

    if name == "bm25_local":
        return BM25LocalProvider()
    if name == "elasticsearch":
        return ElasticsearchProvider()
    raise ValueError(f"Unknown keyword search provider '{name}'. Expected 'bm25_local' or 'elasticsearch'.")


@lru_cache
def get_cached_keyword_provider() -> KeywordSearchProvider:
    """Cached so BM25LocalProvider's in-memory corpus (loaded from disk) is
    built once per process, and so ElasticsearchProvider's client connection
    is reused across requests."""
    return get_keyword_search_provider()
