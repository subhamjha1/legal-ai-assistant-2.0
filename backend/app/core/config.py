"""
Centralized application configuration.

Why this exists:
Hardcoding paths, thresholds, and model names throughout the codebase makes a
system brittle and impossible to tune without touching business logic. This
module is the single source of truth for every tunable value, loaded from
environment variables (with sane defaults for local dev).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App metadata ---
    app_name: str = "Legal AI Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"

    # --- Storage paths ---
    base_dir: Path = Path(__file__).resolve().parents[2]
    upload_dir: Path = base_dir / "storage" / "uploads"
    processed_dir: Path = base_dir / "storage" / "processed"

    # --- Ingestion tuning ---
    # A page is treated as "scanned" (and routed to OCR) when native text
    # extraction yields fewer than this many characters.
    ocr_trigger_char_threshold: int = 20
    ocr_language: str = "eng"
    max_upload_size_mb: int = 100
    allowed_extensions: tuple[str, ...] = (".pdf",)

    # --- Document classification ---
    # Confidence below which we fall back to the LLM-suggested type instead
    # of trusting a purely heuristic keyword match.
    classification_min_confidence: float = 0.55

    # --- Chunking (Milestone 2) ---
    # Structure-aware chunking splits on legal markers (Section/Article/Clause/
    # numbered paragraphs) first; these caps only apply to sections that are
    # still too large after structural splitting.
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 200  # sections smaller than this get merged with a neighbor

    # --- Embeddings (Milestone 3) ---
    # "bge" runs BAAI/bge-large-en-v1.5 locally via sentence-transformers (free,
    # no API key, needs the model weights downloaded once from HuggingFace).
    # "openai" calls text-embedding-3-large via the OpenAI API (needs an API key,
    # no local model download, generally higher quality on out-of-domain text).
    embedding_provider: str = "bge"
    bge_model_name: str = "BAAI/bge-large-en-v1.5"
    bge_embedding_dim: int = 1024
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dim: int = 3072
    openai_api_key: str = ""
    embedding_batch_size: int = 32

    # --- Vector store (Qdrant) ---
    # "local" uses an embedded, file-backed Qdrant instance (no server needed;
    # good for dev/demo, single-process only). "server" connects to a real
    # Qdrant instance (used in docker-compose / production, Milestone 10).
    qdrant_mode: str = "local"
    qdrant_local_path: Path = base_dir / "storage" / "qdrant_local"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "legal_chunks"
    vector_search_top_k: int = 10

    # --- Keyword search (Milestone 4) ---
    # "bm25_local" runs an in-process rank_bm25 index, persisted to disk as
    # JSON - genuinely testable in any environment, no server/JVM needed.
    # "elasticsearch" connects to a real Elasticsearch cluster - the
    # production path, wired for docker-compose (Milestone 10).
    keyword_search_provider: str = "bm25_local"
    bm25_local_storage_path: Path = base_dir / "storage" / "bm25_local"
    elasticsearch_host: str = "http://localhost:9200"
    elasticsearch_index_name: str = "legal_chunks"
    keyword_search_top_k: int = 10

    # --- Hybrid search (Milestone 5) ---
    # Reciprocal Rank Fusion: score(chunk) = sum over each ranker of
    # 1 / (rrf_k + rank_in_that_ranker). rrf_k=60 is the standard default
    # from the original RRF paper (Cormack et al.) - it discounts rank
    # differences at the tail without needing any score normalization
    # between vector cosine similarity and BM25 scores, which live on
    # completely different scales.
    rrf_k: int = 60
    # How many candidates to pull from EACH ranker before fusing - wider
    # than the final top_k so a chunk that ranks #8 in vector search but
    # #1 in keyword search still has a chance to surface after fusion.
    hybrid_candidate_pool_size: int = 25
    hybrid_search_top_k: int = 10

    # --- Retriever: MMR diversity (Milestone 6) ---
    # MMR trades off relevance against diversity when selecting the final
    # candidate set: lambda=1 is pure relevance (equivalent to plain top-K),
    # lambda=0 is pure diversity. 0.5 is a balanced default - avoids
    # returning 5 chunks that all say the same thing while still favoring
    # genuinely relevant results over merely "different" ones.
    mmr_lambda: float = 0.5
    mmr_pool_size: int = 20  # how many hybrid-fused candidates feed into MMR
    retriever_top_k: int = 8  # how many MMR selects, before re-ranking

    # --- Re-ranking (Milestone 6) ---
    # "cross_encoder" runs BAAI/bge-reranker-large (real cross-encoder,
    # jointly scores query+passage - highest quality, needs a HuggingFace
    # model download). "lightweight" is a dependency-free stand-in blending
    # normalized RRF score with query-term overlap - lower quality than a
    # real cross-encoder, but fully local and testable with no model
    # download, and still meaningfully better than raw RRF order alone.
    reranker_provider: str = "lightweight"
    reranker_model_name: str = "BAAI/bge-reranker-large"
    lightweight_rerank_alpha: float = 0.5  # weight of RRF score vs. term overlap
    rerank_final_top_k: int = 5
    # Chunks scoring below this confidence are dropped entirely rather than
    # returned as weak/misleading context to the LLM (Milestone 7).
    rerank_min_confidence: float = 0.15

    # --- LLM / Answer generation (Milestone 7) ---
    llm_provider: str = "anthropic"
    anthropic_qa_model: str = "claude-sonnet-4-6"
    openai_qa_model: str = "gpt-4.1"
    gemini_qa_model: str = "gemini-2.0-flash"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0  # deterministic, citation-grounded answers - not creative writing
    gemini_api_key: str = ""
    no_evidence_phrase: str = "I could not find supporting evidence."

    # --- Rate limiting (Milestone 10 production hardening) ---
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    # Paths exempt from rate limiting - health checks are hit frequently by
    # orchestrators/load balancers and shouldn't count against real traffic.
    rate_limit_exempt_paths: tuple[str, ...] = ("/health",)


    # --- LLM (used later for classification-suggestion; wired in Milestone 7 fully) ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so we parse env vars once per process."""
    return Settings()
