from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Legal & Tax RAG System"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    base_dir: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = base_dir / "data" / "raw"
    processed_data_dir: Path = base_dir / "data" / "processed"
    golden_dataset_path: Path = base_dir / "data" / "golden" / "golden_dataset.csv"
    faiss_index_dir: Path = base_dir / "data" / "processed" / "faiss_index"
    bm25_index_path: Path = base_dir / "data" / "processed" / "bm25_index.pkl"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "legal_rag"
    postgres_user: str = "legal_rag"
    postgres_password: str = "legal_rag"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    use_redis_cache: bool = False

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "legal_chunks"
    elasticsearch_enabled: bool = True

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"
    graph_rag_enabled: bool = True

    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    min_chunk_size_tokens: int = 100
    parent_chunk_size_tokens: int = 3000

    faiss_top_k: int = 50
    bm25_top_k: int = 50
    elasticsearch_top_k: int = 50
    rrf_k_constant: int = 60
    fusion_top_k: int = 30
    final_top_k: int = 8
    neighbor_window: int = 1
    enable_query_expansion: bool = True
    enable_multi_query: bool = True
    multi_query_count: int = 3

    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    llm_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = Field(default="", repr=False)
    gemini_api_key: str = Field(default="", repr=False)
    llm_model_name: str = "gemini-3.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500
    llm_request_timeout: int = 60

    @property
    def resolved_llm_api_key(self) -> str:
        return self.llm_api_key or self.gemini_api_key

    api_host: str = "0.0.0.0"
    api_port: int = 8004
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    max_upload_mb: int = 200

    eval_recall_k_values: list[int] = [5, 10]
    eval_report_dir: Path = base_dir / "docs" / "eval_reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
