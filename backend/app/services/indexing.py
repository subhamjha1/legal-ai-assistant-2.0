"""
Indexing service (Milestone 3).

Why this exists:
This is the orchestration layer between "we have chunks" and "we can search
them" - it embeds chunk text and pushes vectors + payload into Qdrant. Kept
as its own service (rather than jamming this logic into the API route) so
it's independently testable and so Milestone 5/6 (hybrid search, re-ranking)
can call the same embedding provider for query-time embedding without
duplicating provider-selection logic.

Both the embedding provider and the vector store are expensive to construct
(model load / DB connection), so we cache singletons here rather than
rebuilding them on every request.
"""
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.chunk import ChunkingResult
from app.schemas.document import ParsedDocument
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


@lru_cache
def get_cached_embedding_provider() -> EmbeddingProvider:
    """Cached so the BGE model is loaded into memory once per process, not
    once per request."""
    return get_embedding_provider()


@lru_cache
def get_cached_vector_store() -> VectorStore:
    """Cached so the embedded local Qdrant client (which holds a file lock)
    is opened once per process, not once per request."""
    provider = get_cached_embedding_provider()
    return VectorStore(vector_dimension=provider.dimension)


class IndexingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_provider = get_cached_embedding_provider()
        self.vector_store = get_cached_vector_store()

    def index_document(self, document: ParsedDocument, chunking_result: ChunkingResult) -> int:
        if not chunking_result.chunks:
            logger.warning("Document %s has no chunks to index.", document.document_id)
            return 0

        texts = [chunk.text for chunk in chunking_result.chunks]
        vectors = self.embedding_provider.embed_texts(texts)
        return self.vector_store.index_chunks(document, chunking_result, vectors)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[dict]:
        query_vector = self.embedding_provider.embed_query(query)
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            document_type=document_type,
        )

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
