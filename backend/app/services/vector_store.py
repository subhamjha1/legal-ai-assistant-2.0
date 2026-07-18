"""
Qdrant vector store service (Milestone 3).

Why this exists:
This is the single point of contact with Qdrant. Everything a search result
needs for a citation (document, page range, structural label, chunk text) is
stored as point *payload* alongside the vector, so a search hit converts
directly into a citation without a second database round-trip - important
for both latency and for keeping "what did we cite" and "what did we search"
guaranteed consistent (no risk of the two falling out of sync).

Supports two Qdrant deployment modes:
- "local": embedded, file-backed Qdrant (no server process needed) - good
  for local dev/demo, single-process only.
- "server": a real Qdrant instance over gRPC/HTTP, used with docker-compose
  in Milestone 10.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.chunk import Chunk, ChunkingResult
from app.schemas.document import ParsedDocument

logger = get_logger(__name__)


class VectorStore:
    """Wraps Qdrant collection lifecycle, indexing, and search."""

    def __init__(self, vector_dimension: int) -> None:
        self.settings = get_settings()
        self.vector_dimension = vector_dimension
        self.collection_name = self.settings.qdrant_collection_name
        self._client = self._build_client()
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Client / collection setup
    # ------------------------------------------------------------------ #
    def _build_client(self) -> QdrantClient:
        if self.settings.qdrant_mode == "local":
            self.settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
            logger.info("Using embedded local Qdrant at %s", self.settings.qdrant_local_path)
            return QdrantClient(path=str(self.settings.qdrant_local_path))
        logger.info("Connecting to Qdrant server at %s:%s", self.settings.qdrant_host, self.settings.qdrant_port)
        return QdrantClient(host=self.settings.qdrant_host, port=self.settings.qdrant_port)

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            return
        logger.info("Creating Qdrant collection '%s' (dim=%d, cosine)", self.collection_name, self.vector_dimension)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        # Payload indexes for the filters we actually use at query time
        # (document_type, document_id) - keeps filtered search fast as the
        # collection grows.
        for field in ("document_id", "document_type"):
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_chunks(
        self,
        document: ParsedDocument,
        chunking_result: ChunkingResult,
        vectors: list[list[float]],
    ) -> int:
        """Upsert one Qdrant point per chunk. `vectors` must be aligned
        index-for-index with `chunking_result.chunks`."""
        if len(vectors) != len(chunking_result.chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) must align 1:1 with chunks ({len(chunking_result.chunks)})"
            )

        points = [
            qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=self._build_payload(document, chunk),
            )
            for chunk, vector in zip(chunking_result.chunks, vectors)
        ]

        self._client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info("Indexed %d chunks for document %s into Qdrant", len(points), document.document_id)
        return len(points)

    @staticmethod
    def _build_payload(document: ParsedDocument, chunk: Chunk) -> dict:
        """Everything the citation formatter (Milestone 7) will need,
        denormalized onto the point so a search hit needs no further lookup."""
        return {
            "document_id": document.document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "structural_label": chunk.structural_label,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "split_reason": chunk.split_reason.value,
            "original_filename": document.metadata.original_filename,
            "document_type": document.metadata.document_type.value,
        }

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[dict]:
        """Vector similarity search, optionally filtered by document_id or
        document_type. Returns payload dicts augmented with a `score` field."""
        top_k = top_k or self.settings.vector_search_top_k

        query_filter = None
        conditions = []
        if document_id:
            conditions.append(qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id)))
        if document_type:
            conditions.append(qmodels.FieldCondition(key="document_type", match=qmodels.MatchValue(value=document_type)))
        if conditions:
            query_filter = qmodels.Filter(must=conditions)

        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        hits = []
        for point in results:
            hit = dict(point.payload)
            hit["score"] = point.score
            hits.append(hit)
        return hits

    # ------------------------------------------------------------------ #
    # Deletion
    # ------------------------------------------------------------------ #
    def delete_document(self, document_id: str) -> None:
        """Remove every point belonging to a document (used when a document
        is deleted, so the index doesn't retain orphaned vectors)."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )
        logger.info("Deleted all vectors for document %s from Qdrant", document_id)

    def count(self) -> int:
        return self._client.count(collection_name=self.collection_name, exact=True).count
