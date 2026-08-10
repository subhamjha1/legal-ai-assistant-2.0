from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.repository import ChunkRepository
from backend.graph.neo4j_client import get_graph_store
from backend.retrieval.hybrid_retriever import RetrievedChunk
from backend.schemas.document import Chunk

logger = logging.getLogger(__name__)


async def expand_with_graph(session: AsyncSession, retrieved: list[RetrievedChunk], hops: int = 1) -> list[Chunk]:
    graph_store = get_graph_store()
    if not graph_store.enabled or not retrieved:
        return []

    seed_document_ids = list({rc.chunk.document_id for rc in retrieved})
    connected = graph_store.get_connected_documents(seed_document_ids, hops=hops)
    if not connected:
        return []

    chunk_repo = ChunkRepository(session)
    extra_chunks: list[Chunk] = []
    for record in connected:
        doc_id = record["document_id"]
        # first chunk is usually enough context for a linked doc
        neighbors = await chunk_repo.get_neighbors(doc_id, chunk_index=0, window=0)
        if neighbors:
            extra_chunks.append(neighbors[0])

    logger.info(
        "Graph expansion added %d supplementary chunks from %d connected documents",
        len(extra_chunks),
        len(connected),
    )
    return extra_chunks
