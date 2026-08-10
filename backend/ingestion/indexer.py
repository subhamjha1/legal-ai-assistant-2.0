from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.repository import ChunkRepository, DocumentRepository
from backend.graph.neo4j_client import get_graph_store
from backend.graph.relation_extractor import build_citation_index, extract_relations_deterministic
from backend.ingestion.pipeline import IngestionResult
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.elasticsearch_store import get_elasticsearch_store
from backend.retrieval.embeddings import get_embedding_model
from backend.retrieval.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


async def index_results(
    session: AsyncSession,
    results: list[IngestionResult],
    faiss_store: FaissVectorStore,
    bm25_index: Bm25Index,
) -> None:
    if not results:
        return

    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)

    all_child_chunks = []
    all_chunks_for_pg = []
    for r in results:
        await doc_repo.upsert(r.document_metadata)
        all_child_chunks.extend(r.child_chunks)
        all_chunks_for_pg.extend(r.all_chunks)

    await chunk_repo.bulk_upsert(all_chunks_for_pg)
    logger.info("Upserted %d documents / %d chunks into Postgres", len(results), len(all_chunks_for_pg))

    if all_child_chunks:
        embedder = get_embedding_model()
        texts = [c.text for c in all_child_chunks]
        vectors = embedder.embed_passages(texts)
        faiss_store.add([c.chunk_id for c in all_child_chunks], vectors)
        faiss_store.save()

    all_documents = await doc_repo.list_all()
    all_children = []
    for doc in all_documents:
        chunks_in_doc = await chunk_repo.get_all_children(doc.document_id)
        all_children.extend(chunks_in_doc)
    if all_children:
        bm25_index.build([c.chunk_id for c in all_children], [c.text for c in all_children])
        bm25_index.save()

    es_store = get_elasticsearch_store()
    if all_child_chunks:
        es_store.index_chunks(all_child_chunks)

    graph_store = get_graph_store()
    if graph_store.enabled:
        graph_store.ensure_constraints()
        for r in results:
            graph_store.upsert_document_node(r.document_metadata)

        doc_meta_by_id = {d.document_id: d for d in all_documents}
        citation_index = build_citation_index(all_documents)
        relations = extract_relations_deterministic(all_child_chunks, doc_meta_by_id, citation_index)
        for rel in relations:
            graph_store.upsert_relation(rel)
        logger.info("Wrote %d relations to Neo4j", len(relations))
