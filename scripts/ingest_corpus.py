from __future__ import annotations

import argparse
import asyncio
import logging

from backend.config.settings import settings
from backend.database.session import SessionLocal, init_db
from backend.ingestion.indexer import index_results
from backend.ingestion.pipeline import ingest_directory
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.embeddings import get_embedding_model
from backend.retrieval.faiss_store import FaissVectorStore

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(path: str) -> None:
    await init_db()

    embedder = get_embedding_model()
    faiss_store = FaissVectorStore.load(dimension=embedder.dimension)
    bm25_index = Bm25Index.load()

    results = ingest_directory(path)
    logger.info("Ingested %d documents from %s", len(results), path)
    if not results:
        return

    async with SessionLocal() as session:
        await index_results(session, results, faiss_store, bm25_index)
        await session.commit()

    total_chunks = sum(len(r.child_chunks) for r in results)
    logger.info("Indexing complete: %d documents, %d child chunks", len(results), total_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(settings.raw_data_dir))
    args = parser.parse_args()
    asyncio.run(main(args.path))
