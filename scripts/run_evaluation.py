from __future__ import annotations

import argparse
import asyncio
import json
import logging

from backend.database.session import SessionLocal
from backend.evaluation.runner import run_full_evaluation
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.embeddings import get_embedding_model
from backend.retrieval.faiss_store import FaissVectorStore
from backend.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(run_generation: bool) -> None:
    embedder = get_embedding_model()
    faiss_store = FaissVectorStore.load(dimension=embedder.dimension)
    bm25_index = Bm25Index.load()
    retriever = HybridRetriever(faiss_store, bm25_index)

    async with SessionLocal() as session:
        results = await run_full_evaluation(session, retriever, run_generation=run_generation)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-generation", action="store_true", help="Skip RAGAS/DeepEval generation metrics")
    args = parser.parse_args()
    asyncio.run(main(run_generation=not args.no_generation))
