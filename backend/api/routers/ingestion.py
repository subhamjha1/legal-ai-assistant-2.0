from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_bm25_index, get_db_session, get_faiss_store
from backend.api.schemas import UploadResponse
from backend.config.settings import settings
from backend.ingestion.indexer import index_results
from backend.ingestion.pipeline import ingest_directory, ingest_single_pdf
from backend.retrieval.bm25_index import Bm25Index
from backend.retrieval.faiss_store import FaissVectorStore

router = APIRouter(tags=["ingestion"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    faiss_store: FaissVectorStore = Depends(get_faiss_store),
    bm25_index: Bm25Index = Depends(get_bm25_index),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    size_limit = settings.max_upload_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > size_limit:
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit")
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    dest = settings.raw_data_dir / file.filename
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_path), str(dest))

    result = ingest_single_pdf(dest)
    await index_results(session, [result], faiss_store, bm25_index)

    return UploadResponse(
        document_id=result.document_metadata.document_id,
        filename=result.document_metadata.filename,
        document_type=result.document_metadata.document_type.value,
        num_child_chunks=len(result.child_chunks),
        num_parent_chunks=len(result.parent_chunks),
        status="indexed",
    )


@router.post("/index")
async def index_corpus(
    session: AsyncSession = Depends(get_db_session),
    faiss_store: FaissVectorStore = Depends(get_faiss_store),
    bm25_index: Bm25Index = Depends(get_bm25_index),
):
    results = ingest_directory(settings.raw_data_dir)
    if not results:
        raise HTTPException(404, f"No PDFs found under {settings.raw_data_dir}")

    await index_results(session, results, faiss_store, bm25_index)

    return {
        "documents_indexed": len(results),
        "total_child_chunks": sum(len(r.child_chunks) for r in results),
        "total_parent_chunks": sum(len(r.parent_chunks) for r in results),
    }
