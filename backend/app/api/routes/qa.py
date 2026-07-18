"""
Q&A API route (Milestone 7).
"""
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.qa import AnswerRequest, AnswerResponse
from app.services.qa_service import QAService

router = APIRouter(tags=["qa"])


def get_qa_service() -> QAService:
    return QAService()


@router.post("/query", response_model=AnswerResponse)
async def query(
    request: AnswerRequest,
    service: QAService = Depends(get_qa_service),
) -> AnswerResponse:
    return service.answer(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        document_type=request.document_type.value if request.document_type else None,
    )


@router.post("/query/stream")
async def query_stream(
    request: AnswerRequest,
    service: QAService = Depends(get_qa_service),
) -> StreamingResponse:
    """
    Server-Sent Events stream of the answer as it's generated. Each event is
    a JSON-encoded line: `data: {...}\\n\\n`, matching the two event shapes
    QAService.answer_stream() yields (`token` deltas, then one final `done`
    event with citations). A plain fetch()+ReadableStream on the frontend
    (rather than the browser's EventSource API) is used to consume this,
    since EventSource only supports GET requests and this needs a POST body.
    """
    def event_generator():
        for event in service.answer_stream(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            document_type=request.document_type.value if request.document_type else None,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

