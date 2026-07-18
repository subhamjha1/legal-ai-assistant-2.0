"""
QA service (Milestone 7): the full grounded question-answering pipeline.

Pipeline:
    retrieval (Milestone 6: hybrid search + MMR + rerank + confidence filter)
        -> if zero chunks retrieved: return the no-evidence answer
           immediately, WITHOUT calling the LLM at all (deterministic,
           free, and impossible for the model to get wrong by guessing)
        -> otherwise: build a citation-grounded prompt from the retrieved
           chunks, call the LLM, and run its answer through the citation
           formatter (which trusts only our own retrieval metadata, never
           the model's own claims about page numbers)
"""
from typing import Iterator

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.qa import AnswerResponse
from app.services.citation_formatter import extract_citations, is_no_evidence_response
from app.services.llm_provider import LLMProvider, get_llm_provider
from app.services.prompts import build_system_prompt, build_user_message
from app.services.retriever import RetrievalService

logger = get_logger(__name__)


class QAService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.settings = get_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider or get_llm_provider()

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> AnswerResponse:
        chunks, candidates_considered, _dropped = self.retrieval_service.retrieve(
            query, top_k=top_k, document_id=document_id, document_type=document_type
        )

        if not chunks:
            logger.info("No chunks retrieved for query '%s...'; skipping LLM call entirely.", query[:50])
            return AnswerResponse(
                query=query,
                answer=self.settings.no_evidence_phrase,
                citations=[],
                has_sufficient_evidence=False,
                chunks_considered=candidates_considered,
                model_used="none (no evidence retrieved)",
            )

        system_prompt = build_system_prompt()
        user_message = build_user_message(query, chunks)
        raw_answer, token_usage = self.llm_provider.generate_with_usage(system_prompt, user_message)

        no_evidence = is_no_evidence_response(raw_answer)
        citations = [] if no_evidence else extract_citations(raw_answer, chunks)

        return AnswerResponse(
            query=query,
            answer=raw_answer.strip(),
            citations=citations,
            has_sufficient_evidence=not no_evidence and len(citations) > 0,
            chunks_considered=candidates_considered,
            model_used=self.settings.llm_provider,
            token_usage=token_usage,
        )

    def answer_stream(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> Iterator[dict]:
        """
        Streaming counterpart to answer(). Yields dicts of two shapes:
            {"type": "token", "text": "..."}          - one per text delta
            {"type": "done", "citations": [...], ...} - exactly once, last

        Retrieval itself is not streamed (it's fast relative to LLM
        generation and citations can only be finalized once the full
        answer text is known anyway), so the caller sees an initial pause
        for retrieval, then a steady stream of tokens, then one final
        "done" event carrying the citation-formatted result.
        """
        chunks, candidates_considered, _dropped = self.retrieval_service.retrieve(
            query, top_k=top_k, document_id=document_id, document_type=document_type
        )

        if not chunks:
            logger.info("No chunks retrieved for query '%s...'; skipping LLM call entirely.", query[:50])
            yield {"type": "token", "text": self.settings.no_evidence_phrase}
            yield {
                "type": "done",
                "citations": [],
                "has_sufficient_evidence": False,
                "chunks_considered": candidates_considered,
                "model_used": "none (no evidence retrieved)",
            }
            return

        system_prompt = build_system_prompt()
        user_message = build_user_message(query, chunks)

        full_text_parts: list[str] = []
        for delta in self.llm_provider.generate_stream(system_prompt, user_message):
            full_text_parts.append(delta)
            yield {"type": "token", "text": delta}

        raw_answer = "".join(full_text_parts)
        no_evidence = is_no_evidence_response(raw_answer)
        citations = [] if no_evidence else extract_citations(raw_answer, chunks)

        yield {
            "type": "done",
            "citations": [c.model_dump() for c in citations],
            "has_sufficient_evidence": not no_evidence and len(citations) > 0,
            "chunks_considered": candidates_considered,
            "model_used": self.settings.llm_provider,
        }
