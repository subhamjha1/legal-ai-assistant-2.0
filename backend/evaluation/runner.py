from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database.repository import DocumentRepository
from backend.evaluation.golden_dataset import GoldenRow, load_golden_dataset
from backend.evaluation.metrics import aggregate_metrics
from backend.evaluation.ragas_eval import evaluate_with_ragas
from backend.evaluation.report_generator import generate_markdown_report, plot_confusion_matrix, plot_metrics_bar_chart
from backend.graph.graph_retriever import expand_with_graph
from backend.llm.answer_generator import generate_answer
from backend.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


def _row_is_hit(chunk_filename: str, chunk_page_start: int, chunk_page_end: int, row: GoldenRow) -> bool:
    return chunk_filename == row.document and chunk_page_start <= row.page <= chunk_page_end


async def run_full_evaluation(session: AsyncSession, retriever: HybridRetriever, run_generation: bool = True) -> dict:
    golden_rows = load_golden_dataset()
    doc_repo = DocumentRepository(session)

    per_query_hits: list[list[bool]] = []
    generation_questions, generation_answers, generation_contexts, generation_ground_truths = [], [], [], []
    predicted_types, golden_types = [], []

    all_documents = {d.filename: d for d in await doc_repo.list_all()}

    for row in golden_rows:
        retrieved = await retriever.retrieve(session, row.query, top_k=max(settings.eval_recall_k_values))
        hits = [_row_is_hit(rc.chunk.filename, rc.chunk.page_start, rc.chunk.page_end, row) for rc in retrieved]
        per_query_hits.append(hits)

        if retrieved:
            predicted_types.append(retrieved[0].chunk.document_type.value)
        golden_doc = all_documents.get(row.document)
        golden_types.append(golden_doc.document_type.value if golden_doc else "other")

        if run_generation and retrieved:
            extra_graph_chunks = await expand_with_graph(session, retrieved)
            result = generate_answer(row.query, retrieved, extra_graph_chunks)
            generation_questions.append(row.query)
            generation_answers.append(result.get("answer", ""))
            generation_contexts.append([rc.chunk.text for rc in retrieved])
            generation_ground_truths.append(row.ground_truth)

    retrieval_metrics = aggregate_metrics(per_query_hits, settings.eval_recall_k_values)

    generation_metrics = {}
    if run_generation and generation_questions:
        generation_metrics = evaluate_with_ragas(
            generation_questions, generation_answers, generation_contexts, generation_ground_truths
        )

    settings.eval_report_dir.mkdir(parents=True, exist_ok=True)
    chart_path = plot_metrics_bar_chart(
        retrieval_metrics, "Retrieval Metrics", settings.eval_report_dir / "retrieval_metrics.png"
    )
    document_types = sorted(set(predicted_types) | set(golden_types)) or ["other"]
    confusion_path = plot_confusion_matrix(
        document_types,
        predicted_types or ["other"],
        golden_types or ["other"],
        settings.eval_report_dir / "confusion_matrix.png",
    )
    report_path = generate_markdown_report(
        retrieval_metrics, generation_metrics, chart_path, confusion_path, len(golden_rows)
    )

    return {
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "report_path": str(report_path),
        "num_golden_queries": len(golden_rows),
    }
