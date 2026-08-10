from __future__ import annotations

import logging

from backend.config.settings import settings

logger = logging.getLogger(__name__)


def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        logger.warning("ragas / datasets / langchain_openai not installed — skipping RAGAS evaluation")
        return {}

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    judge_llm = ChatOpenAI(
        model=settings.llm_model_name,
        base_url=settings.llm_api_base,
        api_key=settings.resolved_llm_api_key or "not-set",
        temperature=0.0,
    )

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
        )
        return {k: round(float(v), 4) for k, v in result.items()}
    except Exception:
        logger.exception("RAGAS evaluation failed")
        return {}


def evaluate_faithfulness_with_deepeval(question: str, answer: str, contexts: list[str]) -> float | None:
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning("deepeval not installed — skipping DeepEval cross-check")
        return None

    metric = FaithfulnessMetric(threshold=0.7, model=settings.llm_model_name)
    test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
    try:
        metric.measure(test_case)
        return round(float(metric.score), 4)
    except Exception:
        logger.exception("DeepEval faithfulness check failed")
        return None
