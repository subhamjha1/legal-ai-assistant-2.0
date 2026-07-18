"""
Tests for evaluation/cli.py (Milestone 9).

Uses the same real-pipeline + PromptEchoLLM fixture pattern as
test_evaluation_runner.py, invoking main() directly (rather than shelling
out to `python -m evaluation.cli`) so these tests stay fast and don't need
a subprocess - but this exercises the exact same code path a real command-
line invocation would.

Forces EMBEDDING_PROVIDER=hash (the offline fallback added in Milestone 8)
so these tests don't attempt a real HuggingFace download for BGE, which
this sandbox's network blocks - keeping the tests fast and deterministic
rather than relying on the per-question error-handling fallback to mask a
slow network timeout on every question.
"""
import json

import pytest

from app.core.config import get_settings
from evaluation.cli import main


@pytest.fixture(autouse=True)
def use_offline_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_cli_test"))
    monkeypatch.setenv("BM25_LOCAL_STORAGE_PATH", str(tmp_path / "bm25_cli_test"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def small_dataset_file(tmp_path):
    dataset = [
        {
            "id": "t1",
            "query": "Which court heard this case?",
            "ground_truth_answer": "The High Court of Delhi at New Delhi heard this case.",
            "source_document": "sample_legal_doc_final.pdf",
            "relevant_pages": [1],
            "expected_citations": [{"document": "sample_legal_doc_final.pdf", "page_start": 1, "page_end": 1}],
            "category": "fact_lookup",
        },
        {
            "id": "t2",
            "query": "What penalty was imposed?",
            "ground_truth_answer": "I could not find supporting evidence.",
            "source_document": "sample_legal_doc_final.pdf",
            "category": "no_evidence_check",
        },
    ]
    path = tmp_path / "mini_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


class TestCLI:
    def test_cli_runs_and_writes_reports(self, small_dataset_file, tmp_path, monkeypatch):
        # Patch get_llm_provider used inside qa_service so main() doesn't
        # need a real API key - same technique used throughout this suite.
        from app.services import llm_provider as llmmod
        from app.services import qa_service as qamod
        from tests.test_qa_service import FakeLLMProvider

        fake = FakeLLMProvider(fixed_response="The High Court of Delhi heard this matter [C1].")
        monkeypatch.setattr(qamod, "get_llm_provider", lambda *a, **kw: fake)
        monkeypatch.setattr(llmmod, "get_llm_provider", lambda *a, **kw: fake)

        output_dir = tmp_path / "results"
        exit_code = main(["--dataset", str(small_dataset_file), "--output-dir", str(output_dir)])

        assert exit_code == 0
        assert (output_dir / "report.md").exists()
        assert (output_dir / "report.html").exists()
        assert (output_dir / "results.json").exists()

        results = json.loads((output_dir / "results.json").read_text())
        assert results["total_questions"] == 2

    def test_cli_returns_error_for_missing_dataset(self, tmp_path):
        exit_code = main(["--dataset", str(tmp_path / "nonexistent.json"), "--output-dir", str(tmp_path / "out")])
        assert exit_code == 1

    def test_ci_gate_fails_when_threshold_not_met(self, small_dataset_file, tmp_path, monkeypatch):
        from app.services import llm_provider as llmmod
        from app.services import qa_service as qamod
        from tests.test_qa_service import FakeLLMProvider

        # A response with zero term overlap with either ground truth answer
        # guarantees a low answer_correctness score, so an aggressive
        # threshold will legitimately fail the gate.
        fake = FakeLLMProvider(fixed_response="Completely unrelated text with no citation tag at all.")
        monkeypatch.setattr(qamod, "get_llm_provider", lambda *a, **kw: fake)
        monkeypatch.setattr(llmmod, "get_llm_provider", lambda *a, **kw: fake)

        exit_code = main(
            [
                "--dataset", str(small_dataset_file),
                "--output-dir", str(tmp_path / "results"),
                "--min-answer-correctness", "0.99",
            ]
        )
        assert exit_code == 1

    def test_ci_gate_passes_when_no_threshold_set(self, small_dataset_file, tmp_path, monkeypatch):
        from app.services import llm_provider as llmmod
        from app.services import qa_service as qamod
        from tests.test_qa_service import FakeLLMProvider

        fake = FakeLLMProvider(fixed_response="Some answer with no citation.")
        monkeypatch.setattr(qamod, "get_llm_provider", lambda *a, **kw: fake)
        monkeypatch.setattr(llmmod, "get_llm_provider", lambda *a, **kw: fake)

        # Default thresholds are 0.0 (no gate) - should always pass regardless of quality.
        exit_code = main(["--dataset", str(small_dataset_file), "--output-dir", str(tmp_path / "results")])
        assert exit_code == 0
