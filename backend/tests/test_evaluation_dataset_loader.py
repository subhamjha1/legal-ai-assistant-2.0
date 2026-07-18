"""
Tests for evaluation/dataset_loader.py (Milestone 9).
"""
import pytest

from evaluation.dataset_loader import load_dataset, save_dataset_json
from evaluation.schema import ExpectedCitation, GoldenQuestion


def test_loads_the_real_golden_dataset():
    """The actual shipped dataset must load and validate cleanly."""
    questions = load_dataset("evaluation/golden_dataset.json")
    assert len(questions) == 18
    assert all(isinstance(q, GoldenQuestion) for q in questions)


def test_golden_dataset_has_expected_categories():
    questions = load_dataset("evaluation/golden_dataset.json")
    categories = {q.category for q in questions}
    assert "no_evidence_check" in categories
    assert "synthesis" in categories
    assert "fact_lookup" in categories


def test_no_evidence_questions_have_no_expected_citations():
    questions = load_dataset("evaluation/golden_dataset.json")
    no_evidence_questions = [q for q in questions if q.category == "no_evidence_check"]
    assert len(no_evidence_questions) >= 2
    for q in no_evidence_questions:
        assert q.expected_citations == []
        assert q.expects_no_evidence is True


def test_missing_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_dataset("evaluation/does_not_exist.json")


def test_unsupported_extension_raises_value_error(tmp_path):
    bad_file = tmp_path / "dataset.txt"
    bad_file.write_text("not a real dataset")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(bad_file)


def test_json_round_trip_preserves_data(tmp_path):
    questions = [
        GoldenQuestion(
            id="t1",
            query="Test query?",
            ground_truth_answer="Test answer.",
            source_document="test.pdf",
            relevant_pages=[1, 2],
            expected_citations=[ExpectedCitation(document="test.pdf", page_start=1, page_end=2)],
            category="fact_lookup",
        )
    ]
    path = tmp_path / "roundtrip.json"
    save_dataset_json(questions, path)

    reloaded = load_dataset(path)
    assert len(reloaded) == 1
    assert reloaded[0].id == "t1"
    assert reloaded[0].expected_citations[0].page_start == 1


def test_csv_loading_produces_valid_questions(tmp_path):
    csv_content = (
        "id,query,ground_truth_answer,source_document,relevant_pages,category,notes\n"
        'c1,"What is the case number?","The case number is 1234/2024.",sample.pdf,1,fact_lookup,\n'
        'c2,"What penalty was imposed?","I could not find supporting evidence.",sample.pdf,,no_evidence_check,No penalty mentioned\n'
    )
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    questions = load_dataset(csv_path)
    assert len(questions) == 2
    assert questions[0].relevant_pages == [1]
    assert questions[0].expected_citations[0].document == "sample.pdf"
    assert questions[1].relevant_pages == []
    assert questions[1].expected_citations == []
    assert questions[1].expects_no_evidence is True


def test_csv_with_multiple_pages_produces_spanning_citation(tmp_path):
    csv_content = (
        "id,query,ground_truth_answer,source_document,relevant_pages,category,notes\n"
        'c1,"Summarize the reasoning.","Some synthesis answer.",sample.pdf,1;2;3,synthesis,\n'
    )
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    questions = load_dataset(csv_path)
    assert questions[0].relevant_pages == [1, 2, 3]
    assert questions[0].expected_citations[0].page_start == 1
    assert questions[0].expected_citations[0].page_end == 3
