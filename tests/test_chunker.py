from backend.ingestion.chunker import _recursive_split, count_tokens


def test_short_text_not_split():
    text = "This is a short paragraph that fits well within the token budget."
    result = _recursive_split(text, target_tokens=1000, overlap_tokens=100)
    assert result == [text]


def test_long_text_is_split_into_multiple_pieces():
    paragraph = "This is a sentence about tax law. " * 100
    result = _recursive_split(paragraph, target_tokens=200, overlap_tokens=30)
    assert len(result) > 1
    for piece in result:
        assert count_tokens(piece) <= 220


def test_split_preserves_overlap_between_consecutive_chunks():
    paragraphs = "\n\n".join(
        [f"Paragraph number {i} discusses a distinct legal issue in some detail." for i in range(30)]
    )
    result = _recursive_split(paragraphs, target_tokens=100, overlap_tokens=20)
    assert len(result) > 1
    first_words_next = set(result[1].split()[:10])
    last_words_prev = set(result[0].split()[-10:])
    assert first_words_next & last_words_prev


def test_count_tokens_matches_expected_order_of_magnitude():
    text = "word " * 100
    tokens = count_tokens(text)
    assert 50 <= tokens <= 200
