from backend.ingestion.citation_extractor import extract_citations


def test_extracts_us_supreme_court_citation():
    text = "As established in 410 U.S. 113 (1973), the doctrine applies broadly."
    citations = extract_citations(text)
    raw_texts = [c.raw_text for c in citations]
    assert any("410 U.S. 113" in r for r in raw_texts)


def test_extracts_usc_and_cfr_citations():
    text = "See 26 U.S.C. § 61 for gross income and 26 C.F.R. § 1.61-1 for the regulation."
    citations = extract_citations(text)
    raw_texts = [c.raw_text for c in citations]
    assert any("U.S.C." in r for r in raw_texts)
    assert any("C.F.R." in r for r in raw_texts)


def test_deduplicates_repeated_citations():
    text = "Section 162(a) allows the deduction. Section 162(a) is the operative provision."
    citations = extract_citations(text)
    normalized = [c.normalized for c in citations]
    assert len(normalized) == len(set(normalized))


def test_no_citations_in_plain_text():
    text = "This paragraph discusses general principles without any formal citation."
    citations = extract_citations(text)
    assert citations == []
