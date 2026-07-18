# Evaluation Report

- **Run started:** 2026-07-17T10:18:09.300937+00:00
- **Run finished:** 2026-07-17T10:18:09.379123+00:00
- **Model:** DemoLLM
- **Total questions:** 18
- **Errored questions:** 0

## Aggregate Metrics

| Metric | Value |
|---|---|
| avg_answer_correctness | 0.2576 |
| avg_citation_precision | 0.6667 |
| avg_citation_recall | 0.7407 |
| avg_exact_match | 0.0000 |
| avg_faithfulness | 0.1593 |
| avg_hallucinated | 1.0000 |
| avg_latency_seconds | 0.0043 |
| avg_mrr | 0.9167 |
| avg_ndcg_at_5 | 0.9420 |
| avg_retrieval_recall_at_5 | 1.0000 |
| errored_questions | 0.0000 |
| questions_with_token_usage | 0.0000 |
| total_input_tokens | 0.0000 |
| total_output_tokens | 0.0000 |
| total_questions | 18.0000 |

## Per-Question Results

| ID | Category | Answer Correctness | Citation P | Citation R | MRR | Faithfulness | Latency (s) | Error |
|---|---|---|---|---|---|---|---|---|
| q001 | fact_lookup | 0.27 | 1.00 | 1.00 | 1.00 | 0.20 | 0.01 |  |
| q002 | fact_lookup | 0.08 | 0.00 | 0.00 | 0.50 | 0.20 | 0.00 |  |
| q003 | fact_lookup | 0.08 | 0.00 | 0.00 | 0.50 | 0.20 | 0.00 |  |
| q004 | fact_lookup | 0.17 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |  |
| q005 | fact_lookup | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |  |
| q006 | fact_lookup | 0.42 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |  |
| q007 | holding | 0.38 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |  |
| q008 | fact_lookup | 0.29 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 |  |
| q009 | fact_lookup | 0.24 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 |  |
| q010 | statutory_requirement | 0.33 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 |  |
| q011 | fact_lookup | 0.55 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 |  |
| q012 | statutory_requirement | 0.61 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |  |
| q013 | holding | 0.27 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |  |
| q014 | holding | 0.41 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |  |
| q015 | synthesis | 0.20 | 0.00 | 0.00 | 0.50 | 0.20 | 0.00 |  |
| q016 | synthesis | 0.34 | 1.00 | 0.33 | 1.00 | 0.20 | 0.00 |  |
| q017 | no_evidence_check | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |  |
| q018 | no_evidence_check | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |  |

## Notes on Metric Rigor

- **Answer correctness** uses token-F1 (SQuAD-style), a deterministic stand-in for an LLM-judge or semantic-similarity metric. See `evaluation/metrics.py` docstrings for the full rationale.
- **Faithfulness** checks that each answer sentence carries a citation tag - it does NOT verify the citation's claim is actually entailed by the source text (that would need an LLM judge, as in RAGAS's real faithfulness metric).
- Citation precision/recall use page-**range overlap** matching against the golden dataset's `expected_citations`, not exact equality.