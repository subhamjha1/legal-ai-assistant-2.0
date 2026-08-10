from __future__ import annotations

import json
import logging

from backend.config.settings import settings
from backend.llm.client import get_llm_client

logger = logging.getLogger(__name__)

_EXPANSION_PROMPT = """You are a US legal research assistant. Given a user's legal/tax \
question, generate alternative search terms that would help find relevant \
statutes, case law, or IRS regulations: synonyms, statutory abbreviations \
(e.g. CGT for capital gains tax), likely U.S.C./I.R.C. section numbers if \
you know them, and related legal terms of art.

Return ONLY a JSON array of strings, 4-8 terms, no explanation.

Question: {query}"""

_MULTI_QUERY_PROMPT = """You are a US legal research assistant. Rewrite the \
following question as {n} distinct search queries that approach it from \
different angles a legal researcher might use (e.g. statutory language, \
case-law framing, practical/plain-English framing, IRS-regulation framing). \
Each should be a complete, standalone question.

Return ONLY a JSON array of {n} strings, no explanation.

Question: {query}"""


def _safe_json_list(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON list output: %r", raw[:200])
    return []


def expand_query(query: str) -> list[str]:
    if not settings.enable_query_expansion:
        return []
    client = get_llm_client()
    try:
        reply = client.complete(_EXPANSION_PROMPT.format(query=query), max_tokens=200, temperature=0.2)
        terms = _safe_json_list(reply)
        logger.info("Query expansion for %r -> %s", query, terms)
        return terms
    except Exception:
        logger.exception("Query expansion failed — continuing without it")
        return []


def generate_multi_queries(query: str, n: int | None = None) -> list[str]:
    if not settings.enable_multi_query:
        return []
    n = n or settings.multi_query_count
    client = get_llm_client()
    try:
        reply = client.complete(_MULTI_QUERY_PROMPT.format(query=query, n=n), max_tokens=300, temperature=0.3)
        reformulations = _safe_json_list(reply)
        logger.info("Multi-query for %r -> %s", query, reformulations)
        return reformulations[:n]
    except Exception:
        logger.exception("Multi-query generation failed — continuing with original query only")
        return []


def build_lexical_query_string(original_query: str, expansion_terms: list[str]) -> str:
    if not expansion_terms:
        return original_query
    return original_query + " " + " ".join(expansion_terms)
