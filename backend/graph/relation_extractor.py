from __future__ import annotations

import logging
import re

from backend.schemas.document import Chunk, DocumentMetadata, DocumentType, GraphRelation, RelationType

logger = logging.getLogger(__name__)

_AMEND_KEYWORDS = re.compile(r"\bamend(?:s|ed|ing|ment)?\b", re.IGNORECASE)
_OVERRULE_KEYWORDS = re.compile(r"\boverrul(?:e|es|ed|ing)\b", re.IGNORECASE)
_IMPLEMENT_KEYWORDS = re.compile(r"\bimplement(?:s|ed|ing|ation)?\b", re.IGNORECASE)


def _infer_relation_type(chunk: Chunk, source_type: DocumentType, target_type: DocumentType) -> RelationType:
    window = chunk.text
    if _AMEND_KEYWORDS.search(window) and source_type == DocumentType.ACT and target_type == DocumentType.ACT:
        return RelationType.AMENDS
    if _OVERRULE_KEYWORDS.search(window) and source_type == DocumentType.JUDGMENT:
        return RelationType.OVERRULES
    if _IMPLEMENT_KEYWORDS.search(window) and source_type == DocumentType.IRS_REGULATION:
        return RelationType.IMPLEMENTS
    if source_type == DocumentType.COMMENTARY:
        return RelationType.REFERENCES
    return RelationType.CITES


def build_citation_index(all_documents: list[DocumentMetadata]) -> dict[str, str]:
    index: dict[str, str] = {}
    for doc in all_documents:
        for key in filter(None, [doc.citation_string, doc.act_name, doc.title, doc.filename]):
            index[key.strip().lower()] = doc.document_id
    return index


def extract_relations_deterministic(
    chunks: list[Chunk],
    doc_meta_by_id: dict[str, DocumentMetadata],
    citation_index: dict[str, str],
) -> list[GraphRelation]:
    relations: list[GraphRelation] = []

    for chunk in chunks:
        source_meta = doc_meta_by_id.get(chunk.document_id)
        if source_meta is None:
            continue

        for citation in chunk.citations:
            key = (citation.normalized or citation.raw_text).strip().lower()
            target_id = citation_index.get(key)
            if not target_id or target_id == chunk.document_id:
                continue

            target_meta = doc_meta_by_id.get(target_id)
            if target_meta is None:
                continue

            relation_type = _infer_relation_type(chunk, source_meta.document_type, target_meta.document_type)
            relations.append(
                GraphRelation(
                    source_document_id=chunk.document_id,
                    target_document_id=target_id,
                    relation_type=relation_type,
                    evidence_chunk_id=chunk.chunk_id,
                    confidence=0.8,
                )
            )

    logger.info("Extracted %d deterministic graph relations from %d chunks", len(relations), len(chunks))
    return relations
