from __future__ import annotations

import logging
from functools import lru_cache

from neo4j import GraphDatabase

from backend.config.settings import settings
from backend.schemas.document import DocumentMetadata, GraphRelation

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    def __init__(self):
        self.enabled = settings.graph_rag_enabled
        self._driver = None
        if self.enabled:
            try:
                self._driver = GraphDatabase.driver(
                    settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
                )
                self._driver.verify_connectivity()
            except Exception:
                logger.warning(
                    "Neo4j unreachable at %s — GraphRAG disabled, hybrid retrieval will proceed without graph expansion",
                    settings.neo4j_uri,
                )
                self.enabled = False
                self._driver = None

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def ensure_constraints(self) -> None:
        if not self.enabled:
            return
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
            )

    def upsert_document_node(self, meta: DocumentMetadata) -> None:
        if not self.enabled:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.title = $title,
                    d.document_type = $document_type,
                    d.filename = $filename,
                    d.year = $year,
                    d.court = $court,
                    d.act_name = $act_name
                """,
                document_id=meta.document_id,
                title=meta.title,
                document_type=meta.document_type.value,
                filename=meta.filename,
                year=meta.year,
                court=meta.court,
                act_name=meta.act_name,
            )

    def upsert_relation(self, rel: GraphRelation) -> None:
        if not self.enabled:
            return
        rel_type = rel.relation_type.value.upper()
        with self._driver.session() as session:
            session.run(
                f"""
                MATCH (a:Document {{document_id: $source}})
                MATCH (b:Document {{document_id: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.evidence_chunk_id = $evidence_chunk_id,
                    r.confidence = $confidence
                """,
                source=rel.source_document_id,
                target=rel.target_document_id,
                evidence_chunk_id=rel.evidence_chunk_id,
                confidence=rel.confidence,
            )

    def get_connected_documents(self, document_ids: list[str], hops: int = 1, limit: int = 20) -> list[dict]:
        if not self.enabled or not document_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (seed:Document)-[r*1..{hops}]-(connected:Document)
                WHERE seed.document_id IN $document_ids
                  AND NOT connected.document_id IN $document_ids
                RETURN DISTINCT connected.document_id AS document_id,
                       connected.title AS title,
                       connected.document_type AS document_type,
                       [rel in r | type(rel)] AS relation_path
                LIMIT $limit
                """,
                document_ids=document_ids,
                limit=limit,
            )
            return [dict(record) for record in result]


@lru_cache
def get_graph_store() -> Neo4jGraphStore:
    return Neo4jGraphStore()
