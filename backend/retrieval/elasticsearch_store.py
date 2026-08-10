from __future__ import annotations

import logging
from functools import lru_cache

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import settings
from backend.schemas.document import Chunk

logger = logging.getLogger(__name__)

_INDEX_MAPPING = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "text": {"type": "text", "analyzer": "english"},
        "filename": {"type": "keyword"},
        "document_type": {"type": "keyword"},
        "page_start": {"type": "integer"},
        "page_end": {"type": "integer"},
        "section": {"type": "keyword"},
        "heading": {"type": "text"},
        "year": {"type": "integer"},
        "court": {"type": "keyword"},
        "act_name": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "is_parent": {"type": "boolean"},
    }
}


class ElasticsearchStore:
    def __init__(self):
        self.enabled = settings.elasticsearch_enabled
        self.index_name = settings.elasticsearch_index
        self._client: Elasticsearch | None = None
        if self.enabled:
            try:
                self._client = Elasticsearch(settings.elasticsearch_url, request_timeout=10)
                if not self._client.ping():
                    raise ConnectionError("ping failed")
            except Exception:
                logger.warning(
                    "Elasticsearch unreachable at %s — lexical ES search disabled, "
                    "falling back to FAISS + BM25 only",
                    settings.elasticsearch_url,
                )
                self.enabled = False
                self._client = None

    def ensure_index(self) -> None:
        if not self.enabled or self._client is None:
            return
        if not self._client.indices.exists(index=self.index_name):
            self._client.indices.create(index=self.index_name, mappings=_INDEX_MAPPING)
            logger.info("Created Elasticsearch index %s", self.index_name)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        if not self.enabled or self._client is None:
            return
        from elasticsearch.helpers import bulk

        self.ensure_index()
        actions = [
            {
                "_index": self.index_name,
                "_id": c.chunk_id,
                "_source": {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "text": c.text,
                    "filename": c.filename,
                    "document_type": c.document_type.value,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                    "heading": c.heading,
                    "year": c.metadata.get("year"),
                    "court": c.metadata.get("court"),
                    "act_name": c.metadata.get("act_name"),
                    "chunk_index": c.chunk_index,
                    "is_parent": c.is_parent,
                },
            }
            for c in chunks
        ]
        bulk(self._client, actions)
        logger.info("Indexed %d chunks into Elasticsearch", len(chunks))

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        if not self.enabled or self._client is None:
            return []

        must_clauses = [{"multi_match": {"query": query, "fields": ["text^2", "heading^1.5"]}}]
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value is None:
                    continue
                filter_clauses.append({"term": {field: value}})

        body = {
            "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
            "size": top_k,
        }
        try:
            resp = self._client.search(index=self.index_name, **body)
        except NotFoundError:
            return []
        except Exception:
            logger.exception("Elasticsearch query failed — returning empty result set")
            return []

        hits = resp["hits"]["hits"]
        return [(h["_source"]["chunk_id"], float(h["_score"])) for h in hits]


@lru_cache
def get_elasticsearch_store() -> ElasticsearchStore:
    return ElasticsearchStore()
