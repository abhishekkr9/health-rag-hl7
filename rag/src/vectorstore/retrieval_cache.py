"""Redis-backed retrieval cache — avoids re-querying Weaviate for identical questions."""

import hashlib
import json
import logging

import redis
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import (
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_RETRIEVAL_PREFIX,
    REDIS_RETRIEVAL_TTL,
)

logger = logging.getLogger(__name__)


def _make_client() -> redis.Redis | None:
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        return client
    except Exception as exc:
        logger.warning("Redis unavailable — retrieval cache disabled: %s", exc)
        return None


def _cache_key(question: str) -> str:
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"{REDIS_RETRIEVAL_PREFIX}{digest}"


def _serialize(docs: list[Document]) -> str:
    return json.dumps([
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ])


def _deserialize(raw: str) -> list[Document]:
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in json.loads(raw)
    ]


class CachedRetriever:
    """
    Wraps a LangChain retriever with a Redis cache.

    Cache key:  SHA-256(lower(question))[:16]
    TTL:        REDIS_RETRIEVAL_TTL seconds (default 5 min)
    Fallback:   calls underlying retriever directly if Redis is down
    """

    def __init__(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever
        self._client    = _make_client()
        if self._client:
            logger.info("Retrieval cache: Redis  (TTL=%ds)", REDIS_RETRIEVAL_TTL)
        else:
            logger.info("Retrieval cache: disabled (Redis unavailable)")

    def invoke(self, question: str) -> list[Document]:
        if not self._client:
            return self._retriever.invoke(question)

        key = _cache_key(question)
        try:
            cached = self._client.get(key)
            if cached:
                logger.debug("Retrieval cache HIT  key=%s", key)
                return _deserialize(cached)
        except Exception as exc:
            logger.warning("Redis get failed: %s", exc)

        docs = self._retriever.invoke(question)

        try:
            self._client.set(key, _serialize(docs), ex=REDIS_RETRIEVAL_TTL)
            logger.debug("Retrieval cache MISS key=%s — cached %d docs", key, len(docs))
        except Exception as exc:
            logger.warning("Redis set failed: %s", exc)

        return docs
