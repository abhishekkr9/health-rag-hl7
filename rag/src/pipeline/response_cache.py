"""Redis-backed LLM response cache — skips Groq API for identical inputs."""

import hashlib
import json
import logging

import redis
from langchain_core.documents import Document

from src.config import (
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_RESPONSE_PREFIX,
    REDIS_RESPONSE_TTL,
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
        logger.warning("Redis unavailable — response cache disabled: %s", exc)
        return None


def _cache_key(question: str, docs: list[Document], history: list[dict]) -> str:
    """
    Key = SHA-256 of (question + doc contents + chat history).
    Different history → different key → no stale answers across conversation turns.
    """
    payload = json.dumps({
        "q": question.strip().lower(),
        "docs": [d.page_content for d in docs],
        "history": history,
    }, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"{REDIS_RESPONSE_PREFIX}{digest}"


class ResponseCache:
    """
    Wraps the generate step with a Redis cache.

    Cache key:  SHA-256(question + doc_contents + chat_history)[:24]
    TTL:        REDIS_RESPONSE_TTL seconds (default 5 min)
    Fallback:   calls generate_fn directly if Redis is down
    """

    def __init__(self) -> None:
        self._client = _make_client()
        if self._client:
            logger.info("Response cache: Redis  (TTL=%ds)", REDIS_RESPONSE_TTL)
        else:
            logger.info("Response cache: disabled (Redis unavailable)")

    def get_or_generate(
        self,
        question: str,
        docs: list[Document],
        history: list[dict],
        generate_fn,
    ) -> tuple[str, bool]:
        """
        Return (answer, cache_hit).
        Calls generate_fn() on a miss and stores the result.
        """
        if not self._client:
            return generate_fn(), False

        key = _cache_key(question, docs, history)
        try:
            cached = self._client.get(key)
            if cached:
                logger.debug("Response cache HIT  key=%s", key)
                return cached, True
        except Exception as exc:
            logger.warning("Redis get failed: %s", exc)

        answer = generate_fn()

        try:
            self._client.set(key, answer, ex=REDIS_RESPONSE_TTL)
            logger.debug("Response cache MISS key=%s — cached answer", key)
        except Exception as exc:
            logger.warning("Redis set failed: %s", exc)

        return answer, False
