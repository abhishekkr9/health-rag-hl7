"""Redis-backed chat history for short-term conversation memory."""

import json
import logging

import redis

from src.config import (
    REDIS_HOST,
    REDIS_HISTORY_PREFIX,
    REDIS_HISTORY_TTL,
    REDIS_PASSWORD,
    REDIS_PORT,
)

logger = logging.getLogger(__name__)


def _make_client() -> redis.Redis | None:
    """Return a Redis client, or None if Redis is unreachable."""
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
        logger.warning("Redis unavailable — falling back to in-memory history: %s", exc)
        return None


class ChatHistory:
    """
    Persist conversation turns in Redis.

    Each session gets its own key: ``fhir_rag:history:<session_id>``.
    History is stored as a JSON list and expires after REDIS_HISTORY_TTL seconds.
    Falls back silently to in-memory if Redis is unreachable.
    """

    def __init__(self, session_id: str) -> None:
        self._key    = f"{REDIS_HISTORY_PREFIX}{session_id}"
        self._client = _make_client()
        self._memory: list[dict] = []   # fallback when Redis is down

        if self._client:
            # Load existing history for this session (resume across restarts)
            raw = self._client.get(self._key)
            if raw:
                try:
                    self._memory = json.loads(raw)
                    logger.info("Resumed %d turns from Redis history.", len(self._memory))
                except json.JSONDecodeError:
                    self._memory = []
            logger.info("Chat history backend: Redis  (session: %s)", session_id)
        else:
            logger.warning("Chat history backend: in-memory (Redis unavailable)")

    def get(self) -> list[dict]:
        """Return the current history list."""
        return list(self._memory)

    def append(self, role: str, content: str) -> None:
        """Add one turn and persist to Redis."""
        self._memory.append({"role": role, "content": content})
        if self._client:
            try:
                self._client.set(self._key, json.dumps(self._memory), ex=REDIS_HISTORY_TTL)
            except Exception as exc:
                logger.warning("Failed to persist history to Redis: %s", exc)

    def clear(self) -> None:
        """Delete history for this session."""
        self._memory = []
        if self._client:
            try:
                self._client.delete(self._key)
            except Exception:
                pass
