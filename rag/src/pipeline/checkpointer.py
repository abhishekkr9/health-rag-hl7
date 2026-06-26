"""LangGraph checkpointer setup — PostgreSQL with MemorySaver fallback."""

import logging

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from src.config import POSTGRES_URI

logger = logging.getLogger(__name__)

# Seconds to wait for a PostgreSQL connection before giving up
_CONNECT_TIMEOUT = 5


def get_checkpointer():
    """
    Return a LangGraph checkpointer.

    Tries PostgreSQL first (persistent conversation memory across server
    restarts).  If the database is unreachable within _CONNECT_TIMEOUT seconds
    it falls back to an in-memory checkpointer so the server still starts.
    Chat history is preserved within a session but lost on server restart.
    """
    logger.info("Connecting to PostgreSQL checkpointer (%s)...", POSTGRES_URI)
    try:
        conn = psycopg.connect(
            POSTGRES_URI,
            autocommit=True,
            connect_timeout=_CONNECT_TIMEOUT,
        )
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()   # creates langgraph checkpoint tables (idempotent)
        logger.info("LangGraph checkpointer: PostgreSQL (persistent conversation memory)")
        return checkpointer
    except Exception as exc:
        logger.warning(
            "PostgreSQL unavailable (%s) — falling back to in-memory checkpointer. "
            "Conversation history will be lost on server restart.",
            exc,
        )
        return MemorySaver()
