"""LangFuse tracing setup — optional, enabled via environment variables."""

import logging
import os
import uuid

logger = logging.getLogger(__name__)

from langfuse import get_client as get_langfuse_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from src.config import LANGFUSE_HOST_DEFAULT, LANGFUSE_RUN_NAME, LANGFUSE_TAGS


def setup_tracing() -> tuple[LangfuseCallbackHandler | None, str | None]:
    """
    Initialise LangFuse tracing if credentials are available.

    Returns (handler, session_id) when enabled, (None, None) otherwise.
    Credentials are read from env vars — never hard-coded:
        LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    """
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        logger.warning("LangFuse tracing disabled (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set).")
        return None, None

    handler    = LangfuseCallbackHandler()
    session_id = str(uuid.uuid4())
    host       = os.environ.get("LANGFUSE_HOST", LANGFUSE_HOST_DEFAULT)
    logger.info("LangFuse tracing enabled → %s", host)
    logger.info("Session ID: %s", session_id)
    return handler, session_id


def build_invoke_config(handler: LangfuseCallbackHandler, session_id: str) -> dict:
    """Build the LangGraph invoke config that attaches tracing to a run."""
    return {
        "callbacks": [handler],
        "run_name": LANGFUSE_RUN_NAME,
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": LANGFUSE_TAGS,
        },
    }


def shutdown_tracing() -> None:
    """Flush all queued trace events before the process exits."""
    get_langfuse_client().shutdown()
