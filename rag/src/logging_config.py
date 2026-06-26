"""Centralised logging configuration for the FHIR RAG application."""

import logging
import logging.config
import os


class TruncateLongLogFilter(logging.Filter):
    """Trim oversized log lines to keep terminal output readable."""

    def __init__(self, max_len: int = 1200):
        super().__init__()
        self.max_len = max_len

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if isinstance(msg, str) and len(msg) > self.max_len:
                extra = len(msg) - self.max_len
                record.msg = f"{msg[:self.max_len]}... [truncated {extra} chars]"
                record.args = ()
        except Exception:
            # Logging must never break app flow.
            pass
        return True


def setup_logging() -> None:
    """
    Configure root logger.

    Level is read from env var LOG_LEVEL (default INFO).
    Format: timestamp | level | module | message
    Third-party noisy loggers are quieted to WARNING.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    max_log_len = int(os.environ.get("LOG_MAX_LEN", "1200"))

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["truncate_long"],
                "stream": "ext://sys.stderr",
            },
        },
        "filters": {
            "truncate_long": {
                "()": "src.logging_config.TruncateLongLogFilter",
                "max_len": max_log_len,
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
        "loggers": {
            # Quiet noisy third-party loggers
            "httpx":               {"level": "WARNING"},
            "httpcore":            {"level": "WARNING"},
            "weaviate":            {"level": "WARNING"},
            "langchain_weaviate":  {"level": "WARNING"},
            "neo4j":               {"level": "WARNING"},
            "sentence_transformers": {"level": "WARNING"},
            "transformers":        {"level": "WARNING"},
            "huggingface_hub":     {"level": "WARNING"},
            "langfuse":            {"level": "WARNING"},
            "urllib3":             {"level": "WARNING"},
            "redis.connection":    {"level": "ERROR"},
            "openai":              {"level": "WARNING"},
            "langchain":           {"level": "WARNING"},
            "langchain_core":      {"level": "WARNING"},
            "langgraph":           {"level": "WARNING"},
            "uvicorn.access":      {"level": "WARNING"},
        },
    })
