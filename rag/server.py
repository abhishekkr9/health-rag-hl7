"""FastAPI server — wraps the FHIR RAG pipeline for the React frontend.

Run from the rag/ directory:
    uvicorn server:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from src.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import LLM_MODEL

# Pipeline state — populated during lifespan startup
_rag_graph = None
_neo4j_driver = None
_langfuse_handler = None
_session_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_graph, _neo4j_driver, _langfuse_handler, _session_id

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Create a key at https://openrouter.ai/keys"
        )

    from src.vectorstore.store import get_vectorstore
    from src.pipeline.checkpointer import get_checkpointer
    from src.pipeline.graph import build_rag_graph
    from src.pipeline.tracing import setup_tracing

    logger.info("[1/4] Connecting to Neo4j and loading embeddings model...")
    retriever, _neo4j_driver = get_vectorstore()
    logger.info("[1/4] Neo4j ready.")

    logger.info("[2/4] Connecting to PostgreSQL checkpointer...")
    checkpointer = get_checkpointer()

    logger.info("[3/4] Initialising LangFuse tracing...")
    _langfuse_handler, _session_id = setup_tracing()

    logger.info("[4/4] Compiling LangGraph RAG pipeline...")
    _rag_graph = build_rag_graph(retriever, checkpointer=checkpointer)
    logger.info("Server ready — listening for requests.")

    yield

    # Graceful shutdown
    if _neo4j_driver is not None:
        _neo4j_driver.close()
    if _langfuse_handler is not None:
        from src.pipeline.tracing import shutdown_tracing
        shutdown_tracing()


app = FastAPI(title="FHIR RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Schemas ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    model: str


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Liveness probe — confirms the server and pipeline are initialised."""
    return HealthResponse(status="ok", model=LLM_MODEL)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Run one turn of the FHIR RAG pipeline.

    The thread_id maps to a LangGraph checkpointer key so that chat history
    is maintained across requests within the same browser session.
    """
    if _rag_graph is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialised.")

    from src.pipeline.tracing import build_invoke_config

    config = (
        build_invoke_config(_langfuse_handler, _session_id)
        if _langfuse_handler
        else {}
    )
    config.setdefault("configurable", {})["thread_id"] = req.thread_id

    try:
        result = _rag_graph.invoke(
            {"question": req.question, "chat_history": []},
            config=config,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail="Pipeline error — see server logs.")

    return ChatResponse(answer=result["answer"], thread_id=req.thread_id)
