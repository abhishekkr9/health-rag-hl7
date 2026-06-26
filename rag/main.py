import logging
import os

from dotenv import load_dotenv
load_dotenv()

from src.logging_config import setup_logging
setup_logging()

from src.vectorstore.store import get_vectorstore
from src.pipeline.checkpointer import get_checkpointer
from src.pipeline.graph import build_rag_graph
from src.pipeline.tracing import setup_tracing, build_invoke_config, shutdown_tracing

logger = logging.getLogger(__name__)


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Create a key at https://openrouter.ai/keys"
        )

    langfuse_handler, session_id = setup_tracing()
    thread_id = session_id or "default"

    logger.info("Connecting to Neo4j (Docker)...")
    retriever, neo4j_driver = get_vectorstore()

    checkpointer = get_checkpointer()

    try:
        logger.info("Compiling LangGraph RAG pipeline...")
        rag_graph = build_rag_graph(retriever, checkpointer=checkpointer)

        print("Ready. Type 'exit' to quit.\n")  # user-facing, intentional

        while True:
            question = input("Question: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            config = build_invoke_config(langfuse_handler, session_id) if langfuse_handler else {}
            config.setdefault("configurable", {})["thread_id"] = thread_id

            try:
                result = rag_graph.invoke(
                    {"question": question, "chat_history": []},
                    config=config,
                )
                answer = result["answer"]
            except Exception as exc:
                logger.error("Pipeline error: %s", exc)
                answer = "Sorry, something went wrong processing your question. Please try again."
            print(f"Answer: {answer}\n")  # user-facing, intentional
    finally:
        neo4j_driver.close()
        if langfuse_handler:
            shutdown_tracing()


if __name__ == "__main__":
    main()


