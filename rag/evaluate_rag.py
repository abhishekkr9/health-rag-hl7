import json
import logging
import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

# Load env vars
load_dotenv()

from src.logging_config import setup_logging
from src.vectorstore.store import get_vectorstore
from src.pipeline.checkpointer import get_checkpointer
from src.pipeline.graph import build_rag_graph
from src.pipeline.tracing import setup_tracing, build_invoke_config, shutdown_tracing
from src.config import DATA_DIR
from langfuse import get_client as get_langfuse_client

setup_logging()
logger = logging.getLogger(__name__)

# Ragas requires a custom embedding wrapper for the judge
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

def run_evaluation():
    # 1. Setup the RAG Pipeline
    langfuse_handler, session_id = setup_tracing()
    retriever, neo4j_driver = get_vectorstore()
    checkpointer = get_checkpointer()
    rag_graph = build_rag_graph(retriever, checkpointer=checkpointer)

    # 3. Setup OpenRouter Judge LLM for Ragas
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    
    # Using a fast/smart model from OpenRouter for the judge
    judge_llm = ChatOpenAI(
        model="openai/gpt-4o-mini", # OpenRouter model suitable for judging
        temperature=0,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)
    
    # Ragas also requires an embedding model to compute answer relevance
    embedding_model = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    # 4. Load the Evaluation Dataset
    dataset_path = DATA_DIR / "evaluation_dataset.json"
    with open(dataset_path, "r") as f:
        eval_data = json.load(f)

    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    trace_ids = []

    # 5. Run the Pipeline on each question
    logger.info(f"Starting evaluation of {len(eval_data)} questions...")
    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        config = build_invoke_config(langfuse_handler, session_id) if langfuse_handler else {}
        config.setdefault("configurable", {})["thread_id"] = str(uuid.uuid4())
        
        logger.info(f"Invoking pipeline for: {question}")
        result = rag_graph.invoke(
            {"question": question, "chat_history": []},
            config=config,
        )
        
        answer = result["answer"]
        # Extract the page content from the retrieved documents
        contexts = [doc.page_content for doc in result.get("documents", [])]
        
        results_data["question"].append(question)
        results_data["answer"].append(answer)
        results_data["contexts"].append(contexts)
        results_data["ground_truth"].append(ground_truth)
        
        # Get the trace ID assigned to this run by Langfuse
        if langfuse_handler:
            trace_id = langfuse_handler.last_trace_id
            trace_ids.append(trace_id)
        else:
            trace_ids.append(None)

    # 6. Prepare dataset and run Ragas
    logger.info("Running Ragas evaluation suite...")
    dataset = Dataset.from_dict(results_data)
    
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    
    logger.info("Evaluation Complete!")
    print("\n--- Aggregate Scores ---")
    print(evaluation_result)
    
    # 7. Push scores back to Langfuse
    logger.info("Pushing scores to Langfuse...")
    langfuse_client = get_langfuse_client()
    df = evaluation_result.to_pandas()

    for i, row in df.iterrows():
        trace_id = trace_ids[i]
        if not trace_id:
            continue

        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            score = row.get(metric)
            if score is not None and not pd.isna(score):
                langfuse_client.create_score(
                    trace_id=trace_id,
                    name=metric,
                    value=float(score),
                    comment="Ragas OpenRouter Judge",
                )

    langfuse_client.flush()
    neo4j_driver.close()
    if langfuse_handler:
        shutdown_tracing()
    logger.info("All traces and scores synced to Langfuse.")

if __name__ == "__main__":
    run_evaluation()
