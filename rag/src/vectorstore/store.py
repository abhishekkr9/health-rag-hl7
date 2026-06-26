"""Neo4j FHIR graph store with vector embeddings on chunks."""

import logging
from typing import Callable

from neo4j import GraphDatabase, Driver
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_CACHE, EMBEDDING_MODEL
from src.fhir.loader import load_documents
from src.vectorstore.neo4j_graph_builder import Neo4jGraphBuilder

logger = logging.getLogger(__name__)


def _build_embeddings() -> CacheBackedEmbeddings:
    """Wrap PubMedBERT embeddings with a local disk cache keyed by model name."""
    import hashlib
    EMBEDDING_CACHE.mkdir(parents=True, exist_ok=True)
    store      = LocalFileStore(str(EMBEDDING_CACHE))
    underlying = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    prefix     = EMBEDDING_MODEL.replace("/", "_")
    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        store,
        key_encoder=lambda text: prefix + "__" + hashlib.sha256(text.encode()).hexdigest(),
    )


class Neo4jRetriever(BaseRetriever):
    """Retrieve documents using Neo4j vector similarity search."""

    graph_builder: Neo4jGraphBuilder
    embeddings: CacheBackedEmbeddings
    k: int = 6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Embed query and search Neo4j for similar chunks."""
        query_embedding = self.embeddings.embed_query(query)
        results = self.graph_builder.similarity_search(query_embedding, k=self.k)
        docs = []
        for result in results:
            doc = Document(
                page_content=result.get("text", ""),
                metadata={
                    "source": result.get("source", ""),
                    "resource_type": result.get("resource_type", ""),
                    "similarity_score": result.get("score", 0),
                },
            )
            docs.append(doc)
        return docs

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """Async retrieval (same as sync for now)."""
        return self._get_relevant_documents(query)


def get_vectorstore() -> tuple[Neo4jRetriever, Driver]:
    """
    Connect to Neo4j and return a retriever + driver.

    First run:  parses FHIR data → creates Neo4j graph nodes/relationships → embeddings.
    Subsequent: connects to existing Neo4j and reuses graph.
    """
    embeddings = _build_embeddings()

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    logger.info("Connected to Neo4j at %s", NEO4J_URI)

    # Check if graph already exists
    with driver.session() as session:
        result = session.run("MATCH (n:Patient) RETURN count(n) as count")
        record = result.single()
        patient_count = record["count"] if record else 0

    if patient_count > 0:
        logger.info("Found existing Neo4j graph with %d patients — skipping re-embedding.", patient_count)
    else:
        logger.info("First run: parsing FHIR documents and building Neo4j graph...")
        docs = load_documents()
        logger.info("Loaded %d documents", len(docs))

        graph_builder = Neo4jGraphBuilder(driver)
        graph_builder.index_documents(docs, embeddings.embed_documents)
        
        # Get embedding dimension from the model
        embedding_dim = len(embeddings.embed_query("test"))
        graph_builder.create_vector_index(embedding_dimension=embedding_dim)
        logger.info("Done. FHIR graph persisted in Neo4j — reused on next run.")

    # Create retriever
    graph_builder = Neo4jGraphBuilder(driver)
    # Also create the index for the retriever (in case it's a fresh connection)
    embedding_dim = len(embeddings.embed_query("test"))
    graph_builder.create_vector_index(embedding_dimension=embedding_dim)
    retriever = Neo4jRetriever(graph_builder=graph_builder, embeddings=embeddings, k=6)

    return retriever, driver
