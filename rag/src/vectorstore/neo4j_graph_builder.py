"""Build Neo4j graph from FHIR resources with vector embeddings on chunks."""

import logging
from typing import Optional

from neo4j import Driver, GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Neo4jGraphBuilder:
    """Ingest FHIR documents and create Neo4j nodes/relationships with vector embeddings."""

    def __init__(self, driver: Driver):
        self.driver = driver
        self.embedding_dimension = None  # Will be set on first use

    def index_documents(self, docs: list[Document], embeddings_func) -> None:
        """
        Ingest documents and create Neo4j nodes + relationships.

        For each document:
        - Extract metadata (patient_name, patient_id, resource_type, source)
        - Create Patient node (if not exists)
        - Create Resource node (Condition, Medication, etc.)
        - Create Chunk nodes with embeddings
        - Link Resource -[:HAS_CHUNK]-> Chunk
        - Link Patient -[:HAS_{RESOURCE_TYPE}]-> Resource
        """
        grouped = self._group_by_patient_and_resource(docs)

        for (patient_name, patient_id), resources_by_type in grouped.items():
            with self.driver.session() as session:
                # Create Patient node
                patient_node_id = self._ensure_patient_node(
                    session, patient_name, patient_id
                )

                # Process each resource type
                for resource_type, documents in resources_by_type.items():
                    for doc in documents:
                        resource_node_id = self._create_or_update_resource_node(
                            session,
                            resource_type,
                            patient_node_id,
                            doc,
                        )

                        # Create chunks and link to resource
                        self._create_chunks_for_resource(
                            session,
                            resource_node_id,
                            resource_type,
                            doc,
                            embeddings_func,
                        )

                        # Link Patient to Resource
                        rel_type = f"HAS_{resource_type.upper()}"
                        session.run(
                            f"""
                            MATCH (p:Patient {{id: $patient_id}})
                            MATCH (r:{resource_type} {{id: $resource_id}})
                            MERGE (p)-[:{rel_type}]->(r)
                            """,
                            patient_id=patient_node_id,
                            resource_id=resource_node_id,
                        )

    def _group_by_patient_and_resource(
        self, docs: list[Document]
    ) -> dict[tuple[str, str], dict[str, list[Document]]]:
        """
        Group documents by (patient_name, patient_id) and then by resource_type.

        Returns: {(patient_name, patient_id): {resource_type: [docs]}}
        """
        grouped: dict[tuple[str, str], dict[str, list[Document]]] = {}
        for doc in docs:
            patient_name = doc.metadata.get("patient", "Unknown")
            patient_id = doc.metadata.get("patient_id", "")
            resource_type = doc.metadata.get("resource_type", "Unknown")

            key = (patient_name, patient_id)
            if key not in grouped:
                grouped[key] = {}
            if resource_type not in grouped[key]:
                grouped[key][resource_type] = []
            grouped[key][resource_type].append(doc)
        return grouped

    def _ensure_patient_node(
        self, session, patient_name: str, patient_id: str
    ) -> str:
        """Create or reuse Patient node. Return its ID."""
        result = session.run(
            """
            MERGE (p:Patient {id: $patient_id})
            SET p.name = $patient_name, p.created_at = timestamp()
            RETURN p.id as id
            """,
            patient_id=patient_id or f"patient_{patient_name.replace(' ', '_')}",
            patient_name=patient_name,
        )
        record = result.single()
        return record["id"] if record else patient_id

    def _create_or_update_resource_node(
        self,
        session,
        resource_type: str,
        patient_node_id: str,
        doc: Document,
    ) -> str:
        """Create or update a resource node. Return its ID."""
        source = doc.metadata.get("source", "unknown")
        resource_id = f"{resource_type}_{source}".replace("/", "_").replace(".", "_")

        query = f"""
        MERGE (r:{resource_type} {{id: $resource_id}})
        SET r.source = $source,
            r.patient_id = $patient_id,
            r.text_preview = $text_preview,
            r.created_at = timestamp()
        RETURN r.id as id
        """
        result = session.run(
            query,
            resource_id=resource_id,
            source=source,
            patient_id=patient_node_id,
            text_preview=doc.page_content[:200],  # Store preview for debugging
        )
        record = result.single()
        return record["id"] if record else resource_id

    def _create_chunks_for_resource(
        self,
        session,
        resource_node_id: str,
        resource_type: str,
        doc: Document,
        embeddings_func,
    ) -> None:
        """Split document into chunks, embed, create Chunk nodes, and link to resource."""
        # For now, treat the entire document as one chunk
        # (Later: use RecursiveCharacterTextSplitter if needed)
        text = doc.page_content
        embedding = embeddings_func([text])[0]

        chunk_id = f"chunk_{resource_node_id}_0"

        # Create Chunk node with embedding
        session.run(
            f"""
            MERGE (c:Chunk {{id: $chunk_id}})
            SET c.text = $text,
                c.embedding = $embedding,
                c.resource_type = $resource_type,
                c.source = $source,
                c.created_at = timestamp()
            """,
            chunk_id=chunk_id,
            text=text,
            embedding=embedding,
            resource_type=resource_type,
            source=doc.metadata.get("source", "unknown"),
        )

        # Link resource to chunk
        session.run(
            f"""
            MATCH (r:{resource_type} {{id: $resource_id}})
            MATCH (c:Chunk {{id: $chunk_id}})
            MERGE (r)-[:HAS_CHUNK]->(c)
            """,
            resource_id=resource_node_id,
            chunk_id=chunk_id,
        )

    def create_vector_index(self, embedding_dimension: int = 768) -> None:
        """Create vector index on Chunk nodes for semantic search.
        
        If an index already exists with different dimensions, drop and recreate it.
        """
        self.embedding_dimension = embedding_dimension
        
        with self.driver.session() as session:
            # First, try to drop any existing mismatched index
            try:
                session.run("DROP INDEX chunk_embeddings")
                logger.info("Dropped existing vector index for recreation.")
            except Exception as e:
                if "not found" not in str(e).lower():
                    logger.debug("Could not drop index (may not exist): %s", e)
            
            # Now create the index with correct dimensions
            try:
                session.run(
                    f"""
                    CALL db.index.vector.createNodeIndex(
                        'chunk_embeddings',
                        'Chunk',
                        'embedding',
                        {embedding_dimension},
                        'cosine'
                    )
                    """
                )
                logger.info("Vector index created with %d dimensions.", embedding_dimension)
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("Vector index already exists with correct dimensions.")
                else:
                    logger.warning("Error creating vector index: %s", e)

    def similarity_search(
        self, query_embedding: list[float], k: int = 6
    ) -> list[dict]:
        """
        Vector similarity search on Chunk nodes.

        Returns: [{"id": chunk_id, "text": text, "resource_type": ..., "score": ...}]
        """
        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'chunk_embeddings',
                        $k,
                        $query_embedding
                    )
                    YIELD node AS c, score
                    RETURN c.id as id, c.text as text, c.resource_type as resource_type,
                           c.source as source, score
                    ORDER BY score DESC
                    LIMIT $k
                    """,
                    query_embedding=query_embedding,
                    k=k,
                )
                return [dict(record) for record in result]
            except Exception as e:
                logger.error("Vector similarity search failed: %s", e)
                return []

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()
