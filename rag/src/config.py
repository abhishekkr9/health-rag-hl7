from pathlib import Path

# Paths
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
EMBEDDING_CACHE = ROOT_DIR / ".cache" / "embeddings"

# Neo4j
NEO4J_URI      = "neo4j://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

# Models
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL       = "openrouter/free"

# Retrieval
RETRIEVER_K     = 6  # Neo4j: number of similar chunks to retrieve

# LangFuse
LANGFUSE_HOST_DEFAULT = "http://localhost:4000"
LANGFUSE_TAGS         = ["fhir-rag", "clinical-data"]
LANGFUSE_RUN_NAME     = "fhir-rag-query"

# Redis (chat history)
REDIS_HOST            = "127.0.0.1"
REDIS_PORT            = 6379
REDIS_PASSWORD        = "myredissecret"
REDIS_HISTORY_TTL        = 300     # seconds — 5 min
REDIS_HISTORY_PREFIX     = "fhir_rag:history:"

# Redis retrieval cache
REDIS_RETRIEVAL_TTL      = 300     # seconds — 5 min
REDIS_RETRIEVAL_PREFIX   = "fhir_rag:retrieval:"

# Redis LLM response cache
REDIS_RESPONSE_TTL       = 300     # seconds — 5 min
REDIS_RESPONSE_PREFIX    = "fhir_rag:response:"

# PostgreSQL (LangGraph checkpointer)
POSTGRES_URI = "postgresql://postgres:postgres@localhost:5432/postgres"
