"""Resource-type-aware chunking for FHIR documents."""

from itertools import groupby

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chunk sizes tuned per resource type:
#   Observations/Procedures  → small  (each line is a self-contained data point)
#   Conditions/Medications   → medium (a few lines give useful context together)
#   Patient/AllergyIntolerance → large (small docs, keep whole)
CHUNK_CONFIG: dict[str, dict] = {
    "Observation":        {"chunk_size": 200, "chunk_overlap": 20},
    "Procedure":          {"chunk_size": 200, "chunk_overlap": 20},
    "DiagnosticReport":   {"chunk_size": 250, "chunk_overlap": 30},
    "Condition":          {"chunk_size": 400, "chunk_overlap": 60},
    "MedicationRequest":  {"chunk_size": 400, "chunk_overlap": 60},
    "Encounter":          {"chunk_size": 400, "chunk_overlap": 60},
    "Immunization":       {"chunk_size": 400, "chunk_overlap": 40},
    "Patient":            {"chunk_size": 800, "chunk_overlap": 0},
    "AllergyIntolerance": {"chunk_size": 800, "chunk_overlap": 0},
    "DocumentReference":  {"chunk_size": 600, "chunk_overlap": 80},  # clinical notes
    "Organization":       {"chunk_size": 500, "chunk_overlap": 0},
    "Location":           {"chunk_size": 500, "chunk_overlap": 0},
}
DEFAULT_CHUNK = {"chunk_size": 300, "chunk_overlap": 50}


def hybrid_chunk_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents using per-resource-type chunk sizes."""
    chunks: list[Document] = []
    keyfn = lambda d: d.metadata.get("resource_type", "")
    for rt, group in groupby(sorted(docs, key=keyfn), key=keyfn):
        cfg      = CHUNK_CONFIG.get(rt, DEFAULT_CHUNK)
        splitter = RecursiveCharacterTextSplitter(**cfg)
        chunks.extend(splitter.split_documents(list(group)))
    return chunks
