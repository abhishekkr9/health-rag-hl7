"""Load documents from the data/ directory (plain text + FHIR JSON bundles)."""

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from src.config import DATA_DIR
from src.fhir.parsers import PARSERS

_FILE_UUID_RE = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.IGNORECASE,
)


def build_patient_id_map(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """
    Scan every FHIR JSON filename and return a mapping of
    lowercase UUID → patient name (as stored in the vector index).

    FHIR bundle filenames follow the pattern:
        FirstName_MiddleName_LastName_<uuid>.json
    e.g.:
        Veronica155_Jacobi462_3e6d9a42-8393-b21c-6666-c3c0249f1495.json

    The extracted name matches the text produced by the Patient parser,
    so hybrid search will reliably find the right documents when the
    UUID is rewritten to the patient name before retrieval.
    """
    id_to_name: dict[str, str] = {}
    for path in data_dir.glob("**/*.json"):
        m = _FILE_UUID_RE.search(path.stem)
        if not m:
            continue
        uuid = m.group(0).lower()
        # Everything before the UUID in the filename stem is the patient name
        name = path.stem[: m.start()].rstrip("_").replace("_", " ").strip()
        if name:
            id_to_name[uuid] = name
    return id_to_name


def parse_fhir_bundle(bundle: dict, source: str) -> list[Document]:
    """Convert a FHIR Bundle into Documents grouped by resource type per patient."""
    groups: dict[str, list[str]] = {}
    patient_name = ""
    patient_id   = ""

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rt       = resource.get("resourceType", "")
        parser   = PARSERS.get(rt)
        if parser is None:
            continue
        text = parser(resource)
        if not text:
            continue
        if rt == "Patient":
            patient_name = text.split("\n")[0].replace("Patient: ", "").strip()
            patient_id   = resource.get("id", "")
        groups.setdefault(rt, []).append(text)

    return [
        Document(
            page_content=f"[{rt} records for {patient_name}]\n" + "\n".join(lines),
            metadata={
                "source":        source,
                "resource_type": rt,
                "patient":       patient_name,
                "patient_id":    patient_id,   # enables filtered retrieval after re-indexing
            },
        )
        for rt, lines in groups.items()
    ]


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load all .txt, .md, and FHIR JSON Bundle files from data_dir."""
    docs: list[Document] = []
    patient_names: list[str] = []

    for path in sorted(data_dir.glob("**/*")):
        if path.suffix in {".txt", ".md"} and path.is_file():
            docs.append(Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path.relative_to(data_dir))},
            ))

    for path in sorted(data_dir.glob("**/*.json")):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if bundle.get("resourceType") == "Bundle":
            bundle_docs = parse_fhir_bundle(bundle, str(path.relative_to(data_dir)))
            docs.extend(bundle_docs)
            for doc in bundle_docs:
                name = doc.metadata.get("patient", "")
                if name and doc.metadata.get("resource_type") == "Patient" and name not in patient_names:
                    patient_names.append(name)

    if patient_names:
        summary = (
            f"Patient Index: There are {len(patient_names)} patients in this dataset.\n"
            f"Patient list:\n" + "\n".join(f"- {n}" for n in sorted(patient_names))
        )
        docs.append(Document(
            page_content=summary,
            metadata={"source": "generated/patient_index", "resource_type": "PatientIndex", "patient": ""},
        ))

    if not docs:
        raise FileNotFoundError(f"No supported files found in '{data_dir}'.")
    return docs
