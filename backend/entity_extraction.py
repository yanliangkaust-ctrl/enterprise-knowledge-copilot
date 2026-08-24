"""Deterministic entity extraction for the Government OCR Platform corpus."""

import re
from collections.abc import Sequence

from backend.document_ingestion import ChunkRecord
from backend.graph_schema import GraphEntity, Provenance


KNOWN_ENTITIES = {
    "OCR Service": "Service / System",
    "API Gateway": "Service / System",
    "Document Storage": "Service / System",
    "Monitoring Service": "Service / System",
    "Kubernetes": "Technology",
    "Security Team": "Team",
    "Platform Engineering Team": "Team",
    "Operations Team": "Team",
    "UAT Environment": "Environment",
    "Risk Register": "Document",
    "API Specification": "Document",
    "Security Requirements": "Document",
    "Deployment Guide": "Document",
    "UAT Report": "Document",
    "ADR-004": "Decision",
    "Batch OCR Failure": "Incident",
}


def normalize_entity_id(name: str) -> str:
    """Create a stable, case-insensitive graph identifier."""

    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def extract_entities(chunks: Sequence[ChunkRecord]) -> list[GraphEntity]:
    """Extract known terminology and risk IDs with section-level provenance."""

    entities: dict[str, GraphEntity] = {}
    for chunk in chunks:
        provenance = Provenance(
            source_document=chunk.document_name,
            source_section=chunk.section_heading,
            evidence_text=chunk.text,
            chunk_id=chunk.chunk_id,
        )
        candidates = dict(KNOWN_ENTITIES)
        candidates[chunk.document_name.removesuffix(".md").replace("_", " ")] = "Document"
        always_include = {chunk.document_name.removesuffix(".md").replace("_", " ")}
        for risk_id in re.findall(r"\bR-\d{3}\b", chunk.text, flags=re.IGNORECASE):
            candidates[risk_id.upper()] = "Risk"
        if "Incident_" in chunk.document_name:
            candidates["Batch OCR Failure"] = "Incident"
            always_include.add("Batch OCR Failure")
        if "ADR_" in chunk.document_name:
            candidates["ADR-004"] = "Decision"
            always_include.add("ADR-004")

        for name, entity_type in candidates.items():
            if name.casefold() not in chunk.text.casefold() and name not in always_include:
                continue
            entity_id = normalize_entity_id(name)
            entity = entities.setdefault(entity_id, GraphEntity(entity_id, name, entity_type))
            if provenance not in entity.provenance:
                entity.provenance.append(provenance)
    return sorted(entities.values(), key=lambda entity: entity.name.casefold())
