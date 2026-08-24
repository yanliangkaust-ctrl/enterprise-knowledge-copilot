"""Rule-based relationships grounded in explicit corpus language."""

import re
from collections.abc import Sequence

from backend.document_ingestion import ChunkRecord
from backend.entity_extraction import normalize_entity_id
from backend.graph_schema import GraphRelationship, Provenance


def extract_relationships(chunks: Sequence[ChunkRecord]) -> list[GraphRelationship]:
    """Extract only relationships supported by explicit phrases in each chunk."""

    relationships: list[GraphRelationship] = []
    for chunk in chunks:
        text = chunk.text
        provenance = Provenance(chunk.document_name, chunk.section_heading, text, chunk.chunk_id)
        if re.search(r"deploy(?:ed|s|ment)? .*OCR Service.*API Gateway.*Kubernetes", text, re.IGNORECASE):
            relationships.extend([
                _edge("OCR Service", "Kubernetes", "DEPLOYED_ON", provenance),
                _edge("API Gateway", "Kubernetes", "DEPLOYED_ON", provenance),
            ])
        if re.search(r"API depends on the OCR Service, Document Storage, Kubernetes.*Monitoring Service", text, re.IGNORECASE):
            for target in ("OCR Service", "Document Storage", "Kubernetes", "Monitoring Service"):
                relationships.append(_edge("API Gateway", target, "DEPENDS_ON", provenance))
        if re.search(r"Platform Engineering Team owns Kubernetes", text, re.IGNORECASE):
            relationships.append(_edge("Kubernetes", "Platform Engineering Team", "OWNED_BY", provenance))
        if re.search(r"Security Team (?:reviews|approves|confirmed|approved)", text, re.IGNORECASE):
            for target in ("API Gateway", "Document Storage"):
                if target.casefold() in text.casefold():
                    relationships.append(_edge("Security Team", target, "ASSOCIATED_WITH", provenance))
        if re.search(r"worker pods reached their memory limit", text, re.IGNORECASE):
            relationships.extend([
                _edge("Batch OCR Failure", "Kubernetes", "CAUSED_BY", provenance),
                _edge("Batch OCR Failure", "OCR Service", "AFFECTS", provenance),
                _edge("Batch OCR Failure", "UAT Environment", "AFFECTS", provenance),
            ])
        if re.search(r"increased Kubernetes worker capacity|increased worker capacity", text, re.IGNORECASE):
            relationships.append(_edge("Batch OCR Failure", "Platform Engineering Team", "MITIGATED_BY", provenance))
        if re.search(r"Kubernetes capacity is tracked as R-001", text, re.IGNORECASE):
            relationships.extend([
                _edge("Kubernetes", "R-001", "ASSOCIATED_WITH", provenance),
                _edge("ADR-004", "R-001", "REFERENCES", provenance),
            ])
        if re.search(r"event is linked to risk R-001", text, re.IGNORECASE):
            relationships.append(_edge("Batch OCR Failure", "R-001", "ASSOCIATED_WITH", provenance))
        if re.search(r"reproduced the capacity behavior documented in the Batch OCR Failure incident", text, re.IGNORECASE):
            relationships.append(_edge("UAT Report", "Batch OCR Failure", "REFERENCES", provenance))
    unique: dict[str, GraphRelationship] = {relationship.relationship_id: relationship for relationship in relationships}
    return list(unique.values())


def _edge(source: str, target: str, relationship_type: str, provenance: Provenance) -> GraphRelationship:
    relationship_id = "|".join((normalize_entity_id(source), relationship_type.casefold(), normalize_entity_id(target), provenance.chunk_id))
    return GraphRelationship(
        relationship_id=relationship_id,
        source_entity=normalize_entity_id(source),
        target_entity=normalize_entity_id(target),
        relationship_type=relationship_type,
        provenance=provenance,
    )
