"""Typed graph records with source provenance."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provenance:
    source_document: str
    source_section: str
    evidence_text: str
    chunk_id: str = ""


@dataclass
class GraphEntity:
    entity_id: str
    name: str
    entity_type: str
    provenance: list[Provenance] = field(default_factory=list)


@dataclass(frozen=True)
class GraphRelationship:
    relationship_id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    provenance: Provenance
