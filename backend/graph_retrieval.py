"""Explainable graph retrieval over the in-memory knowledge graph."""

from dataclasses import dataclass
import re
from collections.abc import Sequence

from backend.document_ingestion import ChunkRecord
from backend.entity_extraction import normalize_entity_id
from backend.knowledge_graph import KnowledgeGraph


@dataclass(frozen=True)
class GraphRetrievalResult:
    """A graph-selected chunk with the fact that selected it."""

    chunk: ChunkRecord
    score: float
    rank: int
    matched_entity: str
    relationship_type: str
    related_entity: str
    path: tuple[str, ...]
    reason: str


def match_question_entities(question: str, graph: KnowledgeGraph) -> list:
    """Match known graph entity names mentioned explicitly in a question."""

    question_text = question.casefold()
    aliases = {
        "batch failure": "Batch OCR Failure",
        "batch ocr incident": "Batch OCR Failure",
        "ocr platform api": "API Gateway",
        "risk r-001": "R-001",
    }
    matched_names = {
        alias_target
        for alias, alias_target in aliases.items()
        if alias in question_text
    }
    matched_names.update(
        entity.name
        for entity in graph.entities()
        if entity.name.casefold() in question_text
    )
    return [entity for entity in graph.entities() if entity.name in matched_names]


def search_graph(
    question: str,
    graph: KnowledgeGraph,
    chunks: Sequence[ChunkRecord],
    top_k: int = 5,
) -> list[GraphRetrievalResult]:
    """Retrieve direct relationship evidence for entities named in the question."""

    if top_k <= 0 or not question.strip():
        return []
    matched_entities = match_question_entities(question, graph)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    candidates: dict[tuple[str, str], GraphRetrievalResult] = {}
    for entity in matched_entities:
        for relationship in graph.relationships(entity.entity_id):
            chunk = chunk_by_id.get(relationship.provenance.chunk_id)
            if chunk is None:
                continue
            target_id = relationship.target_entity if relationship.source_entity == entity.entity_id else relationship.source_entity
            target = graph.entity(target_id)
            related_name = target.name if target else target_id
            path = (entity.name, related_name) if relationship.source_entity == entity.entity_id else (related_name, entity.name)
            result = GraphRetrievalResult(
                chunk=chunk,
                score=1.0,
                rank=0,
                matched_entity=entity.name,
                relationship_type=relationship.relationship_type,
                related_entity=related_name,
                path=path,
                reason=f"Question names {entity.name}; explicit {relationship.relationship_type} fact matched.",
            )
            key = (chunk.chunk_id, relationship.relationship_id)
            candidates[key] = result

    ranked = sorted(candidates.values(), key=lambda result: (-result.score, result.chunk.document_name, result.chunk.chunk_id))
    return [
        GraphRetrievalResult(
            chunk=result.chunk,
            score=result.score,
            rank=rank,
            matched_entity=result.matched_entity,
            relationship_type=result.relationship_type,
            related_entity=result.related_entity,
            path=result.path,
            reason=result.reason,
        )
        for rank, result in enumerate(ranked[:top_k], start=1)
    ]
