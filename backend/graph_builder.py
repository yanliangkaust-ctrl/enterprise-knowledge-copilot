"""Build the in-memory knowledge graph from existing document chunks."""

from collections.abc import Sequence

from backend.document_ingestion import ChunkRecord
from backend.entity_extraction import extract_entities
from backend.knowledge_graph import KnowledgeGraph
from backend.relationship_extraction import extract_relationships


def build_knowledge_graph(chunks: Sequence[ChunkRecord]) -> KnowledgeGraph:
    """Extract entities and explicit relationships into a traceable graph."""

    graph = KnowledgeGraph()
    for entity in extract_entities(chunks):
        graph.add_entity(entity)
    for relationship in extract_relationships(chunks):
        graph.add_relationship(relationship)
    return graph
