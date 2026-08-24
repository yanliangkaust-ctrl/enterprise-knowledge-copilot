from pathlib import Path

from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.entity_extraction import extract_entities, normalize_entity_id
from backend.graph_builder import build_knowledge_graph
from backend.relationship_extraction import extract_relationships


PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _chunks():
    return chunk_documents(load_markdown_documents(SAMPLE_DOCS))


def test_entity_extraction_and_deduplication():
    entities = extract_entities(_chunks())
    names = [entity.name for entity in entities]

    assert names.count("OCR Service") == 1
    assert names.count("Kubernetes") == 1
    assert any(entity.name == "R-001" and entity.entity_type == "Risk" for entity in entities)
    assert any(entity.name == "API Specification" and entity.entity_type == "Document" for entity in entities)


def test_relationship_extraction_preserves_provenance():
    relationships = extract_relationships(_chunks())
    deployed = next(
        relationship
        for relationship in relationships
        if relationship.source_entity == normalize_entity_id("OCR Service")
        and relationship.target_entity == normalize_entity_id("Kubernetes")
    )

    assert deployed.relationship_type == "DEPLOYED_ON"
    assert deployed.provenance.source_document == "ADR_Deployment.md"
    assert deployed.provenance.source_section == "Decision"
    assert deployed.provenance.evidence_text
    assert deployed.provenance.chunk_id


def test_graph_construction_and_lookup():
    graph = build_knowledge_graph(_chunks())
    ocr = graph.entity("OCR Service")

    assert ocr is not None
    assert ocr.entity_type == "Service / System"
    assert graph.entity_type_counts()["Service / System"] >= 4
    assert graph.relationship_type_counts()["DEPLOYED_ON"] >= 2
    assert graph.relationship_count > 0
    assert graph.provenance("OCR Service")


def test_neighbor_and_relationship_lookup():
    graph = build_knowledge_graph(_chunks())
    neighbor_names = {entity.name for entity in graph.neighbors("OCR Service")}
    relationship_types = {relationship.relationship_type for relationship in graph.relationships("OCR Service")}

    assert "Kubernetes" in neighbor_names
    assert "Batch OCR Failure" in neighbor_names
    assert "DEPLOYED_ON" in relationship_types
    assert "AFFECTS" in relationship_types


def test_valid_and_nonexistent_path_traversal():
    graph = build_knowledge_graph(_chunks())
    paths = graph.paths("OCR Service", "R-001")

    assert paths
    assert [relationship.relationship_type for relationship in paths[0]] == [
        "DEPLOYED_ON",
        "ASSOCIATED_WITH",
    ]
    assert graph.paths("OCR Service", "nonexistent entity") == []


def test_unsupported_relationship_is_not_invented():
    graph = build_knowledge_graph(_chunks())

    assert not any(
        relationship.source_entity == normalize_entity_id("OCR Service")
        and relationship.relationship_type == "OWNED_BY"
        for relationship in graph.relationships()
    )
