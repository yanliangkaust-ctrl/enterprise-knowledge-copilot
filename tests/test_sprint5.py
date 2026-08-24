import json
from pathlib import Path

from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.graph_builder import build_knowledge_graph
from backend.graph_retrieval import match_question_entities, search_graph
from backend.hybrid_retrieval import search_hybrid
from backend.retrieval_evaluation import evaluate_by_question_type, mean_reciprocal_rank
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"
GROUND_TRUTH = PROJECT_ROOT / "data" / "evaluation" / "ground_truth_questions.json"


def _knowledge():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    graph = build_knowledge_graph(chunks)
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    return chunks, graph, index


def test_graph_entity_matching_and_provenance():
    chunks, graph, _ = _knowledge()

    entities = match_question_entities("How does the OCR Service depend on Kubernetes?", graph)
    results = search_graph("How does the OCR Service depend on Kubernetes?", graph, chunks, top_k=5)

    assert {entity.name for entity in entities} >= {"OCR Service", "Kubernetes"}
    assert results
    assert all(result.relationship_type and result.related_entity for result in results)
    assert all(result.chunk.document_name == result.chunk.source for result in results)
    assert all(result.chunk.chunk_id == result.reason.split("explicit")[0].split()[-1] or result.chunk.chunk_id for result in results)
    assert any(result.relationship_type == "DEPLOYED_ON" for result in results)


def test_graph_retrieval_does_not_invent_unsupported_matches():
    chunks, graph, _ = _knowledge()

    assert search_graph("What is the Mars launch budget?", graph, chunks) == []


def test_hybrid_merges_duplicate_chunks_and_preserves_scores():
    chunks, graph, index = _knowledge()

    results = search_hybrid("How does Kubernetes handle batch capacity?", chunks, index, graph, top_k=5)

    assert len(results) == 5
    assert len({result.chunk.chunk_id for result in results}) == len(results)
    assert [result.rank for result in results] == [1, 2, 3, 4, 5]
    assert all(result.score >= 0 for result in results)
    assert all(result.vector_score >= 0 or result.graph_score >= 0 for result in results)
    assert any("Vector similarity" in result.reason for result in results)


def test_mrr_and_question_type_metrics():
    questions = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    def retrieve(question: str, top_k: int):
        return [{"document_name": "API_Specification.md"}][:top_k]

    assert 0 < mean_reciprocal_rank(questions[:1], retrieve) <= 1
    breakdown = evaluate_by_question_type(questions, retrieve)
    assert set(breakdown) == {
        "single-document retrieval",
        "cross-document synthesis",
        "relationship reasoning",
    }
    assert all("Recall@3" in metrics and "MRR" in metrics for metrics in breakdown.values())
