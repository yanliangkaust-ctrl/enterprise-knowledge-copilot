from pathlib import Path

from backend.agentic_orchestrator import ask_agentic, plan_tools
from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex

PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _kb():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    return chunks, index, graph


def test_semantic_oriented_question_routes_to_semantic():
    _, _, graph = _kb()
    intent, tools = plan_tools("How is batch capacity handled?", graph)
    assert intent == "semantic knowledge question"
    assert tools[0].tool == "semantic_search"


def test_graph_dependency_question_uses_graph_capability():
    chunks, index, graph = _kb()
    result = ask_agentic("What depends on Kubernetes?", chunks, index, graph)
    assert result.trace.tool_decisions[0].tool in {"graph_search", "hybrid_search"}
    assert result.trace.retrieved_evidence_count > 0
    assert result.response.retrieved_evidence


def test_multi_hop_question_routes_to_hybrid():
    chunks, index, graph = _kb()
    result = ask_agentic(
        "What risks could affect the OCR production deployment and which teams should be involved?",
        chunks, index, graph,
    )
    assert result.trace.tool_decisions[0].tool == "hybrid_search"
    assert result.trace.retrieved_evidence_count > 0


def test_unsupported_question_returns_insufficient_evidence():
    chunks, index, graph = _kb()
    result = ask_agentic(
        "What is the company's 2028 international expansion budget?",
        chunks, index, graph,
    )
    assert result.response.grounding_status == "Insufficient evidence"
    assert "Insufficient evidence" in result.response.answer
    assert not result.trace.evidence_sufficient


def test_multi_part_agentic_answer_is_bulleted_and_covers_risks_and_teams():
    chunks, index, graph = _kb()
    result = ask_agentic(
        "What risks could affect the OCR production deployment and which teams should be involved?",
        chunks, index, graph,
    )
    assert result.trace.evidence_sufficient
    assert "**Key risks**" in result.response.answer
    assert "**Teams involved**" in result.response.answer
    assert "- " in result.response.answer
    assert "Platform Engineering Team" in result.response.answer or "Security Team" in result.response.answer
