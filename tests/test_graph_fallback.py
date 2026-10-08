import pytest

from backend.agentic_orchestrator import ask_agentic
from backend.document_ingestion import chunk_documents, parse_markdown_content
from backend.graph_builder import build_knowledge_graph
from backend.langgraph_orchestrator import _graph_weakness
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


def knowledge(*texts):
    chunks = chunk_documents([parse_markdown_content(f"policy{i}.md", text) for i, text in enumerate(texts)])
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    return chunks, index, build_knowledge_graph(chunks)


def test_empty_graph_activates_fallback():
    kb = knowledge("# Policy\nKubernetes runs worker jobs.")
    result = ask_agentic("What depends on Kubernetes?", *kb)
    assert [d.tool for d in result.trace.tool_decisions] == ["graph_search", "hybrid_search"]
    assert result.trace.retrieval_checks[0]["status"] == "empty"
    assert result.trace.retrieval_checks[0]["final_mode"] == "hybrid_search"
    assert "no direct evidence" in result.trace.tool_decisions[-1].reason


def test_nonempty_irrelevant_graph_activates_fallback_deterministically():
    # Relationship extraction yields a UAT edge, while the evidence sentence has
    # none of this question's retained support words. No benchmark-specific code.
    kb = knowledge("# Incident\nKubernetes worker pods reached their memory limit.",
                   "# Validation\nThe UAT Environment validates security controls and deployment readiness.")
    question = "How does the UAT Environment support release validation?"
    result = ask_agentic(question, *kb)
    assert result == ask_agentic(question, *kb)
    assert [d.tool for d in result.trace.tool_decisions] == ["graph_search", "hybrid_search"]
    check = result.trace.retrieval_checks[0]
    assert check["original_mode"] == "graph_search"
    assert check["final_mode"] == "hybrid_search"
    assert check["fallback_activated"]
    assert check["status"] == "weak_support"
    assert "no question-relevant textual support" in result.trace.tool_decisions[-1].reason
    assert check["original_evidence"][0]["chunk_id"] == kb[0][0].chunk_id
    assert result.trace.evidence_sufficient
    assert len(result.response.retrieved_evidence) <= 5
    assert result.trace.retrieved_evidence_count == len(result.response.retrieved_evidence)
    by_id = {c.chunk_id: c for c in kb[0]}
    for e in result.response.retrieved_evidence:
        c = by_id[e.chunk_id]
        assert (e.source_document, e.section, e.text) == (c.document_name, c.section_heading, c.text)


def test_strong_graph_relationship_does_not_fallback():
    kb = knowledge("# Ownership\nPlatform Engineering Team owns Kubernetes.")
    result = ask_agentic("Who owns Kubernetes?", *kb)
    assert [d.tool for d in result.trace.tool_decisions] == ["graph_search"]
    assert result.trace.retrieval_checks[0]["status"] == "adequate_support"
    assert not result.trace.retrieval_checks[0]["fallback_activated"]
    assert result.trace.evidence_sufficient
    assert result.response.retrieved_evidence[0].relationship_type == "OWNED_BY"


@pytest.mark.parametrize("question", ["What is the OCR Service license price?",
                                     "What protocol does Kubernetes use to replicate the archive?"])
def test_unsupported_facts_fallback_but_still_refuse(question):
    kb = knowledge("# Deployment\nThe platform will deploy the OCR Service and API Gateway on Kubernetes.")
    result = ask_agentic(question, *kb)
    assert [d.tool for d in result.trace.tool_decisions] == ["graph_search", "hybrid_search"]
    assert result.trace.retrieval_checks[0]["status"] == "weak_support"
    assert "requested factual support" in result.trace.tool_decisions[-1].reason
    assert not result.trace.evidence_sufficient
    assert result.response.grounding_status == "Insufficient evidence"
    assert result.response.sources == []


def test_empty_criterion_and_factually_strong_graph_support():
    assert _graph_weakness("question", "question", []) == "Graph search returned no direct evidence."


def test_api_serializes_original_and_fallback_evidence_trace(monkeypatch):
    from backend import api
    from fastapi.testclient import TestClient
    chunks, index, graph = knowledge(
        "# Incident\nKubernetes worker pods reached their memory limit.",
        "# Validation\nThe UAT Environment validates security controls and deployment readiness.")
    monkeypatch.setattr(api, "build_knowledge_base", lambda: ([], chunks, index, graph))
    payload = TestClient(api.app).post("/query", json={
        "question": "How does the UAT Environment support release validation?"}).json()
    check = payload["execution_trace"]["retrieval_checks"][0]
    assert payload["selected_retrieval_route"] == check["original_mode"] == "graph_search"
    assert check["status"] == "weak_support" and check["fallback_activated"]
    assert check["final_mode"] == "hybrid_search"
    assert check["original_evidence"][0]["relationship_type"] == "AFFECTS"
    assert payload["provenance"] and payload["evidence_sufficient"]
