from pathlib import Path

from backend.agentic_orchestrator import ask_agentic
from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex
from backend.session_context import InMemorySessionStore


PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _kb():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    return chunks, index, graph


def test_retrieval_only_query_does_not_call_operational_tool():
    chunks, index, graph = _kb()
    result = ask_agentic("What caused the OCR batch failure?", chunks, index, graph)

    assert result.trace.tool_executions == []
    assert result.response.retrieved_evidence
    assert result.operational_data == []


def test_tool_only_query_returns_simulated_incident_status():
    chunks, index, graph = _kb()
    result = ask_agentic("What is the current status of INC-001?", chunks, index, graph)

    assert result.trace.tool_decisions[0].tool == "incident_status_tool"
    assert result.trace.tool_executions[0].tool == "get_incident_status"
    assert result.trace.tool_executions[0].status == "found"
    assert result.response.retrieved_evidence == []
    assert result.operational_data[0]["owner"] == "Platform Engineering Team"
    assert "Simulated operational data" in result.response.answer


def test_combined_query_keeps_retrieval_and_operational_outputs_distinct():
    chunks, index, graph = _kb()
    result = ask_agentic("What caused INC-001 and what is its current status?", chunks, index, graph)

    assert {execution.tool for execution in result.trace.tool_executions} == {"get_incident_status"}
    assert result.response.retrieved_evidence
    assert result.operational_data[0]["incident_id"] == "INC-001"
    assert "Simulated operational data" in result.response.answer


def test_unknown_incident_does_not_fabricate_tool_result():
    chunks, index, graph = _kb()
    result = ask_agentic("What is the current status of INC-999?", chunks, index, graph)

    assert result.trace.tool_executions[0].status == "not_found"
    assert result.operational_data == []
    assert result.trace.evidence_sufficient is False
    assert result.response.grounding_status == "Insufficient evidence"
    assert "Insufficient evidence" in result.response.answer


def test_follow_up_resolves_incident_for_operational_tool():
    chunks, index, graph = _kb()
    store = InMemorySessionStore()
    first = ask_agentic("What is the current status of INC-001?", chunks, index, graph)
    store.record("incident-session", "What is the current status of INC-001?", first.response, graph, ("INC-001",))

    follow_up = ask_agentic(
        "Who owns it?",
        chunks,
        index,
        graph,
        session_context=store.get("incident-session"),
    )

    assert follow_up.trace.tool_executions[0].tool == "get_incident_status"
    assert follow_up.trace.tool_executions[0].status == "found"
    assert "Platform Engineering Team" in follow_up.response.answer


def test_incident_context_isolated_between_sessions():
    chunks, index, graph = _kb()
    store = InMemorySessionStore()
    first = ask_agentic("What is the current status of INC-001?", chunks, index, graph)
    store.record("known", "What is the current status of INC-001?", first.response, graph, ("INC-001",))

    isolated = ask_agentic(
        "Who owns it?",
        chunks,
        index,
        graph,
        session_context=store.get("different"),
    )

    assert isolated.trace.tool_executions == []
    assert isolated.operational_data == []