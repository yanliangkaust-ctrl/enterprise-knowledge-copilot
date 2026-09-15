import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


client = TestClient(api.app)


@pytest.fixture
def deterministic_knowledge_base(monkeypatch):
    documents = load_markdown_documents(api.SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)
    vector_index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    vector_index.build(chunks)
    knowledge_graph = build_knowledge_graph(chunks)
    monkeypatch.setattr(
        api,
        "build_knowledge_base",
        lambda: (documents, chunks, vector_index, knowledge_graph),
    )


def test_health_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "enterprise-knowledge-copilot",
    }


def test_query_exposes_grounded_answer_route_provenance_trace_and_session(deterministic_knowledge_base):
    response = client.post(
        "/query",
        json={
            "question": "What depends on Kubernetes?",
            "session_id": "session-test-001",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-test-001"
    assert payload["answer"]
    assert payload["selected_retrieval_route"] in {"graph_search", "hybrid_search"}
    assert payload["evidence_sufficient"] is True
    assert payload["sources"]
    assert payload["provenance"]
    assert payload["execution_trace"]["tool_decisions"]
    assert payload["execution_trace"]["retrieved_evidence_count"] == len(payload["provenance"])


def test_query_preserves_insufficient_evidence_behavior(deterministic_knowledge_base):
    response = client.post(
        "/query",
        json={"question": "What is the company's 2028 international expansion budget?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_sufficient"] is False
    assert payload["grounding_status"] == "Insufficient evidence"
    assert "Insufficient evidence" in payload["answer"]
    assert payload["sources"] == []
    assert payload["session_id"]


def test_query_rejects_blank_questions(deterministic_knowledge_base):
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_follow_up_uses_relevant_context_in_same_session(deterministic_knowledge_base):
    session_id = "session-follow-up"
    first = client.post(
        "/query",
        json={
            "question": "What is the impact of Kubernetes capacity during OCR batch peaks?",
            "session_id": session_id,
        },
    )
    second = client.post(
        "/query",
        json={"question": "Who owns that risk?", "session_id": session_id},
    )

    assert first.status_code == second.status_code == 200
    payload = second.json()
    assert payload["session_id"] == session_id
    assert payload["evidence_sufficient"] is True
    assert "Platform Engineering Team" in payload["answer"]


def test_different_sessions_do_not_share_follow_up_context(deterministic_knowledge_base):
    first = client.post(
        "/query",
        json={
            "question": "What is the impact of Kubernetes capacity during OCR batch peaks?",
            "session_id": "session-isolated-source",
        },
    )
    unrelated = client.post(
        "/query",
        json={"question": "Who owns that risk?", "session_id": "session-isolated-target"},
    )

    assert first.status_code == unrelated.status_code == 200
    payload = unrelated.json()
    assert payload["evidence_sufficient"] is False
    assert payload["grounding_status"] == "Insufficient evidence"


def test_insufficient_context_does_not_become_authoritative_evidence(deterministic_knowledge_base):
    session_id = "session-insufficient-context"
    client.post(
        "/query",
        json={
            "question": "What is the impact of Kubernetes capacity during OCR batch peaks?",
            "session_id": session_id,
        },
    )
    response = client.post(
        "/query",
        json={
            "question": "What is the company's 2028 international expansion budget?",
            "session_id": session_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_sufficient"] is False
    assert payload["grounding_status"] == "Insufficient evidence"
    assert payload["sources"] == []


def test_api_exposes_operational_tool_trace_and_data(deterministic_knowledge_base):
    response = client.post(
        "/query",
        json={"question": "What is the current status of INC-001?", "session_id": "api-incident"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["operational_data"][0]["incident_id"] == "INC-001"
    assert payload["execution_trace"]["tool_executions"][0]["tool"] == "get_incident_status"


def test_api_incident_follow_up_uses_same_session(deterministic_knowledge_base):
    session_id = "api-incident-follow-up"
    client.post("/query", json={"question": "What is the current status of INC-001?", "session_id": session_id})
    response = client.post("/query", json={"question": "Who owns it?", "session_id": session_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_sufficient"] is True
    assert "Platform Engineering Team" in payload["answer"]