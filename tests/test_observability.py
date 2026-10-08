from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from backend import api
from backend.knowledge_base import KnowledgeBase
from backend.observability import JsonFormatter, emit, logger
from backend.semantic_retrieval import LocalEmbeddingModel


@pytest.fixture
def events(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    monkeypatch.setattr(logger, "handlers", [handler])
    return stream


def rows(events):
    return [json.loads(line) for line in events.getvalue().splitlines()]


@pytest.fixture
def base(monkeypatch):
    base = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False))
    base.ingest_markdown("private.md", "# Private\nThe platform will deploy the OCR Service and API Gateway on Kubernetes.\n"
                         "## Validation\nThe UAT Environment validates release readiness.")
    monkeypatch.setattr(api, "get_shared_knowledge_base", lambda: base)
    monkeypatch.delattr(api.app.state, "knowledge_base", raising=False)
    return base


def test_ids_on_success_validation_failure_and_concurrent_requests(events, base):
    client = TestClient(api.app)
    def request(i):
        if i % 2:
            return client.post("/query", json={"question": "What protocol does Kubernetes use?", "session_id": "same"},
                               headers={"X-Request-ID": "untrusted-client-id"})
        return client.get("/health")
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(request, range(8)))
    responses.append(client.post("/query", json={"question": ""}))
    ids = [response.headers["X-Request-ID"] for response in responses]
    assert len(set(ids)) == len(ids)
    assert all(UUID(identifier).version == 4 for identifier in ids)
    assert responses[-1].status_code == 422
    events_by_id = {identifier: [r for r in rows(events) if r["request_id"] == identifier] for identifier in ids}
    for response, identifier in zip(responses, ids):
        rs = events_by_id[identifier]
        assert len([r for r in rs if r["event"] == "request_started"]) == 1
        completion = [r for r in rs if r["event"] == "request_completed"]
        assert len(completion) == 1
        assert completion[0]["http_status"] == response.status_code
        assert completion[0]["duration_ms"] >= 0
        if completion[0].get("original_route"):
            assert any(r["event"] == "sufficiency_check" for r in rs)
            assert any(r["stage"] == "draft_generation" for r in rs)


def test_allowlist_json_and_no_content_logging(events):
    emit("test_event", question="QUERY_SECRET", answer="ANSWER_SECRET", text="DOCUMENT_SECRET",
         provenance={"text": "SECRET"}, exception="EXCEPTION_SECRET", duration_ms=1.25)
    r = rows(events)[0]
    assert r["duration_ms"] == 1.25
    assert {"event", "timestamp", "request_id", "stage", "status"} <= r.keys()
    assert "SECRET" not in events.getvalue()


def test_fallback_refusal_codes_and_timing_fields(events, base):
    client = TestClient(api.app)
    response = client.post("/query", json={"question": "What protocol does Kubernetes use?"})
    rs = [r for r in rows(events) if r["request_id"] == response.headers["X-Request-ID"]]
    assert response.json()["grounding_status"] == "Insufficient evidence"
    assert any(r["event"] == "graph_fallback" and r["reason_code"] == "graph_missing_fact_support" for r in rs)
    assert any(r["event"] == "sufficiency_check" and r["reason_code"] == "missing_factual_support" for r in rs)
    stages = {r["stage"] for r in rs if r["event"] == "stage_completed"}
    assert {"knowledge_snapshot", "context", "routing", "primary_retrieval", "fallback_retrieval",
            "hybrid_selection", "factual_support", "evidence_sufficiency", "draft_generation",
            "orchestration", "refusal_finalization", "trace_finalization"} <= stages
    for r in rs:
        if "duration_ms" in r:
            assert isinstance(r["duration_ms"], (int, float)) and r["duration_ms"] >= 0
    assert any(r["event"] == "request_completed" and r["knowledge_revision"] == base.revision for r in rs)


def test_error_is_sanitized_and_completion_has_id(events, monkeypatch):
    def fail():
        raise RuntimeError("QUERY_SECRET DOCUMENT_SECRET PASSWORD_SECRET")
    monkeypatch.setattr(api, "build_knowledge_base", fail)
    response = TestClient(api.app).post("/query", json={"question": "QUERY_SECRET"})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "SECRET" not in events.getvalue()
    rs = [r for r in rows(events) if r["request_id"] == response.headers["X-Request-ID"]]
    assert any(r["event"] == "application_error" and r["exception_type"] == "RuntimeError" for r in rs)
    assert any(r["event"] == "request_completed" and r["http_status"] == 500 for r in rs)


def test_health_never_initializes_knowledge_and_ready_without_base_is_503(monkeypatch, events):
    monkeypatch.delattr(api.app.state, "knowledge_base", raising=False)
    monkeypatch.setattr(api, "get_shared_knowledge_base", lambda: pytest.fail("health/readiness initialized knowledge"))
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok", "service": "enterprise-knowledge-copilot"}
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["reason_code"] == "knowledge_uninitialized"


def test_ready_checks_store_and_staleness_without_rebuild(tmp_path, monkeypatch, events):
    path = tmp_path / "knowledge.sqlite3"
    base = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), path)
    monkeypatch.setattr(api.app.state, "knowledge_base", base, raising=False)
    monkeypatch.setattr(base, "_prepare", lambda *args: pytest.fail("readiness rebuilt indexes"))
    monkeypatch.setattr(base, "snapshot", lambda: pytest.fail("readiness refreshed knowledge"))
    client = TestClient(api.app)
    assert client.get("/ready").status_code == 200
    writer = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), path)
    writer.ingest_markdown("policy.md", "# Private\nDOCUMENT_SECRET")
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["reason_code"] == "knowledge_stale"
    monkeypatch.setattr(base.store, "path", tmp_path / "missing.sqlite3")
    assert client.get("/ready").json()["reason_code"] == "store_unavailable"
    assert not base.store.path.exists()
    assert "DOCUMENT_SECRET" not in events.getvalue()


def test_reload_revision_events_and_failure_sanitization(tmp_path, monkeypatch, events):
    path = tmp_path / "knowledge.sqlite3"
    writer = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), path)
    reader = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), path)
    previous = reader.revision
    writer.ingest_markdown("policy.md", "# Private\nDOCUMENT_SECRET")
    reader.snapshot()
    completed = [r for r in rows(events) if r["event"] == "knowledge_reload" and r["status"] == "completed"][-1]
    assert completed["previous_revision"] == previous
    assert completed["new_revision"] == reader.revision == writer.revision
    assert completed["document_count"] == completed["chunk_count"] == 1
    assert completed["duration_ms"] >= 0
    writer.ingest_markdown("policy.md", "# Private\nUPDATED_SECRET")
    def fail(*args):
        raise RuntimeError("DOCUMENT_SECRET")
    monkeypatch.setattr(reader, "_prepare", fail)
    with pytest.raises(RuntimeError):
        reader.snapshot()
    assert any(r["event"] == "knowledge_reload" and r["status"] == "error" for r in rows(events))
    assert "SECRET" not in events.getvalue()


def test_index_unavailable_is_not_ready(base, events):
    base.vector_index.chunks = []
    available, code, _ = base.readiness_status()
    assert not available and code == "index_unavailable"
