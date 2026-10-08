from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import api, hybrid_retrieval as hybrid, langgraph_orchestrator as orchestration
from backend.knowledge_base import KnowledgeBase
from backend.observability import logger, JsonFormatter
from backend.resilience import ServiceFailure
from backend.semantic_retrieval import LocalEmbeddingModel


def fail(*args, **kwargs):
    raise RuntimeError("PRIVATE_QUERY_DOCUMENT_SECRET")


@pytest.fixture
def base(monkeypatch):
    base = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False))
    base.ingest_markdown("policy.md", "# Ownership\nPlatform Engineering Team owns Kubernetes.\n")
    monkeypatch.setattr(api, "get_shared_knowledge_base", lambda: base)
    return base


@pytest.fixture
def events(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    monkeypatch.setattr(logger, "handlers", [handler])
    return stream


@pytest.mark.parametrize("target", ["chunk_documents", "encode", "build_knowledge_graph"])
def test_memory_atomic_failure(base, monkeypatch, target):
    import backend.knowledge_base as module
    before = base.snapshot()
    vectors = base.vector_index._vectors.copy()
    with monkeypatch.context() as patch:
        if target == "encode":
            patch.setattr(base.vector_index.embedding_model, target, fail)
        else:
            patch.setattr(module, target, fail)
        with pytest.raises(RuntimeError):
            base.ingest_markdown("new.md", "# New\nNew evidence")
    assert base.snapshot() == before
    assert base.vector_index is before[2]
    assert np.array_equal(base.vector_index._vectors, vectors)
    assert base.vector_index.search("Kubernetes")


def test_index_build_is_atomic(base, monkeypatch):
    index = base.vector_index
    old_chunks, old_vectors = index.chunks, index._vectors
    monkeypatch.setattr(index.embedding_model, "encode", fail)
    with pytest.raises(RuntimeError):
        index.build(index.chunks * 2)
    assert index.chunks is old_chunks and index._vectors is old_vectors


@pytest.mark.parametrize("commit", [False, True])
def test_persistent_rollback(tmp_path, monkeypatch, commit):
    base = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), tmp_path / "kb.db")
    base.ingest_markdown("one.md", "# One\nOriginal")
    before = base.snapshot()
    revision = base.revision
    if commit:
        from contextlib import contextmanager
        original = base.store.connect
        @contextmanager
        def broken():
            with original() as connection:
                yield connection
                raise sqlite3.OperationalError("PRIVATE_COMMIT_SECRET")
        monkeypatch.setattr(base.store, "connect", broken)
    else:
        monkeypatch.setattr(base, "_prepare", fail)
    with pytest.raises(Exception):
        base.ingest_markdown("one.md", "# One\nChanged")
    restored = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), base.store.path)
    assert restored.revision == revision
    assert restored.documents == before[0] and base._snapshot == before


def test_reload_failure_concurrent_and_recovery(tmp_path, monkeypatch, events):
    model = LocalEmbeddingModel(use_sentence_transformer=False)
    reader = KnowledgeBase(model, tmp_path / "kb.db")
    writer = KnowledgeBase(model, reader.store.path)
    before = reader.snapshot()
    writer.ingest_markdown("one.md", "# One\nCommitted")
    monkeypatch.setattr(api, "get_shared_knowledge_base", lambda: reader)
    with monkeypatch.context() as patch:
        patch.setattr(reader, "_prepare", fail)
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(lambda _: TestClient(api.app).post("/query", json={"question": "One"}), range(2)))
        assert all(r.status_code == 503 for r in results)
        assert reader._snapshot == before
        assert TestClient(api.app).get("/ready").status_code == 503
    reader.snapshot()
    assert TestClient(api.app).get("/ready").status_code == 200
    assert "PRIVATE_QUERY_DOCUMENT_SECRET" not in events.getvalue()


@pytest.mark.parametrize("failed", ["vector", "graph"])
def test_hybrid_component_degradation(base, monkeypatch, events, failed):
    if failed == "vector":
        monkeypatch.setattr(base.vector_index, "search", fail)
    else:
        monkeypatch.setattr(hybrid, "search_graph", fail)
    results = hybrid.search_hybrid("Who owns Kubernetes?", base.chunks, base.vector_index, base.knowledge_graph)
    assert 0 < len(results) <= 5
    assert all(r.chunk in base.chunks for r in results)
    assert all(r.score == .7 * r.vector_score + .3 * r.graph_score for r in results)
    assert "retrieval_degraded" in events.getvalue()


def test_both_components_fail(base, monkeypatch):
    monkeypatch.setattr(base.vector_index, "search", fail)
    monkeypatch.setattr(hybrid, "search_graph", fail)
    with pytest.raises(ServiceFailure, match="hybrid_retrieval_failed"):
        hybrid.search_hybrid("Kubernetes", base.chunks, base.vector_index, base.knowledge_graph)


@pytest.mark.parametrize("target,code", [("generate_grounded_answer", "generation_failed"), ("_evidence_sufficient", "sufficiency_internal_error")])
def test_pipeline_failure_then_healthy(base, monkeypatch, events, target, code):
    client = TestClient(api.app)
    with monkeypatch.context() as patch:
        patch.setattr(orchestration, target, fail)
        response = client.post("/query", json={"question": "Who owns Kubernetes?", "session_id": "isolated"})
        assert response.status_code == 500
        assert not api.session_store.get("isolated").recent_turns
    assert client.post("/query", json={"question": "Who owns Kubernetes?"}).status_code == 200
    assert code in events.getvalue() and "PRIVATE_QUERY_DOCUMENT_SECRET" not in events.getvalue()


def test_response_construction_does_not_record(base, monkeypatch):
    monkeypatch.setattr(api, "QueryResponse", fail)
    response = TestClient(api.app).post("/query", json={"question": "Who owns Kubernetes?", "session_id": "construction"})
    assert response.status_code == 500
    assert not api.session_store.get("construction").recent_turns


def test_graph_exception_fallback_once(base, monkeypatch, events):
    original = orchestration._run_tool
    calls = []
    def run(tool, *args):
        calls.append(tool)
        if tool == "graph_search":
            return fail()
        return original(tool, *args)
    monkeypatch.setattr(orchestration, "_run_tool", run)
    response = TestClient(api.app).post("/query", json={"question": "How does Kubernetes relate to Platform Engineering Team?"})
    assert response.status_code == 200
    assert calls == ["graph_search", "semantic_search"]
    assert "retrieval_degraded" in events.getvalue()


def test_degraded_unsupported_fact_still_refuses(base, monkeypatch):
    monkeypatch.setattr(hybrid, "search_graph", fail)
    response = TestClient(api.app).post("/query", json={"question": "What is the Kubernetes license price?"})
    assert response.status_code == 200
    assert response.json()["evidence_sufficient"] is False


def test_vector_exception_graph_fallback_once(base, monkeypatch):
    monkeypatch.setattr(orchestration, "plan_tools", lambda *args: ("factual", [orchestration.ToolDecision("semantic_search", "test route")]))
    monkeypatch.setattr(base.vector_index, "search", fail)
    response = TestClient(api.app).post("/query", json={"question": "Who owns Kubernetes?"})
    assert response.status_code == 200
    assert response.json()["execution_trace"]["tool_decisions"][-1]["tool"] == "graph_search"


def test_fallback_failure_no_partial_answer(base, monkeypatch):
    calls = []
    def run(tool, *args):
        calls.append(tool)
        if tool == "graph_search":
            return []
        return fail()
    monkeypatch.setattr(orchestration, "_run_tool", run)
    response = TestClient(api.app).post("/query", json={"question": "How does Kubernetes relate to Platform Engineering Team?"})
    assert response.status_code == 503
    assert "answer" not in response.text and calls == ["graph_search", "hybrid_search"]


def test_revision_error_is_503(base, monkeypatch, tmp_path):
    persistent = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), tmp_path / "kb.db")
    monkeypatch.setattr(api, "get_shared_knowledge_base", lambda: persistent)
    monkeypatch.setattr(persistent.store, "revision", fail)
    response = TestClient(api.app).post("/query", json={"question": "Anything"})
    assert response.status_code == 503
    assert not persistent.readiness_status()[0]


def test_readiness_tracks_known_runtime_failure_and_recovery(base, monkeypatch):
    client = TestClient(api.app)
    with monkeypatch.context() as patch:
        patch.setattr(orchestration, "generate_grounded_answer", fail)
        assert client.post("/query", json={"question": "Who owns Kubernetes?"}).status_code == 500
        assert client.get("/ready").json()["reason_code"] == "generation_failed"
    assert client.post("/query", json={"question": "Who owns Kubernetes?"}).status_code == 200
    assert client.get("/ready").status_code == 200


def test_startup_sanitized(monkeypatch, events):
    monkeypatch.setattr(api, "build_knowledge_base", fail)
    with pytest.raises(ServiceFailure, match="startup_initialization_failed"):
        with TestClient(api.app):
            pass
    assert "PRIVATE_QUERY_DOCUMENT_SECRET" not in events.getvalue()
