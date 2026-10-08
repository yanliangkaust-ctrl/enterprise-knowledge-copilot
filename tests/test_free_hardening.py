from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
import pytest
from backend import api
from backend.session_context import InMemorySessionStore
from backend.request_limits import ClientRateLimiter
from backend.rag_schema import GroundedResponse


def response():
    return GroundedResponse(answer="Supported", grounding_status="Grounded in retrieved evidence", generation_mode="Demo / Local")


def test_question_and_session_boundaries():
    assert len(api.QueryRequest(question="x" * 2000).question) == 2000
    client = TestClient(api.app)
    for body in ({"question":"x" * 2001}, {"question":"hello", "session_id":"x" * 129}):
        result = client.post('/query', json=body)
        assert result.status_code == 422 and result.headers['x-request-id']


def test_rate_limit_ids_and_health_exemption():
    client = TestClient(api.app)
    ids = set()
    for _ in range(30):
        result = client.post('/query', json={"question":""})
        assert result.status_code == 422
        ids.add(result.headers['x-request-id'])
    result = client.post('/query', json={"question":"PRIVATE_QUESTION"})
    assert result.status_code == 429 and int(result.headers['retry-after']) > 0
    assert result.headers['x-request-id'] not in ids
    assert 'PRIVATE_QUESTION' not in result.text
    assert client.get('/health').status_code == 200


def test_forwarded_header_does_not_bypass_limit(monkeypatch, caplog):
    monkeypatch.setattr(api.query_limiter, 'limit', 1)
    client = TestClient(api.app)
    assert client.post('/query', json={"question":""}, headers={"X-Forwarded-For":"198.51.100.1"}).status_code == 422
    result = client.post('/query', json={"question":"PRIVATE_QUESTION"}, headers={"X-Forwarded-For":"198.51.100.2"})
    assert result.status_code == 429
    assert 'PRIVATE_QUESTION' not in caplog.text


def test_limit_independent_clients_expiry_and_capacity():
    now = [0]
    limiter = ClientRateLimiter(limit=1, max_clients=2, clock=lambda:now[0])
    assert limiter.check('a')[0] and limiter.check('b')[0]
    assert not limiter.check('a')[0] and not limiter.check('c')[0]
    assert len(limiter._clients) == 2
    now[0] = 60
    assert limiter.check('c')[0]


def test_rate_limit_concurrent_bound():
    limiter = ClientRateLimiter(limit=5)
    with ThreadPoolExecutor(8) as executor:
        assert sum(executor.map(lambda _:limiter.check('a')[0],range(40))) == 5


def test_sessions_lru_ttl_and_followup():
    now = [0]
    store = InMemorySessionStore(max_sessions=2, ttl_seconds=10, clock=lambda:now[0])
    store.record('a', 'First question', response())
    now[0] = 1
    store.record('b', 'Second question', response())
    now[0] = 2
    assert store.get('a').recent_turns[0].question == 'First question'
    store.record('c', 'Third question', response())
    assert not store.get('b').recent_turns
    assert len(store._sessions) == 2
    now[0] = 12
    assert not store.get('a').recent_turns and not store._sessions
    assert not store._accessed


def test_concurrent_session_bound():
    store = InMemorySessionStore(max_sessions=3)
    with ThreadPoolExecutor(8) as executor:
        list(executor.map(lambda i:store.record(str(i),'Question',response()),range(40)))
    assert len(store._sessions) == len(store._accessed) == 3


def test_empty_store_auto_reconstructs_eight_docs(tmp_path, monkeypatch):
    from backend.knowledge_base import create_shared_knowledge_base
    from scripts.validate_container import DEMO_FILES
    monkeypatch.setenv('KNOWLEDGE_EMBEDDING_BACKEND','hash')
    for directory in (tmp_path/'first', tmp_path/'replacement'):
        monkeypatch.setenv('KNOWLEDGE_STORE_DIR',str(directory))
        base = create_shared_knowledge_base()
        assert {d.name for d in base.documents} == DEMO_FILES
        assert all(d.metadata['version']==1 for d in base.documents)
        assert base.readiness_status()[0]
