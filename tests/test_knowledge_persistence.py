import subprocess
import sys
from uuid import UUID

import pytest

from backend.knowledge_base import KnowledgeBase, create_shared_knowledge_base
from backend.semantic_retrieval import LocalEmbeddingModel
from backend.rag_pipeline import ask_knowledge


def kb(path):
    return KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False), path)


def test_restart_identity_versions_and_noop(tmp_path):
    path = tmp_path / 'knowledge.sqlite3'
    first = kb(path)
    document = first.ingest_markdown('policy.md', '# Policy\nQuartz replication runs every 17 minutes.')
    UUID(document.metadata['document_id'])
    revision = first.revision
    first.ingest_markdown('policy.md', document.raw_text)
    assert first.revision == revision
    restarted = kb(path)
    assert restarted.documents == first.documents
    assert restarted.chunks == first.chunks
    changed = restarted.ingest_markdown('policy.md', '# Policy\nQuartz replication runs every 23 minutes.')
    assert changed.metadata['document_id'] == document.metadata['document_id']
    assert changed.metadata['version'] == 2
    assert changed.metadata['content_hash'] != document.metadata['content_hash']
    assert restarted.revision == revision + 1
    assert kb(path).documents == restarted.documents
    with restarted.store.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM document_versions').fetchone()[0] == 2


def test_unrelated_chunk_ids_and_live_service_refresh(tmp_path):
    path = tmp_path / 'knowledge.sqlite3'
    writer, reader = kb(path), kb(path)
    writer.ingest_markdown('first.md', '# First\nOriginal text.')
    writer.ingest_markdown('second.md', '# Second\nStable quartz policy.')
    stable = [c.chunk_id for c in writer.chunks if c.document_name == 'second.md']
    writer.ingest_markdown('first.md', '# First\nChanged.\n## Extra\nAdditional section.')
    documents, chunks, _, _ = reader.snapshot()
    assert documents == writer.documents
    assert [c.chunk_id for c in chunks if c.document_name == 'second.md'] == stable
    assert all('::v' in c.chunk_id for c in chunks)


@pytest.mark.parametrize('mode', ['Keyword', 'Semantic', 'Graph', 'Hybrid'])
def test_retrieval_before_after_restart(tmp_path, mode):
    first = kb(tmp_path / 'knowledge.sqlite3')
    first.ingest_markdown('architecture.md', '# Ownership\nPlatform Engineering Team owns Kubernetes.')
    def ask(base):
        return ask_knowledge('Who owns Kubernetes?', base.chunks, base.vector_index, base.knowledge_graph, retrieval_mode=mode)
    assert ask(first) == ask(kb(first.store.path))
    assert ask(first).retrieved_evidence


def test_shared_factory_and_api_observe_ui_ingestion(tmp_path, monkeypatch):
    from backend import api
    monkeypatch.setenv('KNOWLEDGE_STORE_DIR', str(tmp_path))
    monkeypatch.setenv('KNOWLEDGE_EMBEDDING_BACKEND', 'hash')
    ui = create_shared_knowledge_base()
    service = create_shared_knowledge_base()
    monkeypatch.setattr(api, 'get_shared_knowledge_base', lambda: service)
    ui.ingest_markdown('quartz.md', '# Quartz\nQuartz replication runs every 17 minutes.')
    assert 'quartz.md' in {d.name for d in api.build_knowledge_base()[0]}
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        response = client.post('/query', json={'question': 'How often does quartz replication run?'})
    assert response.status_code == 200
    assert 'quartz.md' in response.json()['sources']
    assert response.json()['grounding_status'] == 'Grounded in retrieved evidence'
    assert ui.vector_index.embedding_model.backend == service.vector_index.embedding_model.backend
    updated = ui.ingest_markdown('ADR_Deployment.md', '# Updated\nCustom release policy.')
    restarted = create_shared_knowledge_base()
    assert next(d for d in restarted.documents if d.name == updated.name) == updated


def test_fresh_process_reload(tmp_path):
    path = tmp_path / 'knowledge.sqlite3'
    first = kb(path)
    first.ingest_markdown('quartz.md', '# Quartz\nDurable quartz policy.')
    code = "from backend.knowledge_base import KnowledgeBase; from backend.semantic_retrieval import LocalEmbeddingModel; import sys; k=KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False),sys.argv[1]); print(k.documents[0].metadata['document_id'])"
    result = subprocess.run([sys.executable, '-B', '-c', code, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == first.documents[0].metadata['document_id']


def test_index_failure_rolls_back(tmp_path, monkeypatch):
    first = kb(tmp_path / 'knowledge.sqlite3')
    first.ingest_markdown('policy.md', '# Policy\nOriginal.')
    before = first.snapshot()
    revision = first.revision
    def fail(*args):
        raise RuntimeError('index failure')
    monkeypatch.setattr(first, '_prepare', fail)
    with pytest.raises(RuntimeError, match='index failure'):
        first.ingest_markdown('policy.md', '# Policy\nChanged.')
    restarted = kb(first.store.path)
    assert restarted.revision == revision
    assert restarted.documents == before[0]
    assert first.documents == before[0]
