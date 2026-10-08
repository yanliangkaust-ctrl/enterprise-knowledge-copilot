"""Shared Markdown knowledge lifecycle with optional local SQLite persistence."""
from collections.abc import Sequence
from contextlib import closing
from functools import lru_cache
import os
import sqlite3
from pathlib import Path
from threading import RLock
from time import perf_counter

from backend.document_ingestion import DocumentRecord, chunk_documents, load_markdown_documents, parse_markdown_content
from backend.graph_builder import build_knowledge_graph
from backend.knowledge_store import KnowledgeStore
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex
from backend.resilience import boundary
from backend.observability import emit, record_metadata, stage, timed


class KnowledgeBase:
    @timed("knowledge_initialization")
    def __init__(self, embedding_model=None, store_path=None):
        self._lock = RLock()
        self.store = KnowledgeStore(store_path) if store_path is not None else None
        self.failure_code = None
        self.revision = -1
        self._snapshot = ([], [], VectorIndex(embedding_model), build_knowledge_graph([]))
        if self.store:
            self.reload_if_changed()
        emit("knowledge_initialized", knowledge_revision=self.revision,
             document_count=len(self.documents), chunk_count=len(self.chunks),
             embedding_backend=self.vector_index.embedding_model.backend)

    @property
    def documents(self):
        return self._snapshot[0]
    @property
    def chunks(self):
        return self._snapshot[1]
    @property
    def vector_index(self):
        return self._snapshot[2]
    @property
    def knowledge_graph(self):
        return self._snapshot[3]

    def _prepare(self, documents, revision):
        if revision == self.revision:
            return self._snapshot
        chunks = chunk_documents(documents)
        index = VectorIndex(self.vector_index.embedding_model)
        index.build(chunks)
        return documents, chunks, index, build_knowledge_graph(chunks)

    @timed("knowledge_snapshot")
    def snapshot(self):
        with self._lock:
            self.reload_if_changed()
            record_metadata(knowledge_revision=self.revision)
            emit("knowledge_snapshot", knowledge_revision=self.revision,
                 document_count=len(self.documents), chunk_count=len(self.chunks),
                 embedding_backend=self.vector_index.embedding_model.backend)
            return self._snapshot

    def reload_if_changed(self):
        with self._lock:
            if not self.store:
                return
            try:
                with boundary("storage_revision_failed", 503):
                    committed = self.store.revision()
            except Exception:
                self.failure_code = "storage_revision_failed"
                raise
            if committed == self.revision and self.failure_code == "storage_revision_failed":
                self.failure_code = None
            if committed != self.revision:
                previous = self.revision
                started = perf_counter()
                emit("knowledge_reload", status="started", previous_revision=previous)
                try:
                    with boundary("knowledge_reload_failed", 503), stage("knowledge_reload"):
                        revision, documents = self.store.read()
                        snapshot = self._prepare(documents, revision)
                        self._snapshot, self.revision = snapshot, revision
                        self.failure_code = None
                        emit("knowledge_reload", status="completed", previous_revision=previous,
                             duration_ms=max(0.0, (perf_counter() - started) * 1000),
                             new_revision=revision, document_count=len(documents), chunk_count=len(snapshot[1]),
                             embedding_backend=snapshot[2].embedding_model.backend)
                except Exception as exc:
                    self.failure_code = "knowledge_reload_failed"
                    emit("knowledge_reload", status="error", previous_revision=previous,
                         duration_ms=max(0.0, (perf_counter() - started) * 1000),
                         exception_type=type(exc).__name__, error_code="knowledge_reload_failed")
                    raise

    def readiness_status(self):
        """Bounded, read-only probe: never initializes or rebuilds knowledge."""
        if not self._lock.acquire(timeout=0.05):
            return False, "knowledge_busy", {}
        try:
            if self.failure_code:
                return False, self.failure_code, {"knowledge_revision": self.revision}
            documents, chunks, index, graph = self._snapshot
            available = (len(index.chunks) == len(chunks) and index._vectors.shape[0] == len(chunks) and graph is not None)
            metadata = {"knowledge_revision": self.revision, "document_count": len(documents),
                        "chunk_count": len(chunks), "index_available": available}
            if not available:
                return False, "index_unavailable", metadata
            if self.store:
                with closing(sqlite3.connect(self.store.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)) as connection:
                    row = connection.execute("SELECT revision FROM knowledge_revision WHERE singleton=1").fetchone()
                if row is None:
                    return False, "store_unavailable", metadata
                metadata["store_revision"] = row[0]
                if self.revision != row[0]:
                    return False, "knowledge_stale", metadata
            return True, "ready", metadata
        except (sqlite3.Error, OSError):
            return False, "store_unavailable", {}
        finally:
            self._lock.release()

    def add_documents(self, documents: Sequence[DocumentRecord], only_missing=False):
        with self._lock:
            if self.store:
                revision, snapshot = self.store.update(documents, self._prepare, only_missing)
                self._snapshot, self.revision = snapshot, revision
                self.failure_code = None
            else:
                by_name = {document.name: document for document in self.documents}
                for document in documents:
                    if not only_missing or document.name not in by_name:
                        by_name[document.name] = document
                self._publish_memory(list(by_name.values()))

    def ingest_markdown(self, filename, raw_text):
        if Path(filename).suffix.casefold() != '.md':
            raise ValueError('Only Markdown (.md) files are supported.')
        document = parse_markdown_content(Path(filename).name, raw_text)
        self.add_documents([document])
        return next(item for item in self.documents if item.name == document.name)

    def refresh(self):
        with self._lock:
            if self.store:
                revision, documents = self.store.read()
                chunks = chunk_documents(documents)
                index = VectorIndex(self.vector_index.embedding_model)
                index.build(chunks)
                self._snapshot = (documents, chunks, index, build_knowledge_graph(chunks))
                self.revision = revision
            else:
                self._publish_memory(self.documents)

    def _publish_memory(self, documents):
        chunks = chunk_documents(documents)
        replacement = VectorIndex(self.vector_index.embedding_model)
        replacement.build(chunks)
        graph = build_knowledge_graph(chunks)
        # Publish only after all fallible preparation; retain the public index identity.
        index = self.vector_index
        index._state = replacement._state
        self._snapshot = (documents, chunks, index, graph)



@timed("knowledge_initialization")
def create_shared_knowledge_base():
    """Both application entry points use this configuration and seed policy."""
    default = Path(os.getenv('LOCALAPPDATA', str(Path.home() / '.local' / 'share'))) / 'EnterpriseKnowledgeCopilot'
    directory = Path(os.getenv('KNOWLEDGE_STORE_DIR', str(default))).expanduser().resolve()
    backend = os.getenv('KNOWLEDGE_EMBEDDING_BACKEND', 'hash')
    if backend not in {'hash', 'sentence-transformers'}:
        raise ValueError('KNOWLEDGE_EMBEDDING_BACKEND must be hash or sentence-transformers')
    model = LocalEmbeddingModel(use_sentence_transformer=backend == 'sentence-transformers')
    if backend == 'sentence-transformers' and model.backend != backend:
        raise RuntimeError('Configured sentence-transformer model is unavailable; refusing silent fallback.')
    knowledge_base = KnowledgeBase(model, directory / 'knowledge.sqlite3')
    knowledge_base.add_documents(load_markdown_documents(Path(__file__).resolve().parents[1] / 'data' / 'sample_docs'), only_missing=True)
    return knowledge_base


@lru_cache(maxsize=1)
def get_shared_knowledge_base():
    return create_shared_knowledge_base()
