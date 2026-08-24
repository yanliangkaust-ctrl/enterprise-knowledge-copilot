"""Local semantic retrieval over document chunks.

The preferred backend is sentence-transformers. A deterministic NumPy fallback keeps
unit tests and offline demos usable when the model package is not installed yet.
"""

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

import numpy as np

from backend.document_ingestion import ChunkRecord


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbeddingModel:
    """Generate local embeddings, preferring a sentence-transformer model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimensions: int = 384,
        use_sentence_transformer: bool = True,
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.backend = "numpy-hash-fallback"
        self._model = None
        if use_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
            except (ImportError, OSError, RuntimeError):
                self._model = None

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        """Return L2-normalized vectors for one string or a sequence of strings."""

        values = [texts] if isinstance(texts, str) else list(texts)
        if self._model is not None:
            vectors = self._model.encode(values, convert_to_numpy=True, normalize_embeddings=True)
            return np.atleast_2d(vectors).astype(np.float32)

        vectors = np.vstack([self._fallback_vector(value) for value in values])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def _fallback_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return vector


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked chunk returned by semantic retrieval."""

    chunk: ChunkRecord
    score: float
    rank: int


class VectorIndex:
    """A small in-memory cosine-similarity index suitable for a local MVP."""

    def __init__(self, embedding_model: LocalEmbeddingModel | None = None):
        self.embedding_model = embedding_model or LocalEmbeddingModel()
        self.chunks: list[ChunkRecord] = []
        self._vectors = np.empty((0, self.embedding_model.dimensions), dtype=np.float32)

    def build(self, chunks: Sequence[ChunkRecord]) -> None:
        self.chunks = list(chunks)
        if self.chunks:
            self._vectors = self.embedding_model.encode([chunk.text for chunk in self.chunks])
        else:
            self._vectors = np.empty((0, self.embedding_model.dimensions), dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return at most top_k chunks ranked by cosine similarity."""

        if not self.chunks or top_k <= 0 or not query.strip():
            return []
        query_vector = self.embedding_model.encode(query)[0]
        scores = self._vectors @ query_vector
        ranked_indexes = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indexes, start=1)
        ]
