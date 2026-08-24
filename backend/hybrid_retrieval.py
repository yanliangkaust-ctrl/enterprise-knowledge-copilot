"""Explainable hybrid vector plus graph retrieval."""

from dataclasses import dataclass
from collections.abc import Sequence

from backend.document_ingestion import ChunkRecord
from backend.graph_retrieval import GraphRetrievalResult, search_graph
from backend.knowledge_graph import KnowledgeGraph
from backend.semantic_retrieval import RetrievalResult, VectorIndex


@dataclass(frozen=True)
class HybridRetrievalResult:
    """A merged result retaining vector and graph selection evidence."""

    chunk: ChunkRecord
    score: float
    rank: int
    vector_score: float
    graph_score: float
    reason: str


def search_hybrid(
    question: str,
    chunks: Sequence[ChunkRecord],
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    top_k: int = 5,
    vector_weight: float = 0.7,
    graph_weight: float = 0.3,
) -> list[HybridRetrievalResult]:
    """Merge vector and graph evidence by chunk ID with transparent weighted scores."""

    vector_results = vector_index.search(question, top_k=max(top_k, 10))
    graph_results = search_graph(question, graph, chunks, top_k=max(top_k, 10))
    merged: dict[str, dict] = {}
    for result in vector_results:
        merged[result.chunk.chunk_id] = {
            "chunk": result.chunk,
            "vector_score": result.score,
            "graph_score": 0.0,
            "reasons": [f"Vector similarity {result.score:.3f}"],
        }
    for result in graph_results:
        item = merged.setdefault(result.chunk.chunk_id, {
            "chunk": result.chunk,
            "vector_score": 0.0,
            "graph_score": 0.0,
            "reasons": [],
        })
        item["graph_score"] = max(item["graph_score"], result.score)
        item["reasons"].append(result.reason)

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -(vector_weight * item["vector_score"] + graph_weight * item["graph_score"]),
            item["chunk"].chunk_id,
        ),
    )[:top_k]
    return [
        HybridRetrievalResult(
            chunk=item["chunk"],
            score=vector_weight * item["vector_score"] + graph_weight * item["graph_score"],
            rank=rank,
            vector_score=item["vector_score"],
            graph_score=item["graph_score"],
            reason="; ".join(item["reasons"]),
        )
        for rank, item in enumerate(ranked, start=1)
    ]
