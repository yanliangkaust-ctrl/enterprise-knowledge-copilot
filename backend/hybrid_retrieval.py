"""Explainable hybrid vector plus graph retrieval."""

from dataclasses import dataclass
from collections.abc import Sequence

from backend.document_ingestion import ChunkRecord
from backend.factual_support import evidence_supports_facet, requested_facets
from backend.rag_schema import EvidenceItem
from backend.resilience import ServiceFailure, component, degraded
from backend.observability import emit, timed
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


@timed("hybrid_selection")
def _preserve_factual_coverage(question: str, ranked: list[dict], top_k: int) -> list[dict]:
    """Reserve best-ranked supporting chunks, then fill in original score order.

Support uses the refusal gate's unchanged predicate. Requests are processed in
question order; if distinct required chunks exceed K, the first K are retained
and the gate still refuses uncovered facets. No additional candidates are fetched.
"""
    if top_k <= 0:
        return []
    selected = []
    selected_ids = set()
    evidence = {}
    for item in ranked:
        chunk = item["chunk"]
        evidence[chunk.chunk_id] = EvidenceItem(
            rank=0, source_document=chunk.document_name, section=chunk.section_heading,
            chunk_id=chunk.chunk_id, text=chunk.text, score=0,
        )
    for facet, anchors in requested_facets(question):
        if any(evidence_supports_facet(facet, anchors, [evidence[item["chunk"].chunk_id]]) for item in selected):
            continue
        support = next((item for item in ranked
                        if evidence_supports_facet(facet, anchors, [evidence[item["chunk"].chunk_id]])), None)
        if support is not None and len(selected) < top_k:
            selected.append({**support, "reasons": [*support["reasons"], f"Preserved requested {facet} support."]})
            selected_ids.add(support["chunk"].chunk_id)
    final = (selected + [item for item in ranked if item["chunk"].chunk_id not in selected_ids])[:top_k]
    emit("hybrid_selection", merged_candidate_count=len(ranked), selected_count=len(final), reserved_count=len(selected))
    return final


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

    failures = []
    try:
        vector_results = component(lambda: vector_index.search(question, top_k=max(top_k, 10)), "vector_retrieval_failed")
    except ServiceFailure as exc:
        failures.append(("vector", exc.code))
        vector_results = []
    try:
        graph_results = component(lambda: search_graph(question, graph, chunks, top_k=max(top_k, 10)), "graph_retrieval_failed")
    except ServiceFailure as exc:
        failures.append(("graph", exc.code))
        graph_results = []
    if failures:
        if len(failures) == 2 or not (vector_results or graph_results):
            raise ServiceFailure("hybrid_retrieval_failed", 503)
        degraded(failures[0][1], failures[0][0], "hybrid_search")
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
    )
    emit("hybrid_candidates", retriever="hybrid_search", vector_candidate_count=len(vector_results),
         graph_candidate_count=len(graph_results), merged_candidate_count=len(merged))
    ranked = _preserve_factual_coverage(question, ranked, top_k)
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
