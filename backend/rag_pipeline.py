"""Orchestration for retrieval, context-preserving conversion, and generation."""

from collections.abc import Sequence

from backend.answer_generation import evidence_items_from_results, generate_grounded_answer
from backend.document_ingestion import ChunkRecord, KeywordChunkResult, search_chunks
from backend.graph_retrieval import GraphRetrievalResult, search_graph
from backend.hybrid_retrieval import HybridRetrievalResult, search_hybrid
from backend.knowledge_graph import KnowledgeGraph
from backend.rag_schema import GroundedResponse
from backend.semantic_retrieval import RetrievalResult, VectorIndex


def retrieve_chunks(
    question: str,
    chunks: Sequence[ChunkRecord],
    vector_index: VectorIndex,
    knowledge_graph: KnowledgeGraph,
    retrieval_mode: str,
    top_k: int,
) -> list[KeywordChunkResult | RetrievalResult | GraphRetrievalResult | HybridRetrievalResult]:
    """Retrieve ranked chunks using the selected baseline or semantic mode."""

    if retrieval_mode == "Keyword":
        return search_chunks(list(chunks), question, top_k=top_k)
    if retrieval_mode == "Graph":
        return search_graph(question, knowledge_graph, chunks, top_k=top_k)
    if retrieval_mode == "Hybrid":
        return search_hybrid(question, chunks, vector_index, knowledge_graph, top_k=top_k)
    return vector_index.search(question, top_k=top_k)


def ask_knowledge(
    question: str,
    chunks: Sequence[ChunkRecord],
    vector_index: VectorIndex,
    knowledge_graph: KnowledgeGraph,
    retrieval_mode: str = "Semantic",
    generation_mode: str = "Demo / Local",
    top_k: int = 5,
) -> GroundedResponse:
    """Run the complete retrieval-to-grounded-response pipeline."""

    results = retrieve_chunks(question, chunks, vector_index, knowledge_graph, retrieval_mode, top_k)
    evidence = evidence_items_from_results(results)
    return generate_grounded_answer(question, evidence, mode=generation_mode)
