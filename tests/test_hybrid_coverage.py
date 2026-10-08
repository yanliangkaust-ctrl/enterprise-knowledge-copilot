import pytest

from backend.agentic_orchestrator import ask_agentic
from backend.document_ingestion import chunk_documents, parse_markdown_content
from backend.graph_builder import build_knowledge_graph
from backend.hybrid_retrieval import search_hybrid
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


def knowledge():
    documents = [parse_markdown_content(f"d{i}.md", f"# Deployment {i}\n"
                 f"The platform will deploy the OCR Service and API Gateway on Kubernetes in zone {i}.")
                 for i in range(7)]
    documents += [parse_markdown_content("worker.md", "# Ember Worker\n"
                                        "Ember Worker uses the Copper Link protocol.")]
    chunks = chunk_documents(documents)
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    return chunks, index, build_knowledge_graph(chunks)


def test_support_survives_graph_heavy_pool_with_raw_scores_unchanged():
    chunks, index, graph = knowledge()
    question = "What protocol does Ember Worker use and what runs on Kubernetes?"
    results = search_hybrid(question, chunks, index, graph, 5)
    assert results[0].chunk.document_name == "worker.md"
    assert "Preserved requested protocol support" in results[0].reason
    assert results[0].graph_score == 0
    vector = {r.chunk.chunk_id: r.score for r in index.search(question, 10)}
    for r in results:
        assert r.vector_score == vector[r.chunk.chunk_id]
        assert r.score == 0.7 * r.vector_score + 0.3 * r.graph_score
    assert any(r.graph_score == 1 for r in results[1:])
    assert len(results) == 5
    assert len({r.chunk.chunk_id for r in results}) == 5
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert results == search_hybrid(question, chunks, index, graph, 5)
    # All eight vector candidates fit the unchanged ten-candidate pool. Sorting
    # the full returned set reconstructs the old score order for remaining slots.
    full = search_hybrid(question, chunks, index, graph, 8)
    old_order = sorted(full, key=lambda r: (-r.score, r.chunk.chunk_id))
    assert old_order.index(next(r for r in old_order if r.chunk.document_name == "worker.md")) >= 5
    remaining = [r.chunk.chunk_id for r in old_order if r.chunk.document_name != "worker.md"]
    assert [r.chunk.chunk_id for r in results[1:]] == remaining[:4]


@pytest.mark.parametrize("k", [0, 1, 3, 5, 20])
def test_count_never_exceeds_k(k):
    chunks, index, graph = knowledge()
    assert len(search_hybrid("What protocol does Ember Worker use?", chunks, index, graph, k)) <= k


def test_missing_support_is_not_fabricated_and_still_refuses():
    chunks, index, graph = knowledge()
    question = "What is Ember Worker license price and what runs on Kubernetes?"
    results = search_hybrid(question, chunks, index, graph, 5)
    assert not any("Preserved requested" in r.reason for r in results)
    assert results == sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))
    assert all(r.chunk in chunks for r in results)
    answer = ask_agentic(question, chunks, index, graph)
    assert answer.trace.tool_decisions[0].tool == "hybrid_search"
    assert not answer.trace.evidence_sufficient
    assert answer.response.grounding_status == "Insufficient evidence"
    assert answer.response.sources == []


def test_multiple_same_facet_subjects_are_preserved_and_capacity_is_bounded():
    from backend.hybrid_retrieval import _preserve_factual_coverage
    chunks = chunk_documents([parse_markdown_content("workers.md", "# Ember Worker\n"
        "Ember Worker uses the Copper Link protocol.\n## Birch Worker\n"
        "Birch Worker uses the Cedar Link protocol.")])
    ranked = [{"chunk": c, "vector_score": 0.1, "graph_score": 0, "reasons": []} for c in reversed(chunks)]
    question = "What protocol does Ember Worker use and what protocol does Birch Worker use?"
    selected = _preserve_factual_coverage(question, ranked, 2)
    assert [r["chunk"].section_heading for r in selected] == ["Ember Worker", "Birch Worker"]
    assert len(_preserve_factual_coverage(question, ranked, 1)) == 1


def test_one_chunk_can_cover_multiple_facets_without_duplicate_slots():
    from backend.hybrid_retrieval import _preserve_factual_coverage
    chunks = chunk_documents([parse_markdown_content("worker.md", "# Ember Worker\n"
        "Ember Worker uses the Copper Link protocol. Ember Worker replicates every 17 minutes.")])
    ranked = [{"chunk": chunks[0], "vector_score": 0.1, "graph_score": 0, "reasons": []}]
    selected = _preserve_factual_coverage(
        "What protocol does Ember Worker use and how often does Ember Worker replicate?", ranked, 5)
    assert len(selected) == 1
