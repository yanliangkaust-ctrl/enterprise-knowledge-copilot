from pathlib import Path

from backend.agentic_orchestrator import ask_agentic
from backend.langgraph_orchestrator import ask_agentic_langgraph, build_agent_graph
from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex
from backend.session_context import InMemorySessionStore

PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _kb():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    return chunks, index, graph


def test_langgraph_routing_matches_legacy_logic_for_semantic_question():
    chunks, index, graph = _kb()
    legacy = ask_agentic("How is batch capacity handled?", chunks, index, graph)
    routed = ask_agentic_langgraph("How is batch capacity handled?", chunks, index, graph)

    assert legacy.trace.interpreted_intent == routed.trace.interpreted_intent
    assert legacy.trace.tool_decisions[0].tool == routed.trace.tool_decisions[0].tool == "semantic_search"


def test_langgraph_evidence_sufficiency_matches_legacy_logic_for_insufficient_evidence():
    chunks, index, graph = _kb()
    legacy = ask_agentic("What is the company's 2028 international expansion budget?", chunks, index, graph)
    routed = ask_agentic_langgraph("What is the company's 2028 international expansion budget?", chunks, index, graph)

    assert legacy.response.grounding_status == routed.response.grounding_status == "Insufficient evidence"
    assert legacy.trace.evidence_sufficient == routed.trace.evidence_sufficient is False
    assert legacy.response.answer == routed.response.answer


def test_langgraph_grounded_answer_matches_legacy_answer_for_multi_part_question():
    chunks, index, graph = _kb()
    legacy = ask_agentic(
        "What risks could affect the OCR production deployment and which teams should be involved?",
        chunks,
        index,
        graph,
    )
    routed = ask_agentic_langgraph(
        "What risks could affect the OCR production deployment and which teams should be involved?",
        chunks,
        index,
        graph,
    )

    assert legacy.response.answer == routed.response.answer
    assert legacy.trace.evidence_sufficient == routed.trace.evidence_sufficient is True
    assert legacy.trace.sufficiency_reason == routed.trace.sufficiency_reason


def test_langgraph_workflow_has_explicit_node_structure():
    graph = build_agent_graph()
    node_names = set(graph.nodes)

    assert {"build_context", "route_query", "retrieve_tool", "generate_answer", "evaluate_evidence", "finalize_trace"}.issubset(node_names)


def test_langgraph_context_resolves_follow_up_without_changing_evidence_boundary():
    chunks, index, graph = _kb()
    store = InMemorySessionStore()
    first = ask_agentic("What is the impact of Kubernetes capacity during OCR batch peaks?", chunks, index, graph)
    store.record(
        "follow-up",
        "What is the impact of Kubernetes capacity during OCR batch peaks?",
        first.response,
        graph,
    )

    follow_up = ask_agentic(
        "Who owns that risk?",
        chunks,
        index,
        graph,
        session_context=store.get("follow-up"),
    )

    assert follow_up.trace.evidence_sufficient is True
    assert "Platform Engineering Team" in follow_up.response.answer
    assert follow_up.response.retrieved_evidence
