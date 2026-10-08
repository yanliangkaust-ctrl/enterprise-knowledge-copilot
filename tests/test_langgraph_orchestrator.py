from pathlib import Path

from backend.agentic_orchestrator import ask_agentic
from backend.langgraph_orchestrator import ask_agentic_langgraph, build_agent_graph
from backend.document_ingestion import chunk_documents, load_markdown_documents, parse_markdown_content
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex
from backend.session_context import InMemorySessionStore, build_relevant_context, has_follow_up_reference

PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _kb():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    return chunks, index, graph


def _payment_policy_kb():
    document = parse_markdown_content(
        "payment_p1_policy.md",
        """# Payment Service P1 Incident Policy

## Initial Escalation

For a P1 Payment Service incident, the Incident Commander must be notified within 10 minutes of confirmation.

If the Incident Commander is unavailable, the Director of Platform Operations is the backup escalation contact.""",
    )
    chunks = chunk_documents([document])
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


def test_payment_incident_follow_up_resolves_recipient_and_preserves_session():
    chunks, index, graph = _payment_policy_kb()
    store = InMemorySessionStore()
    session_id = "payment-incident-session"
    first_question = "Who should be notified first during a P1 Payment Service incident?"

    first = ask_agentic(first_question, chunks, index, graph)
    assert first.response.grounding_status == "Grounded in retrieved evidence"
    assert "Incident Commander" in first.response.answer
    store.record(session_id, first_question, first.response, graph)

    follow_up_question = "What if they're unavailable?"
    prior_context = store.get(session_id)
    assert len(prior_context.recent_turns) == 1
    assert prior_context.recent_turns[0].question == first_question
    assert has_follow_up_reference(follow_up_question)
    assert build_relevant_context(follow_up_question, prior_context)

    follow_up = ask_agentic(
        follow_up_question,
        chunks,
        index,
        graph,
        session_context=store.get(session_id),
    )

    assert "Incident Commander" in follow_up.trace.effective_question
    assert "P1 Payment Service incident" in follow_up.trace.effective_question
    assert any(item.source_document == "payment_p1_policy.md" for item in follow_up.response.retrieved_evidence)
    assert "Director of Platform Operations" in follow_up.response.answer
    assert "backup escalation contact" in follow_up.response.answer
    assert follow_up.response.grounding_status == "Grounded in retrieved evidence"


def test_new_session_does_not_inherit_payment_incident_context():
    chunks, index, graph = _payment_policy_kb()
    store = InMemorySessionStore()
    first_question = "Who should be notified first during a P1 Payment Service incident?"
    first = ask_agentic(first_question, chunks, index, graph)
    store.record("first-session", first_question, first.response, graph)

    new_session_context = store.get("new-session")
    assert not new_session_context.recent_turns
    assert not build_relevant_context("What if they're unavailable?", new_session_context)
    isolated = ask_agentic(
        "What if they're unavailable?",
        chunks,
        index,
        graph,
        session_context=new_session_context,
    )

    assert "Incident Commander" not in isolated.trace.effective_question
    assert "P1 Payment Service incident" not in isolated.trace.effective_question
