from pathlib import Path

from backend.agentic_orchestrator import ask_agentic, plan_tools
from backend.document_ingestion import chunk_documents, load_markdown_documents, parse_markdown_content
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex

PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _kb():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    return chunks, index, graph


def _payment_policy_kb(include_sample_docs=False):
    documents = load_markdown_documents(SAMPLE_DOCS) if include_sample_docs else []
    documents.append(parse_markdown_content(
        "payment_p1_policy.md",
        """# Payment Service P1 Incident Policy

## Incident Classification

A complete outage of the Payment Service affecting customer transactions should be classified as a P1 incident.

## Initial Escalation

For a P1 Payment Service incident, the Incident Commander must be
notified within 10 minutes of confirmation.

If the Incident Commander is unavailable, the Director of Platform
Operations is the backup escalation contact.

## Communication

The incident team must publish an initial internal status update within 20 minutes of confirmation.""",
    ))
    chunks = chunk_documents(documents)
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    return chunks, index, graph


def test_semantic_oriented_question_routes_to_semantic():
    _, _, graph = _kb()
    intent, tools = plan_tools("How is batch capacity handled?", graph)
    assert intent == "semantic knowledge question"
    assert tools[0].tool == "semantic_search"


def test_graph_dependency_question_uses_graph_capability():
    chunks, index, graph = _kb()
    result = ask_agentic("What depends on Kubernetes?", chunks, index, graph)
    assert result.trace.tool_decisions[0].tool in {"graph_search", "hybrid_search"}
    assert result.trace.retrieved_evidence_count > 0
    assert result.response.retrieved_evidence


def test_multi_hop_question_routes_to_hybrid():
    chunks, index, graph = _kb()
    result = ask_agentic(
        "What risks could affect the OCR production deployment and which teams should be involved?",
        chunks, index, graph,
    )
    assert result.trace.tool_decisions[0].tool == "hybrid_search"
    assert result.trace.retrieved_evidence_count > 0


def test_unsupported_question_returns_insufficient_evidence():
    chunks, index, graph = _kb()
    result = ask_agentic(
        "What is the company's 2028 international expansion budget?",
        chunks, index, graph,
    )
    assert result.response.grounding_status == "Insufficient evidence"
    assert "Insufficient evidence" in result.response.answer
    assert not result.trace.evidence_sufficient


def test_multi_part_agentic_answer_is_bulleted_and_covers_risks_and_teams():
    chunks, index, graph = _kb()
    result = ask_agentic(
        "What risks could affect the OCR production deployment and which teams should be involved?",
        chunks, index, graph,
    )
    assert result.trace.evidence_sufficient
    assert "**Key risks**" in result.response.answer
    assert "**Teams involved**" in result.response.answer
    assert "- " in result.response.answer
    assert "Platform Engineering Team" in result.response.answer or "Security Team" in result.response.answer


def test_notification_question_accepts_payment_escalation_evidence():
    chunks, index, graph = _payment_policy_kb(include_sample_docs=True)
    result = ask_agentic(
        "Who should be notified first during a P1 Payment Service incident?",
        chunks,
        index,
        graph,
        top_k=5,
    )

    escalation_evidence = next(
        item
        for item in result.response.retrieved_evidence
        if item.section == "Initial Escalation"
    )
    assert escalation_evidence.source_document == "payment_p1_policy.md"
    assert "Incident Commander must be notified" in escalation_evidence.text.replace("\n", " ")
    assert result.trace.evidence_sufficient
    assert result.response.grounding_status == "Grounded in retrieved evidence"
    assert "The Incident Commander must be notified within 10 minutes of confirmation." in result.response.answer
    assert "Teams involved" not in result.response.answer
    assert result.response.sources == ["payment_p1_policy.md"]
    assert not any(
        name in result.response.answer
        for name in ("UAT_Report.md", "API_Specification.md", "Security_Requirements.md")
    )
    assert {
        "UAT_Report.md",
        "API_Specification.md",
        "Security_Requirements.md",
    } <= {item.source_document for item in result.response.retrieved_evidence}
    assert len(result.response.retrieved_evidence) == result.trace.retrieved_evidence_count


def test_backup_notification_question_prefers_backup_escalation_contact():
    chunks, index, graph = _payment_policy_kb(include_sample_docs=True)
    result = ask_agentic(
        "Who is the backup escalation contact if the Incident Commander is unavailable?",
        chunks,
        index,
        graph,
        top_k=5,
    )

    assert "Director of Platform Operations" in result.response.answer
    assert "backup escalation contact" in result.response.answer
    assert result.response.sources == ["payment_p1_policy.md"]
    assert result.response.grounding_status == "Grounded in retrieved evidence"


def test_team_ownership_question_still_requires_existing_team_evidence():
    chunks, index, graph = _payment_policy_kb()
    result = ask_agentic(
        "Who owns the P1 Payment Service incident?",
        chunks,
        index,
        graph,
        top_k=3,
    )

    assert not result.trace.evidence_sufficient
    assert result.trace.sufficiency_reason == (
        "The question asks which teams are involved, but the retrieved evidence does not identify supported team ownership."
    )
