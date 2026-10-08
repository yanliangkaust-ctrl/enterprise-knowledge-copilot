import pytest

from backend.agentic_orchestrator import ask_agentic
from backend.document_ingestion import chunk_documents, parse_markdown_content
from backend.factual_support import explicit_ownership_support
from backend.graph_builder import build_knowledge_graph
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


def ask(question, text):
    chunks = chunk_documents([parse_markdown_content("policy.md", "# Responsibility\n" + text)])
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    return ask_agentic(question, chunks, index, build_knowledge_graph(chunks))


def test_graph_ownership_remains_graph_supported():
    result = ask("Who owns Kubernetes?", "Platform Engineering Team owns Kubernetes.")
    assert result.trace.evidence_sufficient
    assert "Relationship support: graph evidence" in result.trace.sufficiency_reason
    assert result.response.retrieved_evidence[0].relationship_type == "OWNED_BY"


def test_unresolvable_graph_explicit_text_ownership_answers():
    result = ask("Which team owns Ember Worker?", "Delivery Team owns Ember Worker.")
    assert result.trace.evidence_sufficient
    assert "explicit textual fallback evidence" in result.trace.sufficiency_reason
    assert "Delivery Team" in result.response.answer
    assert result.response.sources == ["policy.md"]
    assert all(not e.relationship_type for e in result.response.retrieved_evidence)


@pytest.mark.parametrize("question,text", [
    ("Which team owns Ember Worker?", "Delivery Team discusses Ember Worker."),
    ("Which team owns Ember Worker?", "Delivery Team owns Birch Worker. Ember Worker handles jobs."),
    ("Which team owns Ember Worker?", "Ember Worker owner is unknown."),
    ("Which team owns Ember Worker?", "Ember Worker has no documented owner."),
    ("Which team owns Unknown Worker?", "Delivery Team owns Ember Worker."),
    ("Who owns that risk?", "Delivery Team owns Ember Worker."),
    ("Which teams own Ember Worker and Birch Worker?", "Delivery Team owns Ember Worker. Birch Worker handles jobs."),
    ("Which team owns Ember Worker and what protocol does Ember Worker use?", "Delivery Team owns Ember Worker."),
])
def test_topical_unknown_and_incomplete_ownership_still_refuse(question, text):
    result = ask(question, text)
    assert not result.trace.evidence_sufficient
    assert result.response.grounding_status == "Insufficient evidence"
    assert result.response.sources == []


def test_coordinated_responsibility_requires_explicit_statements_for_all_subjects():
    from backend.rag_schema import EvidenceItem
    def evidence(text):
        return [EvidenceItem(rank=1, source_document="policy.md", section="Policy", chunk_id="chunk", text=text, score=1)]
    question = "Which teams own archive operations and compliance?"
    assert explicit_ownership_support(question, evidence(
        "Archive Team owns archive storage. Operations Team maintains backup jobs. Compliance Team reviews access controls."))
    assert not explicit_ownership_support(question, evidence(
        "Archive Team owns archive storage. Operations Team maintains backup jobs. Compliance was mentioned."))


def test_previous_phase_outcomes_remain_intact():
    from backend.evaluation_runner import run_evaluation
    report = run_evaluation()
    queries = {q["id"]: q for q in report["queries"]}
    assert report["outcomes"]["unsupported_acceptance_count"] == 0
    assert report["outcomes"]["false_refusal_count"] == 0
    assert not queries["q10"]["strategies"]["agentic"]["outcome"]["refused"]
    assert queries["s04"]["strategies"]["agentic"]["outcome"]["facts"][0]["present"]
    assert not queries["q08"]["strategies"]["agentic"]["outcome"]["refused"]
