from types import SimpleNamespace

import pytest

from backend.agentic_orchestrator import _evidence_sufficient, ask_agentic
from backend.document_ingestion import chunk_documents, parse_markdown_content
from backend.factual_support import unsupported_facets
from backend.graph_builder import build_knowledge_graph
from backend.rag_schema import EvidenceItem, GroundedResponse
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


def ask(question, text):
    documents = [parse_markdown_content("policy.md", "# Policy\n" + text)]
    chunks = chunk_documents(documents)
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    return ask_agentic(question, chunks, index, build_knowledge_graph(chunks))


@pytest.mark.parametrize("question,text,facet", [
    ("What is Ember Worker license price?", "Ember Worker handles document processing.", "price/cost"),
    ("What protocol does Ember Worker use?", "Ember Worker runs document processing.", "protocol"),
    ("What protocol does Ember Worker use?", "Ember Worker runs jobs. Birch Worker uses the Copper Link protocol.", "protocol"),
    ("What is Ember Worker license price?", "Ember Worker has 32 replicas and runs every 8 hours.", "price/cost"),
    ("Who owns Kubernetes?", "Kubernetes is deployed for OCR Service. Security Team reviews document access.", "ownership"),
])
def test_related_topic_without_requested_fact_refuses_and_retains_trace(question, text, facet):
    result = ask(question, text)
    assert not result.trace.evidence_sufficient
    assert result.response.grounding_status == "Insufficient evidence"
    assert facet in result.trace.sufficiency_reason
    assert result.response.sources == []
    assert result.response.retrieved_evidence
    assert result.trace.retrieved_evidence_count == len(result.response.retrieved_evidence)
    assert result.trace.tool_decisions


@pytest.mark.parametrize("question,text,answer", [
    ("What protocol does Ember Worker use?", "Ember Worker uses the Copper Link protocol.", "Copper Link"),
    ("How often does Ember Worker replicate the archive?", "Ember Worker replicates the archive every 17 minutes.", "17 minutes"),
    ("How many jobs does Ember Worker retain?", "Ember Worker retains 32 jobs.", "32"),
    ("Who owns Kubernetes?", "Platform Engineering Team owns Kubernetes.", "Platform Engineering Team"),
    ("What is Ember Worker license price?", "Ember Worker license costs 25 USD per month.", "25 USD"),
    ("Where does Ember Worker store snapshots?", "Ember Worker stores snapshots in Cedar Vault.", "Cedar Vault"),
    ("What is Ember Worker timeout?", "Ember Worker timeout is 30 seconds.", "30 seconds"),
    ("When was Ember Worker deployed?", "Ember Worker was deployed on 2025-06-12.", "2025-06-12"),
])
def test_supported_requested_facts_answer(question, text, answer):
    result = ask(question, text)
    assert result.trace.evidence_sufficient
    assert result.response.grounding_status == "Grounded in retrieved evidence"
    assert answer in result.response.answer
    assert result.response.sources == ["policy.md"]


def evidence(text, section="Policy"):
    return [EvidenceItem(rank=1, source_document="policy.md", section=section,
                         chunk_id="chunk", text=text, score=1)]


@pytest.mark.parametrize("question,text", [
    ("Where does Ember Worker store snapshots?", "Ember Worker handles jobs. Birch Worker stores snapshots in Cedar Vault."),
    ("What is Ember Worker timeout?", "Ember Worker version is 3."),
    ("How many jobs does Ember Worker retain?", "Ember Worker has 3 replicas."),
    ("When was Ember Worker deployed?", "Ember Worker has 3 replicas."),
    ("What protocol does Ember Worker use?", "Ember Worker protocol is unknown."),
    ("What protocol does Ember Worker use?", "Ember Worker protocol is not specified."),
    ("What protocol does Ember Worker use?", "Ember Worker protocol is TBD."),
    ("What protocol does Ember Worker use?", "Ember Worker does not use the Copper Link protocol."),
    ("What protocol does Ember Worker use?", "What protocol does Ember Worker use?"),
    ("Who owns Ember Worker?", "Ember Worker works with the Security Team."),
    ("What protocol does Ember Worker use and how often does it replicate?", "Ember Worker uses the Copper Link protocol."),
])
def test_facet_binding_negatives_unknown_values_and_partial_questions(question, text):
    assert unsupported_facets(question, evidence(text))


def test_labeled_assignment_uses_section_subject():
    assert unsupported_facets("Who owns Ember Worker?", evidence("Owner: Delivery Team", "Ember Worker")) == []
    assert unsupported_facets("Who owns Ember Worker?", evidence("Owner: Delivery Team", "Birch Worker")) == ["ownership"]
    assert unsupported_facets("What protocol does Ember Worker use?", evidence(
        "Birch Worker uses the Copper Link protocol.", "Ember Worker")) == ["protocol"]


def test_graph_requirement_is_preserved_even_with_facet_support():
    e = evidence("Platform Engineering Team owns Kubernetes.")
    response = GroundedResponse(answer=e[0].text, retrieved_evidence=e,
                                grounding_status="Grounded in retrieved evidence", generation_mode="Demo / Local")
    sufficient, reason = _evidence_sufficient("Who is the owner of Kubernetes?", response, [SimpleNamespace()])
    assert not sufficient
    assert reason == "A relationship-oriented question requires explicit graph-supported evidence."
    result = ask("What depends on Kubernetes?", "The platform will deploy the OCR Service and API Gateway on Kubernetes.")
    assert result.trace.evidence_sufficient
    assert result.response.retrieved_evidence


def test_empty_retrieval_still_refuses():
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    result = ask_agentic("What protocol does Ember Worker use?", [], index, build_knowledge_graph([]))
    assert not result.trace.evidence_sufficient
    assert result.trace.sufficiency_reason == "No evidence was retrieved by the selected tool."
    assert result.response.grounding_status == "Insufficient evidence"
    assert result.response.retrieved_evidence == []
