from pathlib import Path

from backend.answer_generation import evidence_items_from_results, generate_grounded_answer
from backend.context_builder import build_context
from backend.document_ingestion import chunk_documents, load_markdown_documents
from backend.knowledge_base import KnowledgeBase
from backend.prompt_builder import GROUNDED_SYSTEM_PROMPT, build_grounded_prompt
from backend.rag_pipeline import ask_knowledge
from backend.rag_schema import EvidenceItem, GroundedResponse
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def _evidence():
    return [
        EvidenceItem(
            rank=1,
            source_document="API_Specification.md",
            section="Overview",
            chunk_id="API_Specification.md::chunk-001",
            text="The API Gateway is the public interface for the Government OCR Platform.",
            score=0.91,
        )
    ]


def test_context_construction_preserves_source_metadata():
    context = build_context(_evidence())

    assert "[SOURCE 1]" in context
    assert "Document: API_Specification.md" in context
    assert "Section: Overview" in context
    assert "Chunk ID: API_Specification.md::chunk-001" in context
    assert "[/SOURCE]" in context


def test_prompt_construction_contains_grounding_instructions_and_context():
    prompt = build_grounded_prompt("What is the API Gateway?", _evidence())

    assert "API_Specification.md" in prompt
    assert "What is the API Gateway?" in prompt
    assert "Insufficient evidence" in prompt
    assert "only the supplied enterprise evidence" in GROUNDED_SYSTEM_PROMPT


def test_demo_generation_is_deterministic_and_structured():
    first = generate_grounded_answer("What is the API Gateway?", _evidence())
    second = generate_grounded_answer("What is the API Gateway?", _evidence())

    assert first == second
    assert isinstance(first, GroundedResponse)
    assert first.grounding_status == "Grounded in retrieved evidence"
    assert first.sources == ["API_Specification.md"]
    assert first.retrieved_evidence[0].chunk_id == "API_Specification.md::chunk-001"


def test_demo_mode_does_not_call_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-that-is-not-used")

    class FailIfConstructed:
        def __init__(self):
            raise AssertionError("Demo mode must not construct an OpenAI client")

    monkeypatch.setitem(__import__("sys").modules, "openai", type("OpenAIModule", (), {"OpenAI": FailIfConstructed}))
    response = generate_grounded_answer("What is the API Gateway?", _evidence(), mode="Demo / Local")

    assert response.generation_mode == "Demo / Local"


def test_openai_generation_uses_mocked_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-that-is-not-sent")

    class FakeResponses:
        def create(self, **kwargs):
            assert "API_Specification.md" in kwargs["input"]
            return type("Response", (), {"output_text": "The API Gateway is the platform entry point."})()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    fake_openai = type("OpenAIModule", (), {"OpenAI": FakeClient})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    response = generate_grounded_answer("What is the API Gateway?", _evidence(), mode="OpenAI")

    assert response.answer == "The API Gateway is the platform entry point."
    assert response.generation_mode == "OpenAI"
    assert response.sources == ["API_Specification.md"]


def test_insufficient_evidence_is_explicit_and_has_no_fabricated_sources():
    response = generate_grounded_answer(
        "What is the company's 2028 international expansion budget?",
        _evidence(),
    )

    assert response.grounding_status == "Insufficient evidence"
    assert "Insufficient evidence" in response.answer
    assert response.sources == []


def test_ask_knowledge_pipeline_retrieves_then_generates_traceable_answer():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    retrieval_results = index.search("How does Kubernetes handle batch capacity?", top_k=3)

    response = generate_grounded_answer(
        "How does Kubernetes handle batch capacity?",
        evidence_items_from_results(retrieval_results),
    )

    assert response.retrieved_evidence
    assert response.sources
    assert response.sources[0] == response.retrieved_evidence[0].source_document
    assert response.grounding_status == "Grounded in retrieved evidence"


def test_uploaded_markdown_refreshes_shared_index_and_is_retrievable():
    knowledge_base = KnowledgeBase(LocalEmbeddingModel(use_sentence_transformer=False))
    original_index = knowledge_base.vector_index
    filename = "Admin_Quartz_Archive.md"
    content = (
        "# Cobalt Relay\n"
        "Cobalt Relay uses the Quartz Lattice protocol to replicate the archive "
        "every 17 minutes from the northern storage vault."
    )

    document = knowledge_base.ingest_markdown(filename, content)

    assert document.name == filename
    assert document.metadata["filename"] == filename
    assert knowledge_base.chunks[0].document_name == filename
    assert knowledge_base.chunks[0].source == filename
    assert knowledge_base.vector_index is original_index
    assert knowledge_base.vector_index.chunks == knowledge_base.chunks

    response = ask_knowledge(
        "What protocol does Cobalt Relay use to replicate the archive?",
        knowledge_base.chunks,
        knowledge_base.vector_index,
        knowledge_base.knowledge_graph,
        retrieval_mode="Semantic",
        generation_mode="Demo / Local",
    )

    assert filename in response.sources
    assert response.retrieved_evidence[0].source_document == filename
    assert "Quartz Lattice" in response.retrieved_evidence[0].text
