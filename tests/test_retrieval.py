import json
from pathlib import Path

import numpy as np

from backend.document_ingestion import chunk_documents, load_markdown_documents, search_documents
from backend.retrieval_evaluation import evaluate_retrieval
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex


PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"
GROUND_TRUTH = PROJECT_ROOT / "data" / "evaluation" / "ground_truth_questions.json"


def test_chunk_creation_and_metadata_preservation():
    documents = load_markdown_documents(SAMPLE_DOCS)
    chunks = chunk_documents(documents)

    assert chunks
    assert len(chunks) >= len(documents)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.document_name and chunk.document_type for chunk in chunks)
    assert all(chunk.section_heading and chunk.text and chunk.source for chunk in chunks)
    assert any(chunk.section_heading == "Core Systems" for chunk in chunks)


def test_embedding_generation_is_normalized():
    model = LocalEmbeddingModel(use_sentence_transformer=False)
    vectors = model.encode(["Kubernetes capacity", "UAT Environment"])

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert model.backend == "numpy-hash-fallback"


def test_semantic_retrieval_and_top_k_behavior():
    chunks = chunk_documents(load_markdown_documents(SAMPLE_DOCS))
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)

    results = index.search("Kubernetes worker capacity", top_k=3)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].chunk.document_name in {
        "ADR_Deployment.md",
        "Incident_Batch_Failure.md",
    }
    assert all(-1.0 <= result.score <= 1.0 for result in results)


def test_evaluation_calculates_recall_at_k():
    questions = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))[:2]
    documents = load_markdown_documents(SAMPLE_DOCS)

    def retrieve(question: str, top_k: int):
        return search_documents(documents, question)[:top_k]

    metrics = evaluate_retrieval(questions, retrieve, ks=(1, 3, 5))

    assert set(metrics) == {1, 3, 5}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert metrics[5] >= metrics[1]
