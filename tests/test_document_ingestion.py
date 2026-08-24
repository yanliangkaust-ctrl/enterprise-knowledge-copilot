import json
from pathlib import Path

from backend.document_ingestion import load_markdown_documents, search_documents


PROJECT_ROOT = Path(__file__).parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"
GROUND_TRUTH = PROJECT_ROOT / "data" / "evaluation" / "ground_truth_questions.json"
EXPECTED_DOCUMENTS = {
    "OCR_Platform_Architecture.md",
    "API_Specification.md",
    "Security_Requirements.md",
    "ADR_Deployment.md",
    "Incident_Batch_Failure.md",
    "Risk_Register.md",
    "Deployment_Guide.md",
    "UAT_Report.md",
}


def test_sample_documents_load_successfully():
    documents = load_markdown_documents(SAMPLE_DOCS)

    assert {document.name for document in documents} == EXPECTED_DOCUMENTS
    assert all(document.raw_text for document in documents)


def test_metadata_is_extracted():
    document = next(
        document
        for document in load_markdown_documents(SAMPLE_DOCS)
        if document.name == "OCR_Platform_Architecture.md"
    )

    assert document.document_type == "Architecture Document"
    assert "Core Systems" in document.section_headings
    assert document.metadata["section_count"] == len(document.section_headings)
    assert document.metadata["character_count"] == len(document.raw_text)


def test_keyword_search_returns_relevant_documents():
    documents = load_markdown_documents(SAMPLE_DOCS)

    results = search_documents(documents, "Kubernetes capacity")
    result_names = {document.name for document in results}

    assert "ADR_Deployment.md" in result_names
    assert "Incident_Batch_Failure.md" in result_names

    risk_results = search_documents(documents, "R-001")
    assert {document.name for document in risk_results} == {
        "ADR_Deployment.md",
        "Deployment_Guide.md",
        "Incident_Batch_Failure.md",
        "Risk_Register.md",
        "UAT_Report.md",
    }


def test_ground_truth_json_loads_and_validates():
    records = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    required_fields = {"id", "question", "expected_answer", "expected_sources", "question_type"}
    valid_types = {
        "single-document retrieval",
        "cross-document synthesis",
        "relationship reasoning",
    }

    assert len(records) == 10
    assert {record["question_type"] for record in records} == valid_types
    assert all(required_fields <= record.keys() for record in records)
    assert all(record["expected_sources"] for record in records)
