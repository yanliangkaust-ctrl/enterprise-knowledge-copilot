import pytest
from pathlib import Path
from backend.procedural_support import requested_operations, procedure_evidence
from backend.answer_generation import generate_grounded_answer
from backend.rag_schema import EvidenceItem
from backend.document_ingestion import load_markdown_documents, chunk_documents
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex
from backend.graph_builder import build_knowledge_graph
from backend.agentic_orchestrator import ask_agentic
from backend.langgraph_orchestrator import _graph_weakness


def item(text, section='Procedure'):
    return EvidenceItem(rank=1, source_document='neutral.md', section=section, chunk_id='one', score=1, text=text)


@pytest.mark.parametrize('q', ['What are the steps to deploy the application?', 'What is the procedure for deployment?', 'How to deploy the application?', 'Describe the deployment sequence'])
@pytest.mark.parametrize('text,section', [('Platform Team owns the deployment pipeline.', 'Ownership'),
    ('1. Restart failed workers.\n2. Review deployment alerts.', 'Incident remediation'),
    ('1. Confirm deployment approval.\n2. Check credentials.', 'Prerequisites'),
    ('1. Configure backup retention.\n2. Verify archive capacity.', 'Other procedure')])
def test_topical_and_prerequisites_refuse(q,text,section):
    assert procedure_evidence(q,[item(text,section)]) is None
    assert generate_grounded_answer(q,[item(text,section)]).sources == []


@pytest.mark.parametrize('text', ['Deploy the gateway first, then deploy workers; verify connectivity.',
                                '1. Deploy gateway configuration.\n2. Deploy workers.\n3. Verify connectivity.'])
def test_supported_extractively_cites_only_used(text):
    evidence = [item('Platform Team owns deployment.', 'Ownership'), item(text)]
    result = generate_grounded_answer('What are the steps to deploy the application?', evidence)
    assert result.answer == text and result.sources == ['neutral.md']
    assert result.retrieved_evidence == [evidence[1]]


def test_multipart_requires_all_operations():
    evidence=[item('1. Deploy workers.\n2. Verify connectivity.')]
    assert procedure_evidence('What are the steps to deploy and restore the database?', evidence) is None


@pytest.mark.parametrize('q', ['Who owns deployment?', 'What protocol does the archive use?', 'How does UAT support deployment?', 'What checks are required before production?'])
def test_nonprocedural_unchanged(q):
    assert not requested_operations(q)


def test_cloud_question_refuses_without_procedural_candidate():
    q='According to the deployment guide, what are the steps to deploy the application?'
    chunks=chunk_documents(load_markdown_documents(Path('data/sample_docs')))
    index=VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False)); index.build(chunks)
    from backend.answer_generation import evidence_items_from_results
    from backend.agentic_orchestrator import _evidence_sufficient
    results = index.search(q, 5)
    response = generate_grounded_answer(q, evidence_items_from_results(results))
    assert len(response.retrieved_evidence) == 5
    assert response.sources == []
    assert _evidence_sufficient(q, response, results)[0] is False


def test_graph_weakness_uses_same_predicate():
    from types import SimpleNamespace
    chunk=SimpleNamespace(document_name='neutral.md',section_heading='Ownership',chunk_id='one',text='Platform Team owns deployment.')
    result=SimpleNamespace(chunk=chunk,rank=1,score=1)
    observation={}
    assert _graph_weakness('How to deploy?', 'How to deploy?', [result],observation)
    assert observation['reason_code']=='graph_missing_procedural_support'
