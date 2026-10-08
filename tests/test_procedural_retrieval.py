from pathlib import Path
import pytest
from backend.document_ingestion import chunk_documents, load_markdown_documents, parse_markdown_content
from backend.semantic_retrieval import VectorIndex, LocalEmbeddingModel
from backend.graph_builder import build_knowledge_graph
from backend.hybrid_retrieval import search_hybrid
from backend.agentic_orchestrator import ask_agentic


def setup(documents):
    chunks=chunk_documents(documents)
    index=VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False));index.build(chunks)
    return chunks,index,build_knowledge_graph(chunks)


def test_cloud_procedure_recovers_exact_source_order_and_citation():
    chunks,index,graph=setup(load_markdown_documents(Path('data/sample_docs')))
    question='According to the deployment guide, what are the steps to deploy the application?'
    result=ask_agentic(question,chunks,index,graph)
    sequence=next(c for c in chunks if c.section_heading=='Deployment Sequence')
    assert result.trace.evidence_sufficient
    assert result.response.answer==sequence.text
    assert result.response.sources==[sequence.document_name]
    assert [e.chunk_id for e in result.response.retrieved_evidence]==[sequence.chunk_id]
    rows=search_hybrid(question,chunks,index,graph)
    assert len(rows)==5 and rows[0].chunk.chunk_id==sequence.chunk_id
    assert rows[0].vector_score==rows[0].graph_score==rows[0].score==0
    assert rows==search_hybrid(question,chunks,index,graph)


@pytest.mark.parametrize('filename',['neutral.md','Deployment_Guide.md'])
def test_filename_not_support_and_incomplete_requests_refuse(filename):
    chunks,index,graph=setup([parse_markdown_content(filename,'# Pre-deployment Checks\n1. Confirm approval.\n2. Review deployment alerts.')])
    result=ask_agentic('What are the steps to deploy?',chunks,index,graph)
    assert not result.trace.evidence_sufficient and not result.response.sources


def test_supported_other_operation_and_missing_multipart():
    chunks,index,graph=setup([parse_markdown_content('arbitrary.md','# Procedure\n1. Restore database backup.\n2. Verify database connectivity.')])
    # Explicit supported spelling; inflection expansion is outside this experiment.
    result=ask_agentic('What are the steps to restore the database?',chunks,index,graph)
    assert result.trace.evidence_sufficient
    assert result.response.answer.startswith('1. Restore')
    assert result.response.sources==['arbitrary.md']
    result=ask_agentic('What are the steps to restore the database and deploy?',chunks,index,graph)
    assert not result.trace.evidence_sufficient and not result.response.sources


def test_remaining_ranking_and_existing_scores_unchanged():
    chunks,index,graph=setup(load_markdown_documents(Path('data/sample_docs')))
    q='What are the steps to deploy the application?'
    vector=index.search(q,10)
    rows=search_hybrid(q,chunks,index,graph)
    scores={r.chunk.chunk_id:r.score for r in vector}
    for row in rows:
        assert row.score==.7*row.vector_score+.3*row.graph_score
        if row.chunk.chunk_id in scores:
            assert row.vector_score==scores[row.chunk.chunk_id]
    assert [r.score for r in rows[1:]]==sorted([r.score for r in rows[1:]],reverse=True)


@pytest.mark.parametrize('requested_text', [
    'deploy the application and delete the archive',
    'delete the archive and deploy the application',
    'restore the database and export the archive',
])
def test_unknown_multipart_operation_cannot_be_silently_accepted(requested_text):
    from backend.procedural_support import requested_operations, procedure_evidence
    from backend.answer_generation import generate_grounded_answer, evidence_items_from_results
    from backend.langgraph_orchestrator import _graph_weakness
    chunks,index,graph=setup(load_markdown_documents(Path('data/sample_docs')))
    question='According to the deployment guide, what are the steps to '+requested_text+'?'
    assert 'unknown' in [op for op, _ in requested_operations(question)]
    rows=search_hybrid(question,chunks,index,graph)
    assert len(rows)<=5
    evidence=evidence_items_from_results(rows)
    assert procedure_evidence(question,evidence) is None
    assert generate_grounded_answer(question,evidence).sources==[]
    observation={}
    assert _graph_weakness(question,question,rows,observation)
    assert observation['reason_code']=='graph_missing_procedural_support'
    result=ask_agentic(question,chunks,index,graph)
    assert not result.trace.evidence_sufficient
    assert not result.response.sources
