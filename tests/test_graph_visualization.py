from backend.public_graph import corpus_graph
from backend.document_ingestion import load_markdown_documents, chunk_documents
from backend.graph_builder import build_knowledge_graph
from pathlib import Path


def test_public_graph_observations_have_exact_source_provenance():
    chunks=chunk_documents(load_markdown_documents(Path('data/sample_docs')))
    originals={(c.document_name,c.section_heading,c.text,c.chunk_id) for c in chunks}
    g=corpus_graph()
    assert len(g['nodes'])<=80 and len(g['edges'])<=160
    ids={n['entity_id'] for n in g['nodes']}
    for e in g['edges']:
        p=e['provenance']
        assert (p['source_document'],p['source_section'],p['evidence_text'],p['chunk_id']) in originals
        assert e['source_entity'] in ids and e['target_entity'] in ids
    summary=[e for e in g['edges'] if e['provenance']['source_document']=='Incident_Batch_Failure.md' and e['provenance']['source_section']=='Summary']
    assert any(e['source_entity']=='kubernetes-worker-pods' and e['target_entity']=='memory-limit' and e['relationship_type']=='REACHED' for e in summary)
    assert any(e['target_entity']=='pending-jobs' for e in summary)
    assert any(e['target_entity']=='api-gateway' and e['relationship_type']=='REPORTS_AVAILABLE' for e in summary)
    assert not any(e['relationship_type']=='CAUSED_BY' or e['target_entity']=='uat-environment' for e in summary)
    assert not any(e['source_entity']=='kubernetes-worker-pods' and e['target_entity']=='pending-jobs' for e in summary)


def test_adr_source_has_real_edges_without_invented_document_link():
    g=corpus_graph()
    assert any(e['provenance']['source_document']=='ADR_Deployment.md' and e['relationship_type']=='DEPLOYED_ON' for e in g['edges'])
    assert not any('adr-deployment' in (e['source_entity'],e['target_entity']) for e in g['edges'])


def test_display_graph_does_not_change_retrieval_graph():
    chunks=chunk_documents(load_markdown_documents(Path('data/sample_docs')))
    before=build_knowledge_graph(chunks)
    corpus_graph()
    after=build_knowledge_graph(chunks)
    assert before.relationships()==after.relationships()
    assert before.entities()==after.entities()
    assert not any(e.entity_id=='memory-limit' for e in after.entities())

def test_visualization_flow_uses_actual_cited_chunks():
    g=corpus_graph()
    deployment=[e for e in g['edges'] if e['provenance']['source_document']=='Deployment_Guide.md' and e['provenance']['source_section']=='Deployment Sequence']
    assert [(e['source_entity'],e['relationship_type'],e['target_entity']) for e in deployment]==[('api-gateway-configuration','DEPLOY_BEFORE','ocr-service-workers')]
    architecture=[e for e in g['edges'] if e['provenance']['source_document']=='OCR_Platform_Architecture.md' and e['provenance']['source_section']=='Platform Flow']
    assert {(e['source_entity'],e['relationship_type'],e['target_entity']) for e in architecture}=={
        ('client','SUBMITS_DOCUMENT_THROUGH','api-gateway'),('api-gateway','SENDS_JOB_TO','ocr-service'),
        ('ocr-service','STORES_SOURCE_AND_RESULT_IN','document-storage'),('ocr-service','PUBLISHES_STATUS_FOR','monitoring-service')}


import pytest
from backend.public_graph import visualization_flow

@pytest.mark.parametrize('text',[
    'API Gateway and OCR Service need deployment configuration.',
    'Deploy API Gateway configuration first. OCR Service workers are available.',
    'Do not deploy API Gateway configuration first, then OCR Service workers, then verify Document Storage connectivity.',
    'If approved, deploy API Gateway configuration first, then OCR Service workers, then verify Document Storage connectivity.',
    'A client submits a document through the API Gateway. The gateway does not send the job to the OCR Service.',
    'A client submits a document through the API Gateway. Another component processes it. The gateway validates the request and sends the job to the OCR Service.',
    'The gateway validates the request and sends the job to the OCR Service.',
    'The OCR Service does not store the source and result in Document Storage.',
    'The OCR Service stores the source and result in Document Storage. Monitoring Service is available.',
])
def test_visualization_flow_rejects_unjustified_relationships(text):
    assert visualization_flow(text,['API Gateway','OCR Service','Document Storage','Monitoring Service'])==[]


def test_visualization_flow_is_entity_based_not_document_based():
    assert visualization_flow('Deploy Edge Gateway configuration first, then Reader Service workers, then verify Archive connectivity.', ['Edge Gateway','Reader Service','Archive'])==[('Edge Gateway configuration','DEPLOY_BEFORE','Reader Service workers')]
