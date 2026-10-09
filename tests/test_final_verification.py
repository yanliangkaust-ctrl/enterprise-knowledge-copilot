import pytest
from backend.claim_verification import verify_answer, review_signal
from backend.rag_schema import GroundedResponse, EvidenceItem
from backend.public_graph import corpus_graph

def response(answer,text,sources=None):
 return GroundedResponse(answer=answer,sources=['demo.md'] if sources is None else sources,retrieved_evidence=[EvidenceItem(rank=1,source_document='demo.md',section='Facts',chunk_id='one',text=text,score=1)],grounding_status='Grounded in retrieved evidence',generation_mode='Demo / Local')

@pytest.mark.parametrize('fact',['The batch limit is 100 jobs.','The release date is 2026-04-18.','The Platform Team owns operations.'])
def test_literal_supported(fact):
 v=verify_answer(response(fact,fact));assert v.status=='SUPPORTED' and not v.blocked and v.claims[0].evidence_ids==['one']

@pytest.mark.parametrize('answer,source',[('The batch limit is 200 jobs.','The batch limit is 100 jobs.'),('The release date is 2026-05-18.','The release date is 2026-04-18.'),('The price is 100 USD.','The price is 100 EUR.')])
def test_unsupported_values_refuse(answer,source):
 v=verify_answer(response(answer,source));assert v.blocked and v.claims[0].status in {'NOT_VERIFIED','UNSUPPORTED'}

def test_unknown_claim_never_passes_and_uncited_evidence_not_used():
 assert verify_answer(response('Workers are highly reliable.','Workers process jobs.')).status=='NOT_VERIFIED'
 assert verify_answer(response('Workers process jobs.','Workers process jobs.',[])).status=='NOT_VERIFIED'

def test_negated_source_not_literal_support():
 v=verify_answer(response('The price is 100 USD.','It is not true that the price is 100 USD.'));assert v.blocked and v.status!='SUPPORTED'

def test_bounds_fail_closed():
 assert verify_answer(response('Fact. '*33,'Fact.')).blocked

def test_simulated_operational_values_not_claimed_verified():
 v=verify_answer(response('Last updated: 2026-01-01.','Source evidence.'),'Last updated: 2026-01-01.');assert v.status=='NOT_VERIFIED' and not v.blocked

def test_review_is_signal_only():
 assert review_signal('Approve production deployment')['status']=='REVIEW_REQUIRED'
 assert review_signal('Describe architecture')['status']=='NOT_FLAGGED'

def test_graph_is_bounded_and_every_edge_has_real_provenance():
 g=corpus_graph();assert g['scope']=='bundled_synthetic_corpus' and len(g['nodes'])<=80 and len(g['edges'])<=160
 ids={n['entity_id'] for n in g['nodes']}
 for e in g['edges']:
  assert e['source_entity'] in ids and e['target_entity'] in ids
  assert e['provenance']['chunk_id'] and e['provenance']['evidence_text']
 assert all(e['provenance']['source_document']!='cobalt_relay.md' for e in g['edges'])

def test_post_generation_blocks_in_pipeline():
 from backend.langgraph_orchestrator import generate_answer
 r=response('The price is 200 USD.','The price is 100 USD.')
 result=generate_answer({'draft_response':r,'operational_results':[]})
 assert result['evidence_sufficient'] is False and result['response'].sources==[] and result['response'].grounding_status=='Insufficient evidence'


def test_wrapped_sentence_remains_supported():
 assert verify_answer(response('The Incident Commander must be notified within 10 minutes of confirmation.','The Incident Commander must be notified\nwithin 10 minutes of confirmation.')).status=='SUPPORTED'


def test_graph_api_and_verification_fields(monkeypatch):
 from fastapi.testclient import TestClient
 from backend import api
 from backend.document_ingestion import load_markdown_documents,chunk_documents
 from backend.semantic_retrieval import VectorIndex,LocalEmbeddingModel
 from backend.graph_builder import build_knowledge_graph
 from pathlib import Path
 docs=load_markdown_documents(Path('data/sample_docs'));chunks=chunk_documents(docs)
 index=VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False));index.build(chunks)
 monkeypatch.setattr(api,'build_knowledge_base',lambda:(docs,chunks,index,build_knowledge_graph(chunks)))
 client=TestClient(api.app)
 g=client.get('/graph');assert g.status_code==200 and g.json()['scope']=='bundled_synthetic_corpus'
 r=client.post('/query',json={'question':'According to the deployment guide, what are the steps to deploy the application?'})
 assert r.status_code==200 and r.json()['evidence_sufficient']
 assert r.json()['verification']['status']=='SUPPORTED' and r.json()['sources']==['Deployment_Guide.md']
 assert r.headers.get('X-Request-ID')
 assert r.json()['review']['status']!='APPROVED'

def test_exact_origin_cors_including_error():
 import os,subprocess,sys
 code="""from fastapi.testclient import TestClient
from backend.api import app
c=TestClient(app)
h={'Origin':'https://copilot-demo.pages.dev','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'}
r=c.options('/query',headers=h)
assert r.status_code==200 and r.headers['access-control-allow-origin']=='https://copilot-demo.pages.dev'
h['Origin']='https://other.pages.dev'
assert c.options('/query',headers=h).status_code==400
r=c.post('/query',json={'question':''},headers={'Origin':'https://copilot-demo.pages.dev'})
assert r.status_code==422 and r.headers['access-control-allow-origin']=='https://copilot-demo.pages.dev'
"""
 env={**os.environ,'FRONTEND_ORIGINS':'https://copilot-demo.pages.dev'}
 subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True)


def test_conditional_excerpt_values_match_but_claim_not_verified():
 v=verify_answer(response('The owner responds within 10 minutes.','For a major incident, the owner responds within 10 minutes.'))
 assert v.status=='NOT_VERIFIED' and not v.blocked
 assert v.claims[0].reason_code=='conditional_context_not_verified'
 assert verify_answer(response('The owner responds within 20 minutes.','For a major incident, the owner responds within 10 minutes.')).blocked
