"""Local real-FastAPI release gate for source-authored demo cases."""
import json,re,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend import api
from backend.document_ingestion import load_markdown_documents,chunk_documents
from backend.semantic_retrieval import VectorIndex,LocalEmbeddingModel
from backend.graph_builder import build_knowledge_graph
from backend.request_limits import ClientRateLimiter
from backend.session_context import InMemorySessionStore
ROOT=Path(__file__).resolve().parents[1]

def check(case,r,chunks):
 errors=[]
 answer=r.get('answer','');low=answer.casefold()
 if not r.get('evidence_sufficient'):errors.append('refused')
 for fact in case['expected_facts']:
  if fact.casefold() not in low:errors.append('missing fact: '+fact)
 if case['ordered_facts'] and all(f.casefold() in low for f in case['expected_facts']):
  ranks=[low.index(f.casefold()) for f in case['expected_facts']]
  if ranks!=sorted(ranks):errors.append('deployment order')
 for bad in case['unacceptable_claims']:
  if bad.casefold() in low:errors.append('unacceptable: '+bad)
 by_id={c.chunk_id:c for c in chunks}
 cited=set(r.get('sources',[]));used=[]
 for e in r.get('provenance',[]):
  c=by_id.get(e['chunk_id'])
  if not c or (c.document_name,c.section_heading,c.text)!=(e['source_document'],e['section'],e['text']):errors.append('provenance integrity')
  if e['source_document'] in cited:used.append(e)
 if cited!={e['source_document'] for e in used}:errors.append('fabricated citation')
 # Every answer sentence must literally occur in cited evidence; headings are
 # not facts. This is intentionally stricter than the old outcome benchmark.
 for line in re.split(r'(?<=[.!?])\s+|\n+',answer):
  text=re.sub(r'^[-*]\s+','',line.strip().replace('**',''))
  if text in {'Summary','Key findings','Teams involved','Root cause','Impact'} or not text:continue
  if not any(text.casefold() in re.sub(r'\s+',' ',e['text']).casefold() for e in used):errors.append('unsupported sentence: '+text)
 v=r.get('verification',{})
 if v.get('blocked') or any(c.get('critical') and c.get('status')!='SUPPORTED' for c in v.get('claims',[])):errors.append('critical claim not verified')
 return errors

def run():
 fixture=json.loads((ROOT/'data/evaluation/demo_accuracy_golden.json').read_text(encoding='utf-8'))
 cases=fixture['cases'];assert len(cases)==16
 docs=load_markdown_documents(ROOT/'data/sample_docs');chunks=chunk_documents(docs)
 for case in cases:
  for gold in case['supporting_chunks']:
   c=next(c for c in chunks if c.document_name==gold['document'] and c.section_heading==gold['section'])
   assert c.text==gold['text'] and hashlib.sha256(c.text.encode()).hexdigest()==gold['text_sha256']
   assert all(any(f in g['text'] for g in case['supporting_chunks']) for f in case['expected_facts'])
 index=VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False));index.build(chunks);graph=build_knowledge_graph(chunks)
 rows=[]
 limiter=ClientRateLimiter()
 with patch.object(api,'session_store',InMemorySessionStore()),patch.object(api,'build_knowledge_base',return_value=(docs,chunks,index,graph)),patch.object(api.query_limiter,'check',side_effect=limiter.check):
  client=TestClient(api.app)
  for case in cases:
   response=client.post('/query',json={'question':case['question'],'session_id':case['id']})
   r=response.json();errors=check(case,r,chunks) if response.status_code==200 else ['HTTP '+str(response.status_code)]
   rows.append({'id':case['id'],'workflow':case['workflow'],'question':case['question'],'passed':not errors,'errors':errors,'response':r})
 return {'passed':all(r['passed'] for r in rows),'pass_count':sum(r['passed'] for r in rows),'case_count':16,'results':rows}
if __name__=='__main__':
 import sys
 result=run();target=ROOT/'artifacts'/('demo_accuracy_'+(sys.argv[1] if len(sys.argv)>1 else 'after')+'.json');target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'report':str(target),'passed':result['pass_count'],'failures':[{k:r[k] for k in ('id','errors')} for r in result['results'] if not r['passed']]}))
 sys.exit(0 if result['passed'] else 1)
