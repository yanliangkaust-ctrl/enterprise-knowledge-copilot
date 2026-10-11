import json
from pathlib import Path
import pytest
from scripts.validate_demo_accuracy import run
from backend.precise_evidence import select
from backend.rag_schema import EvidenceItem
CASES=json.loads(Path('data/evaluation/demo_accuracy_golden.json').read_text())['cases']
@pytest.fixture(scope='module')
def report():return run()
@pytest.mark.parametrize('case',CASES,ids=[c['id'] for c in CASES])
def test_source_grounded_demo_case(case,report):
 row=next(r for r in report['results'] if r['id']==case['id'])
 assert row['passed'], row['errors']
 assert all(c['status']=='SUPPORTED' for c in row['response']['verification']['claims'])
 assert len(row['response']['provenance'])<=5

def test_unknown_incident_is_not_answered_from_other_service():
 e=EvidenceItem(rank=1,source_document='demo.md',section='Incident',chunk_id='one',score=1,text='A batch of OCR Service jobs failed. Workers reached their memory limit.')
 assert select('Why did Zephyr Service batch jobs fail?',[e])==[]

def test_impact_is_not_selected_as_cause():
 e=EvidenceItem(rank=1,source_document='demo.md',section='Impact',chunk_id='one',score=1,text='Batch jobs were delayed for six hours.')
 assert select('What caused the batch incident?',[e])==[]

def test_presets_match_actual_frontend_questions():
 app=Path('frontend/src/App.tsx').read_text(encoding='utf-8')
 for case in CASES:
  if case['preset']:assert case['question'] in app
