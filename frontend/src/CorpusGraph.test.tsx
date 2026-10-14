// @vitest-environment jsdom
import {afterEach,it,expect,vi} from 'vitest';
import {render,screen,cleanup,fireEvent} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import CorpusGraph from './CorpusGraph';
const p={source_document:'real.md',source_section:'Facts',evidence_text:'Gateway calls worker.',chunk_id:'actual-1'};
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('loads real nodes, selection and provenance without inferring edges',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'gateway',name:'Gateway',entity_type:'Service',provenance:[p]},{entity_id:'worker',name:'Worker',entity_type:'Service',provenance:[p]}],edges:[{relationship_id:'actual-edge',source_entity:'gateway',target_entity:'worker',relationship_type:'CALLS',provenance:p}]}))));
 render(<CorpusGraph/>);await screen.findByLabelText('Graph scope');fireEvent.change(screen.getByLabelText('Graph scope'),{target:{value:'corpus'}});fireEvent.change(screen.getByLabelText('Select corpus entity'),{target:{value:'gateway'}});
 expect(screen.getByText('1. Gateway → calls → Worker')).toBeInTheDocument();expect(screen.getByText('Gateway calls worker.')).toBeInTheDocument();
 fireEvent.change(screen.getByLabelText('Select corpus entity'),{target:{value:'worker'}});expect(screen.getByText('Worker · Service')).toBeInTheDocument();
 expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/graph');
});
it('shows unavailable instead of fake graph when endpoint is absent',async()=>{vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response('{}',{status:404})));render(<CorpusGraph/>);await screen.findByText(/Corpus graph unavailable/);expect(screen.queryByRole('img')).not.toBeInTheDocument();});

it('has no unrelated default selection without answer evidence',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'adr',name:'ADR Deployment',entity_type:'Document',provenance:[p]}],edges:[]}))));
 render(<CorpusGraph question="Why did the batch fail?"/>);
 await screen.findByText(/No source-matched relationships/);
 expect(screen.queryByText('ADR Deployment · Document')).not.toBeInTheDocument();
 expect(screen.queryByRole('img')).not.toBeInTheDocument();
});
it('matches cited text across different runtime IDs and clears on refusal',async()=>{
 const g={scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'gateway',name:'Gateway',entity_type:'Service',provenance:[p]},{entity_id:'worker',name:'Worker',entity_type:'Service',provenance:[p]}],edges:[{relationship_id:'edge',source_entity:'gateway',target_entity:'worker',relationship_type:'CALLS',provenance:p}]};
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify(g))));
 const response={evidence_sufficient:true,sources:['real.md'],provenance:[{source_document:'real.md',section:'Facts',text:p.evidence_text,chunk_id:'runtime-uuid'}]} as unknown as import('./api').QueryResponse;
 const view=render(<CorpusGraph question="Gateway?" response={response}/>);
 await screen.findByText('1. Gateway → calls → Worker');
 expect(screen.getByText('Answer chunk: runtime-uuid')).toBeInTheDocument();
 view.rerender(<CorpusGraph question="Gateway?" response={{...response,evidence_sufficient:false}}/>);
 expect(screen.queryByRole('img')).not.toBeInTheDocument();
});
it('document selection shows source relationships without fabricated document edges',async()=>{
 const g={scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'doc',name:'ADR Deployment',entity_type:'Document',provenance:[p]},{entity_id:'gateway',name:'Gateway',entity_type:'Service',provenance:[p]},{entity_id:'worker',name:'Worker',entity_type:'Service',provenance:[p]}],edges:[{relationship_id:'edge',source_entity:'gateway',target_entity:'worker',relationship_type:'CALLS',provenance:p}]};
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify(g))));render(<CorpusGraph/>);
 await screen.findByLabelText('Graph scope');fireEvent.change(screen.getByLabelText('Graph scope'),{target:{value:'corpus'}});fireEvent.change(screen.getByLabelText('Select corpus entity'),{target:{value:'doc'}});
 expect(screen.getByText('1. Gateway → calls → Worker')).toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Select ADR Deployment'})).not.toBeInTheDocument();
});
it('automatically selects the incident rather than the alphabetical document',async()=>{
 const g={scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'adr',name:'ADR Deployment',entity_type:'Document',provenance:[p]},{entity_id:'incident',name:'Batch failure',entity_type:'Incident',provenance:[p]},{entity_id:'worker',name:'Worker',entity_type:'Observation',provenance:[p]}],edges:[{relationship_id:'edge',source_entity:'incident',target_entity:'worker',relationship_type:'REPORTS',provenance:p}]};
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify(g))));
 const response={evidence_sufficient:true,sources:['real.md'],provenance:[{source_document:'real.md',section:'Facts',text:p.evidence_text,chunk_id:'uuid'}]} as unknown as import('./api').QueryResponse;
 const view=render(<CorpusGraph question="What happened to the batch?" response={response}/>);
 await screen.findByText('Batch failure · Incident');
 expect(screen.queryByText('ADR Deployment · Document')).not.toBeInTheDocument();
 view.rerender(<CorpusGraph question="What happened?" response={{...response,provenance:[{...response.provenance[0],text:'Other evidence'}]}}/>);
 expect(screen.queryByRole('img')).not.toBeInTheDocument();
});
it.each([
 {document:'Deployment_Guide.md',section:'Deployment Sequence',text:'Deploy API Gateway configuration first, then OCR Service workers, then verify Document Storage connectivity. Validate job submission and status through the API Gateway. Do not expose raw document content in Monitoring Service logs.',source:'API Gateway configuration',target:'OCR Service workers',kind:'DEPLOY_BEFORE'},
 {document:'OCR_Platform_Architecture.md',section:'Platform Flow',text:'A client submits a document through the API Gateway. The gateway validates the request and sends the job to the OCR Service. The OCR Service stores the source and result in Document Storage and publishes processing status for the Monitoring Service.',source:'API Gateway',target:'OCR Service',kind:'SENDS_JOB_TO'},
])('automatically displays cited $section relationships and excludes unrelated sources',async c=>{
 const provenance={source_document:c.document,source_section:c.section,evidence_text:c.text,chunk_id:'corpus-id'};
 const unrelated={...provenance,source_section:'Unrelated section'};
 const g={scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'source',name:c.source,entity_type:'Observation',provenance:[provenance]},{entity_id:'target',name:c.target,entity_type:'Observation',provenance:[provenance]}],edges:[{relationship_id:'supported',source_entity:'source',target_entity:'target',relationship_type:c.kind,provenance},{relationship_id:'unrelated',source_entity:'target',target_entity:'source',relationship_type:'UNRELATED',provenance:unrelated}]};
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify(g))));
 const response={evidence_sufficient:true,sources:[c.document],provenance:[{source_document:c.document,section:c.section,text:c.text,chunk_id:'runtime-id'}]} as unknown as import('./api').QueryResponse;
 const view=render(<CorpusGraph response={response}/>);
 await screen.findByRole('img');expect(screen.getByText(`1. ${c.source} → ${c.kind.replaceAll('_',' ').toLowerCase()} → ${c.target}`)).toBeInTheDocument();
 expect(screen.getByText(`${c.document} · ${c.section}`)).toBeInTheDocument();expect(screen.getByText(c.text)).toBeInTheDocument();expect(screen.getByText('Corpus chunk: corpus-id')).toBeInTheDocument();expect(screen.getByText('Answer chunk: runtime-id')).toBeInTheDocument();
 expect(screen.queryByText(/→ unrelated →/)).not.toBeInTheDocument();
 view.rerender(<CorpusGraph response={{...response,sources:[]}}/>);expect(screen.queryByRole('img')).not.toBeInTheDocument();
});
