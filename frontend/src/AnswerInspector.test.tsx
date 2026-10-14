// @vitest-environment jsdom
import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import {render,screen,cleanup,fireEvent} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import AnswerInspector,{graphEvidence} from './AnswerInspector';
import type {QueryResponse,Evidence} from './api';
const evidence:Evidence={rank:2,source_document:'actual.md',section:'Actual section',chunk_id:'real-chunk',text:'A depends on B.',score:0.3,selection_reason:'Actual selection',relationship_type:'DEPENDS_ON',entity_path:['A','B']};
const response:QueryResponse={answer:'Actual API answer',selected_retrieval_route:'graph_search',evidence_sufficient:true,grounding_status:'Grounded in retrieved evidence',sufficiency_reason:'Actual API reason',sources:['actual.md'],provenance:[evidence],session_id:'actual-session',execution_trace:{interpreted_intent:'Actual intent',tool_decisions:[{tool:'hybrid_search',reason:'Actual fallback reason'}],retrieved_evidence_count:5,evidence_sufficient:true,sufficiency_reason:'Actual API reason',final_generation_mode:'Demo / Local',tool_executions:[],retrieval_checks:[{reason_code:'graph_empty',fallback_activated:true}]},operational_data:[]};
beforeEach(()=>vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('unavailable'))));
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('distinguishes actual evidence decision from unavailable verification and approval',()=>{render(<AnswerInspector response={response}/>);expect(screen.getByText('Actual API reason')).toBeInTheDocument();expect(screen.getByText('Not evaluated / unavailable')).toBeInTheDocument();fireEvent.click(screen.getByText('Limitations'));expect(screen.getByText('Unavailable — no approval workflow')).toBeInTheDocument();expect(screen.getByText(/not claim-by-claim verification/)).toBeInTheDocument();expect(screen.queryByText(/verified claims|100%/i)).not.toBeInTheDocument();});
it('renders only typed directed graph metadata and its evidence',()=>{render(<AnswerInspector response={response}/>);fireEvent.click(screen.getByRole('tab',{name:'Knowledge Graph'}));expect(screen.getByText('A')).toBeInTheDocument();expect(screen.getByText('B')).toBeInTheDocument();expect(screen.getByText('DEPENDS_ON →')).toBeInTheDocument();expect(screen.getByText('A depends on B.')).toBeInTheDocument();expect(screen.getByText(/Loading real corpus graph/)).toBeInTheDocument();});
it('does not infer graph edges from text, hybrid reasons or multi-node paths',()=>{expect(graphEvidence([{...evidence,relationship_type:'',selection_reason:'A OWNS B'}, {...evidence,entity_path:['A','B','C']},{...evidence,entity_path:[]},{...evidence,relationship_type:' '}])).toEqual([]);render(<AnswerInspector response={{...response,provenance:[{...evidence,relationship_type:'',entity_path:[]}]}}/>);fireEvent.click(screen.getByRole('tab',{name:'Knowledge Graph'}));expect(screen.getByText(/No explicit graph relationships/)).toBeInTheDocument();expect(screen.queryByText('DEPENDS_ON →')).not.toBeInTheDocument();});
it('labels refusal graph context and separates cited sources from provenance',()=>{render(<AnswerInspector response={{...response,evidence_sufficient:false,grounding_status:'Insufficient evidence',sources:[]}}/>);fireEvent.click(screen.getByRole('tab',{name:'Knowledge Graph'}));expect(screen.getByText(/response refused/)).toBeInTheDocument();fireEvent.click(screen.getByRole('tab',{name:'Sources'}));expect(screen.getByText('No cited sources for the active response.')).toBeInTheDocument();expect(screen.getByText('A depends on B.')).toBeInTheDocument();expect(screen.getByText(/not supporting citations/)).toBeInTheDocument();expect(screen.getByText('0.3')).toBeInTheDocument();});
it('displays only supplied trace events, including fallback',()=>{render(<AnswerInspector response={response}/>);fireEvent.click(screen.getByText('Technical Details'));expect(screen.getByText('Intent: Actual intent')).toBeInTheDocument();expect(screen.getByText('hybrid_search: Actual fallback reason')).toBeInTheDocument();expect(screen.getByText(/"reason_code": "graph_empty"/)).toBeInTheDocument();expect(screen.getByText('No tool executions reported.')).toBeInTheDocument();});
it('supports arrow-key tabs and does not invent empty-response outcomes',()=>{render(<AnswerInspector/>);const tab=screen.getByRole('tab',{name:'Verification'});tab.focus();fireEvent.keyDown(tab,{key:'ArrowRight'});expect(screen.getByRole('tab',{name:'Knowledge Graph'})).toHaveFocus();expect(screen.getByRole('tab',{name:'Knowledge Graph'})).toHaveAttribute('aria-selected','true');fireEvent.keyDown(screen.getByRole('tab',{name:'Knowledge Graph'}),{key:'End'});expect(screen.getByText('Ask a question to inspect evidence and claim checks.')).toBeInTheDocument();expect(screen.queryByText('Technical Details')).not.toBeInTheDocument();});

it('shows real verification claims and review signal without approval',()=>{render(<AnswerInspector response={{...response,verification:{version:'literal-v1',status:'NOT_VERIFIED',coverage_complete:true,blocked:false,limitation:'Literal matching only.',claims:[{text:'Actual claim',status:'NOT_VERIFIED',reason_code:'outside_literal_coverage',critical:false,evidence_ids:[]}]},review:{required:true,status:'REVIEW_REQUIRED',reason_code:'high_risk_keyword',limitation:'No approval queue.'}}}/>);expect(screen.getByText('Not verified')).toBeInTheDocument();fireEvent.click(screen.getByText('Limitations · Review signal'));expect(screen.getByText('Actual claim')).toBeInTheDocument();expect(screen.getByText('REVIEW_REQUIRED')).toBeInTheDocument();expect(screen.queryByText('Approved')).not.toBeInTheDocument();});
it('omits coarse causal metadata and endpoints absent from source text',()=>{
 expect(graphEvidence([{...evidence,relationship_type:'CAUSED_BY'}, {...evidence,entity_path:['A','Unmentioned environment']}])).toEqual([]);
});
const checked={version:'literal-v1',status:'SUPPORTED',coverage_complete:true,blocked:false,limitation:'Literal source support only.',claims:[{text:'A depends on B.',status:'SUPPORTED',reason_code:'literal_cited_evidence',critical:false,evidence_ids:['real-chunk']}]};
it('shows evaluated evidence and bounded checks with accessible collapsed details',()=>{
 render(<AnswerInspector response={{...response,verification:checked}}/>);
 expect(screen.getByText('Sufficient evidence')).toBeInTheDocument();
 expect(screen.getByText('Source-text supported · bounded check')).toBeInTheDocument();
 expect(screen.getByText('Limitations').closest('details')).not.toHaveAttribute('open');
 expect(screen.getByText('Technical Details').closest('details')).not.toHaveAttribute('open');
 expect(screen.queryByText(/^Verified$/)).not.toBeInTheDocument();
 fireEvent.click(screen.getByText('Source-text supported · bounded check · Claim'));
 expect(screen.getByText('Evidence: real-chunk')).toBeVisible();
 fireEvent.click(screen.getByText('Technical Details'));
 expect(screen.getByText('graph_search')).toBeVisible();
 expect(screen.getByText('API verification status')).toBeVisible();
});
it.each(['UNSUPPORTED','NOT_VERIFIED','NOT_APPLICABLE','FUTURE_STATUS'])('keeps %s truthful and nonpositive',status=>{
 render(<AnswerInspector response={{...response,evidence_sufficient:false,verification:{...checked,status,blocked:status==='UNSUPPORTED',claims:[]}}}/>);
 const label={UNSUPPORTED:'Unsupported',NOT_VERIFIED:'Not verified',NOT_APPLICABLE:'Not applicable',FUTURE_STATUS:'Unknown check status'}[status];
 expect(screen.getByText(label!)).toHaveClass('unavailable');
 expect(screen.getByText('Insufficient evidence')).toBeInTheDocument();
 expect(screen.queryByText('Source-text supported · bounded check')).not.toBeInTheDocument();
});
it('does not infer successful checks from empty SUPPORTED data',()=>{
 render(<AnswerInspector response={{...response,verification:{...checked,claims:[]}}}/>);
 expect(screen.getByText('Not evaluated / unavailable')).toBeInTheDocument();
 expect(screen.queryByText('Source-text supported · bounded check')).not.toBeInTheDocument();
});
it('replaces prior evaluated checks with a concise pending state',()=>{
 render(<AnswerInspector pending response={{...response,verification:checked}}/>);
 expect(screen.getByRole('status')).toHaveTextContent('Checks pending.');
 expect(screen.queryByText('Sufficient evidence')).not.toBeInTheDocument();
 expect(screen.queryByText('Technical Details')).not.toBeInTheDocument();
});
