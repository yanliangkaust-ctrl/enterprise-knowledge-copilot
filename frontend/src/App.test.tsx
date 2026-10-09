// @vitest-environment jsdom
import {afterEach,beforeEach,describe,it,expect,vi} from 'vitest';
import {render,screen,cleanup,fireEvent,waitFor,within} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import App from './App';
import {queryApi,ApiError} from './api';
const response={answer:'Deploy gateway first, then workers.',selected_retrieval_route:'graph_search',evidence_sufficient:true,grounding_status:'Grounded in retrieved evidence',sufficiency_reason:'Supported procedure.',sources:['Deployment_Guide.md'],provenance:[{rank:1,source_document:'Deployment_Guide.md',section:'Deployment Sequence',chunk_id:'demo-1',text:'Deploy gateway first, then workers.',score:0,selection_reason:'Procedural support',relationship_type:'',entity_path:[]}],session_id:'server-session',execution_trace:{},operational_data:[]};
beforeEach(()=>{vi.stubGlobal('fetch',vi.fn().mockImplementation(async()=>new Response(JSON.stringify(response),{status:200})));});
afterEach(()=>{cleanup();vi.unstubAllGlobals();vi.restoreAllMocks();});
function ask(q='What are the steps to deploy?'){fireEvent.change(screen.getByLabelText('What would you like to know?'),{target:{value:q}});fireEvent.click(screen.getByRole('button',{name:/Ask Copilot/}));}
describe('workspace',()=>{
 it('shows synthetic boundary and fills examples without inventing answers',()=>{render(<App/>);expect(screen.getByText(/Synthetic-document portfolio demo/)).toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:/^Deployment/}));expect(screen.getByLabelText('What would you like to know?')).toHaveValue('According to the deployment guide, what are the steps to deploy the application?');expect(fetch).not.toHaveBeenCalled();});
 it('renders live answer and verbatim citations; reuses server session then resets',async()=>{render(<App/>);ask();await screen.findByText('Deploy gateway first, then workers.',{selector:'.answer'});expect(screen.getByText('● Grounded')).toBeInTheDocument();fireEvent.click(screen.getByText(/Inspect 1 evidence chunks/));expect(screen.getByText('Deployment Sequence')).toBeInTheDocument();ask('What next?');await waitFor(()=>expect(fetch).toHaveBeenCalledTimes(2));expect(JSON.parse(vi.mocked(fetch).mock.calls[1][1]!.body as string).session_id).toBe('server-session');await screen.findAllByText('● Grounded');await waitFor(()=>expect(screen.getByRole('button',{name:/New conversation/})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:/New conversation/}));expect(screen.queryByText('What next?')).not.toBeInTheDocument();});
 it('shows refusal without presenting retrieved chunks as citations',async()=>{vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify({...response,evidence_sufficient:false,grounding_status:'Insufficient evidence',answer:'Insufficient evidence.',sources:[]})));render(<App/>);ask();await screen.findByText('○ Insufficient evidence');expect(screen.getByText('No sources cited')).toBeInTheDocument();});
 it('handles loading and prevents concurrent submits',async()=>{let resolve!:(r:Response)=>void;vi.mocked(fetch).mockReturnValue(new Promise(r=>resolve=r));render(<App/>);ask();expect(screen.getByRole('button',{name:/Searching/})).toBeDisabled();expect(screen.getByText(/Searching the corpus/)).toBeInTheDocument();resolve(new Response(JSON.stringify(response)));await screen.findByText('● Grounded');});
 it('does not interpret source content as HTML',async()=>{vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify({...response,answer:'<script>alert(1)</script>'})));render(<App/>);ask();await screen.findByText('<script>alert(1)</script>');expect(document.querySelector('script')).toBeNull();});
});
describe('API boundary',()=>{
 it.each([422,429,500,503])('handles HTTP %s with safe message and request ID',async status=>{vi.mocked(fetch).mockResolvedValue(new Response('{"detail":"private internal error"}',{status,headers:{'X-Request-ID':'req-1','Retry-After':'30'}}));try{await queryApi('question','session',new AbortController().signal);throw new Error('expected error');}catch(e){expect(e).toBeInstanceOf(ApiError);expect((e as ApiError).status).toBe(status);expect((e as ApiError).requestId).toBe('req-1');expect((e as Error).message).not.toContain('private');}});
 it('rejects malformed success responses',async()=>{vi.mocked(fetch).mockResolvedValue(new Response('{}'));await expect(queryApi('q','s',new AbortController().signal)).rejects.toThrow('incompatible');});
 it('handles network failures',async()=>{vi.mocked(fetch).mockRejectedValue(new TypeError('network'));await expect(queryApi('q','s',new AbortController().signal)).rejects.toThrow('Could not reach');});
});

describe('client-side conversations',()=>{
 it('switches actual history and preserves separate server sessions and drafts',async()=>{
  vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({...response,session_id:'session-a',answer:'Answer A'}))).mockResolvedValueOnce(new Response(JSON.stringify({...response,session_id:'session-b',answer:'Answer B'})));
  render(<App/>);ask('Question A');await screen.findByText('Answer A');
  fireEvent.change(screen.getByLabelText('What would you like to know?'),{target:{value:'Draft A'}});
  fireEvent.click(screen.getByRole('button',{name:/New conversation/}));
  expect(screen.getByLabelText('What would you like to know?')).toHaveValue('');
  ask('Question B');await screen.findByText('Answer B');
  const nav=within(screen.getByRole('navigation',{name:'Client-side conversations'}));
  fireEvent.click(nav.getByRole('button',{name:'Question A'}));
  expect(screen.getByText('Answer A')).toBeInTheDocument();expect(screen.queryByText('Answer B')).not.toBeInTheDocument();
  expect(screen.getByLabelText('What would you like to know?')).toHaveValue('Draft A');
  expect(nav.getByRole('button',{name:'Question A'})).toHaveAttribute('aria-current','page');
  ask('Follow-up A');await waitFor(()=>expect(fetch).toHaveBeenCalledTimes(3));
  expect(JSON.parse(vi.mocked(fetch).mock.calls[2][1]!.body as string).session_id).toBe('session-a');
 });
 it('new conversation uses a new request session without deleting prior history',async()=>{
  render(<App/>);ask('First question');await screen.findByText('● Grounded');
  const first=JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string).session_id;
  fireEvent.click(screen.getByRole('button',{name:/New conversation/}));
  expect(screen.queryByText('● Grounded')).not.toBeInTheDocument();
  ask('Second question');await waitFor(()=>expect(fetch).toHaveBeenCalledTimes(2));
  const second=JSON.parse(vi.mocked(fetch).mock.calls[1][1]!.body as string).session_id;
  expect(second).not.toBe(first);expect(second).not.toBe('server-session');expect(second.length).toBeLessThanOrEqual(128);
  await screen.findByText('● Grounded');
  fireEvent.click(within(screen.getByRole('navigation',{name:'Client-side conversations'})).getByRole('button',{name:'First question'}));
  expect(within(screen.getByRole('region',{name:'Conversation'})).getByText('First question')).toBeInTheDocument();
 });
 it('blocks switching while a response is pending',async()=>{
  render(<App/>);ask('Saved conversation');await screen.findByText('● Grounded');
  fireEvent.click(screen.getByRole('button',{name:/New conversation/}));
  let resolve!:(r:Response)=>void;vi.mocked(fetch).mockReturnValue(new Promise(r=>resolve=r));ask('Pending conversation');
  expect(within(screen.getByRole('navigation',{name:'Client-side conversations'})).getByRole('button',{name:'Saved conversation'})).toBeDisabled();
  resolve(new Response(JSON.stringify(response)));await screen.findByText('● Grounded');
 });
});

describe('workflow examples',()=>{
 it.each(['Deployment','Architecture','Security','Incidents'])('%s fills the composer without sending a request',label=>{
  render(<App/>);fireEvent.click(screen.getByRole('button',{name:new RegExp('^'+label)}));
  expect((screen.getByLabelText('What would you like to know?') as HTMLTextAreaElement).value.length).toBeGreaterThan(0);
  expect(screen.getByLabelText('What would you like to know?')).toHaveFocus();expect(fetch).not.toHaveBeenCalled();
 });
});

describe('active answer inspector',()=>{
 it('keeps the newest answer first and clears old results while a new request is pending',async()=>{
  vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({...response,answer:'Earlier answer'})));
  render(<App/>);ask('Earlier question');await screen.findByText('Earlier answer');
  let resolve!:(r:Response)=>void;vi.mocked(fetch).mockReturnValue(new Promise(r=>resolve=r));ask('Latest question');
  const inspector=within(screen.getByRole('complementary',{name:'Answer Inspector'}));
  expect(inspector.getByText('Latest question')).toBeInTheDocument();expect(inspector.queryByText('Supported procedure.')).not.toBeInTheDocument();
  expect(inspector.getByText('Pending API response')).toBeInTheDocument();
  resolve(new Response(JSON.stringify({...response,answer:'Latest answer',sufficiency_reason:'Latest API reason'})));
  await screen.findByText('Latest answer');expect(inspector.getByText('Latest API reason')).toBeInTheDocument();
  const articles=within(screen.getByRole('region',{name:'Conversation'})).getAllByRole('article');
  expect(within(articles[0]).getByText('Latest answer')).toBeInTheDocument();
 });
 it('resets inspector state on new conversation and restores actual earlier response on switch',async()=>{
  render(<App/>);ask('Inspected question');await screen.findByText('● Grounded');
  fireEvent.click(screen.getByRole('button',{name:/New conversation/}));
  const inspector=within(screen.getByRole('complementary',{name:'Answer Inspector'}));
  expect(inspector.getByText('Not evaluated — no active response')).toBeInTheDocument();
  fireEvent.click(within(screen.getByRole('navigation',{name:'Client-side conversations'})).getByRole('button',{name:'Inspected question'}));
  expect(within(screen.getByRole('complementary',{name:'Answer Inspector'})).getByText('Supported procedure.')).toBeInTheDocument();
 });
});
