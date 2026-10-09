// @vitest-environment jsdom
import {afterEach,it,expect,vi} from 'vitest';
import {render,screen,cleanup,fireEvent} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import CorpusGraph from './CorpusGraph';
const p={source_document:'real.md',source_section:'Facts',evidence_text:'Gateway calls worker.',chunk_id:'actual-1'};
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('loads real nodes, selection and provenance without inferring edges',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({scope:'bundled_synthetic_corpus',truncated:false,nodes:[{entity_id:'gateway',name:'Gateway',entity_type:'Service',provenance:[p]},{entity_id:'worker',name:'Worker',entity_type:'Service',provenance:[p]}],edges:[{relationship_id:'actual-edge',source_entity:'gateway',target_entity:'worker',relationship_type:'CALLS',provenance:p}]}))));
 render(<CorpusGraph/>);await screen.findByText('Gateway · Service');
 expect(screen.getByText('Gateway → CALLS → Worker')).toBeInTheDocument();expect(screen.getByText('Gateway calls worker.')).toBeInTheDocument();
 fireEvent.change(screen.getByLabelText('Select corpus entity'),{target:{value:'worker'}});expect(screen.getByText('Worker · Service')).toBeInTheDocument();
 expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/graph');
});
it('shows unavailable instead of fake graph when endpoint is absent',async()=>{vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response('{}',{status:404})));render(<CorpusGraph/>);await screen.findByText(/Corpus graph unavailable/);expect(screen.queryByRole('img')).not.toBeInTheDocument();});
