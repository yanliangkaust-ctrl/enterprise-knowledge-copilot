// @vitest-environment jsdom
import {afterEach,it,expect,vi} from 'vitest';
import {render,screen,fireEvent,cleanup} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import DocumentLibrary from './DocumentLibrary';
import App from './App';
const doc={filename:'violet.txt',document_id:'id',version:1,status:'ready',chunk_count:1,text_bytes:80};
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
function setup(){vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({documents:[]}))));}
function select(file:File){fireEvent.change(screen.getByLabelText('Choose a document'),{target:{files:[file]}});fireEvent.click(screen.getByRole('button',{name:'Upload and index'}));}
it('uploads actual file bytes and shows ready filename',async()=>{
 setup();vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify(doc),{status:201}));
 render(<DocumentLibrary/>);await screen.findByText('Library ready.');
 const file=new File(['Violet fact.'],'violet.txt',{type:'text/plain'});select(file);
 await screen.findByText('violet.txt is indexed and ready in Ask Copilot.');
 expect(screen.getByText('Ready · 1 chunks')).toBeInTheDocument();
 expect(vi.mocked(fetch).mock.calls[1][0]).toContain('/documents?filename=violet.txt');
 expect(vi.mocked(fetch).mock.calls[1][1]?.body).toBe(file);
});
it.each([new File(['bad'],'bad.exe'),new File([],'empty.txt'),new File(['x'.repeat(1024*1024+1)],'large.pdf')])('rejects invalid file $name before upload',async file=>{
 setup();render(<DocumentLibrary/>);await screen.findByText('Library ready.');select(file);
 expect(screen.getByRole('alert')).toHaveTextContent('no larger than 1 MiB');expect(fetch).toHaveBeenCalledTimes(1);
});
it('shows server parsing errors without claiming indexed success',async()=>{
 setup();vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({detail:'No readable text found. Scanned PDFs require OCR.'}),{status:422}));
 render(<DocumentLibrary/>);await screen.findByText('Library ready.');select(new File(['broken'],'scan.pdf'));
 expect(await screen.findByRole('alert')).toHaveTextContent('Scanned PDFs require OCR');
 expect(screen.queryByText('Ready · 1 chunks')).not.toBeInTheDocument();
});
it('opens the library from navigation and preserves the question draft',async()=>{
 setup();render(<App/>);fireEvent.change(screen.getByLabelText('What would you like to know?'),{target:{value:'My draft'}});
 fireEvent.click(screen.getByRole('button',{name:'Document Library'}));await screen.findByText('Library ready.');
 expect(screen.getByRole('heading',{name:'Document Library'})).toBeVisible();
 fireEvent.click(screen.getByRole('button',{name:'Ask your knowledge'}));expect(screen.getByLabelText('What would you like to know?')).toHaveValue('My draft');
});
it('announces indexing and prevents duplicate submissions',async()=>{
 setup();let finish!:(response:Response)=>void;
 vi.mocked(fetch).mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));
 render(<DocumentLibrary/>);await screen.findByText('Library ready.');select(new File(['Fact.'],'violet.txt'));
 expect(screen.getByRole('status')).toHaveTextContent('Uploading and indexing violet.txt');
 expect(screen.getByRole('button',{name:'Indexing…'})).toBeDisabled();
 expect(screen.getByLabelText('Choose a document')).toBeDisabled();
 finish(new Response(JSON.stringify(doc),{status:201}));
 await screen.findByText('violet.txt is indexed and ready in Ask Copilot.');
});
it('shows an actionable library loading error',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new TypeError('network')));
 render(<DocumentLibrary/>);
 expect(await screen.findByRole('alert')).toHaveTextContent('Could not load the library');
 expect(screen.queryByText('Loading library…')).not.toBeInTheDocument();
});
it('waits for the initial listing before allowing uploads',async()=>{
 let finish!:(response:Response)=>void;
 vi.stubGlobal('fetch',vi.fn().mockReturnValueOnce(new Promise(resolve=>{finish=resolve;})));
 render(<DocumentLibrary/>);
 select(new File(['Fact.'],'violet.txt'));
 expect(screen.getByRole('button',{name:'Upload and index'})).toBeDisabled();
 expect(fetch).toHaveBeenCalledTimes(1);
 finish(new Response(JSON.stringify({documents:[]})));
 await screen.findByText('Library ready.');
 expect(screen.getByRole('button',{name:'Upload and index'})).toBeEnabled();
});
