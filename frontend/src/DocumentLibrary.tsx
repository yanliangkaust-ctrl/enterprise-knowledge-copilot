import {useEffect,useRef,useState} from 'react';
import {listDocuments,uploadDocument,type LibraryDocument} from './documentApi';

export default function DocumentLibrary(){
 const [documents,setDocuments]=useState<LibraryDocument[]>([]);
 const [file,setFile]=useState<File|null>(null);
 const [busy,setBusy]=useState(false);
 const [loading,setLoading]=useState(true);
 const [message,setMessage]=useState('Loading library…');
 const [error,setError]=useState('');
 const controller=useRef<AbortController|null>(null);
 useEffect(()=>{
  const c=new AbortController();controller.current=c;
  const timer=setTimeout(()=>c.abort(),90000);
  let mounted=true;
  listDocuments(c.signal).then(d=>{if(mounted){setDocuments(d.documents);setMessage('Library ready.');}}).catch(()=>{if(mounted){setMessage('');setError('Could not load the library. The API may be waking up or may need the upload release.');}}).finally(()=>{clearTimeout(timer);if(mounted)setLoading(false);});
  return()=>{mounted=false;clearTimeout(timer);c.abort();controller.current?.abort();};
 },[]);
 async function upload(event:React.FormEvent){
  event.preventDefault();if(!file || busy || loading)return;
  setError('');
  if(!/\.(pdf|txt|md)$/i.test(file.name) || !file.size || file.size>1024*1024){setError('Choose a nonempty PDF, TXT, or Markdown file no larger than 1 MiB.');return;}
  const c=new AbortController();controller.current=c;setBusy(true);setMessage(`Uploading and indexing ${file.name}…`);
  const timer=setTimeout(()=>c.abort(),90000);
  try{
   const indexed=await uploadDocument(file,c.signal);
   setDocuments(old=>[...old.filter(d=>d.filename!==indexed.filename),indexed]);
   setMessage(`${indexed.filename} is indexed and ready in Ask Copilot.`);
  }catch(e){setMessage('');setError(c.signal.aborted?'Request timed out. Indexing may have completed; reopen the library before retrying.':e instanceof Error?e.message:'Upload failed. Please retry.');}
  finally{clearTimeout(timer);setBusy(false);}
 }
 return <section className="document-library" aria-labelledby="library-heading"><h1 id="library-heading">Document Library</h1>
 <p>Upload PDF, UTF-8 TXT, or Markdown. Maximum 1 MiB and 200 KB extracted text; PDFs up to 20 pages. Scanned PDFs, encrypted PDFs, and DOCX are unsupported.</p>
 <p className="unavailable">Shared demo library: everyone using this API can query uploaded content. Use synthetic documents only. Uploads may be lost when Render Free sleeps, restarts, or redeploys.</p>
 <form onSubmit={upload}><label htmlFor="document-file">Choose a document</label><input id="document-file" type="file" accept=".pdf,.txt,.md" disabled={busy} onChange={e=>{setFile(e.target.files?.[0] || null);setError('');}}/>
 {file && <p>Selected: {file.name}</p>}<button className="ask-button" disabled={!file || busy || loading}>{busy?'Indexing…':'Upload and index'}</button></form>
 {message && <p role="status">{message}</p>}{error && <p role="alert" className="error">{error}</p>}
 <h2>Indexed documents</h2><ul>{documents.map(d=><li key={d.document_id || d.filename}><strong>{d.filename}</strong> <span>Ready · {d.chunk_count} chunks</span></li>)}</ul>
 </section>;
}
