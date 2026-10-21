export interface LibraryDocument {filename:string;document_id:string|null;version:number|null;status:'ready';text_bytes:number;chunk_count:number}
const base=()=> (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/,'');
function validDocument(d:LibraryDocument){return d && typeof d.filename==='string' && d.status==='ready' && typeof d.chunk_count==='number';}
async function read(response:Response){
 if(!response.ok){
  const data=await response.json().catch(()=>null);
  throw new Error(typeof data?.detail==='string'?data.detail:`Library request failed (${response.status}). Please retry later.`);
 }
 return response.json();
}
export async function listDocuments(signal:AbortSignal):Promise<{documents:LibraryDocument[]}>{
 const data=await read(await fetch(`${base()}/documents`,{signal}));
 if(!Array.isArray(data.documents) || !data.documents.every(validDocument))throw new Error('Incompatible library response.');
 return data;
}
export async function uploadDocument(file:File,signal:AbortSignal):Promise<LibraryDocument>{
 const data=await read(await fetch(`${base()}/documents?filename=${encodeURIComponent(file.name)}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file,signal}));
 if(!validDocument(data))throw new Error('Incompatible upload response.');
 return data;
}
