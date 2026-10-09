export interface Evidence { rank: number; source_document: string; section: string; chunk_id: string; text: string; score: number; selection_reason: string; relationship_type: string; entity_path: string[] }
export interface Verification {version:string;status:string;coverage_complete:boolean;blocked:boolean;limitation:string;claims:{text:string;status:string;reason_code:string;critical:boolean;evidence_ids:string[]}[]}
export interface Review {required:boolean;status:string;reason_code:string;limitation:string}
export interface GraphProvenance {source_document:string;source_section:string;evidence_text:string;chunk_id:string}
export interface CorpusGraph {scope:string;truncated:boolean;nodes:{entity_id:string;name:string;entity_type:string;provenance:GraphProvenance[]}[];edges:{relationship_id:string;source_entity:string;target_entity:string;relationship_type:string;provenance:GraphProvenance}[]}
export interface QueryResponse {verification?:Verification;review?:Review; answer: string; selected_retrieval_route: string | null; evidence_sufficient: boolean; grounding_status: string; sufficiency_reason: string; sources: string[]; provenance: Evidence[]; session_id: string; execution_trace: { interpreted_intent: string; tool_decisions: {tool: string; reason: string}[]; retrieved_evidence_count: number; evidence_sufficient: boolean; sufficiency_reason: string; final_generation_mode: string; tool_executions: Record<string,string>[]; retrieval_checks: Record<string,unknown>[] }; operational_data: Record<string,unknown>[] }
export class ApiError extends Error { constructor(message: string, public status=0, public requestId: string|null=null) { super(message); } }
export async function queryApi(question: string, sessionId: string, signal: AbortSignal): Promise<QueryResponse> {
 const base=(import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
 let response: Response;
 try { response=await fetch(`${base}/query`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question,session_id:sessionId}),signal}); }
 catch (error) { if (signal.aborted) throw error; throw new ApiError('Could not reach the API. Check your connection or the API proxy / CORS configuration.'); }
 const requestId=response.headers.get('X-Request-ID');
 if(!response.ok) {
 const messages: Record<number,string>={422:'The request was not accepted. Use a question of 1–2,000 characters and start a new conversation if needed.',429:`Too many requests. ${response.headers.get('Retry-After') ? `Try again in ${response.headers.get('Retry-After')} seconds.` : 'Please wait before trying again.'}`,500:'The service could not complete this request. Please try again.',503:'The service is temporarily unavailable or starting up. Please wait and try again.'};
 throw new ApiError(messages[response.status] || 'The API returned an unexpected error.',response.status,requestId);
 }
 const data: unknown=await response.json().catch(()=>{throw new ApiError('The API returned an unreadable response.',502,requestId)});
 if (!data || typeof data!=='object') throw new ApiError('The API returned an incompatible response.',502,requestId);
 const d=data as QueryResponse;
 if(typeof d.answer!=='string' || typeof d.evidence_sufficient!=='boolean' || typeof d.grounding_status!=='string' || typeof d.sufficiency_reason!=='string' || typeof d.session_id!=='string' || !d.session_id || d.session_id.length>128 || !Array.isArray(d.sources) || !d.sources.every(s=>typeof s==='string') || !Array.isArray(d.provenance) || !d.provenance.every(e=>e && typeof e.text==='string' && typeof e.source_document==='string' && typeof e.section==='string' && typeof e.chunk_id==='string' && typeof e.rank==='number' && typeof e.score==='number')) throw new ApiError('The API returned an incompatible response.',502,requestId);
 if(d.verification && (typeof d.verification.status!=='string' || !Array.isArray(d.verification.claims) || !d.verification.claims.every(c=>typeof c.text==='string' && typeof c.status==='string' && Array.isArray(c.evidence_ids) && c.evidence_ids.every(id=>typeof id==='string')))) throw new ApiError('The API returned incompatible verification data.',502,requestId);
 if(d.review && (typeof d.review.status!=='string' || typeof d.review.required!=='boolean' || typeof d.review.limitation!=='string')) throw new ApiError('The API returned incompatible review data.',502,requestId);
 return d;
}
export function newSession(): string {return crypto.randomUUID();}

export async function getCorpusGraph(signal:AbortSignal):Promise<CorpusGraph>{
 const base=(import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/,'');
 const r=await fetch(`${base}/graph`,{signal});
 if(!r.ok)throw new ApiError('Corpus graph is unavailable on this API version.',r.status);
 const d=await r.json() as CorpusGraph;
 if(d.scope!=='bundled_synthetic_corpus' || !Array.isArray(d.nodes) || d.nodes.length>80 || !Array.isArray(d.edges) || d.edges.length>160)throw new ApiError('Invalid corpus graph response.');
 const ids=new Set(d.nodes.map(n=>n.entity_id));
 if(!d.nodes.every(n=>typeof n.entity_id==='string' && typeof n.name==='string') || !d.edges.every(e=>ids.has(e.source_entity)&&ids.has(e.target_entity)&&typeof e.relationship_type==='string'&&e.provenance&&typeof e.provenance.evidence_text==='string'))throw new ApiError('Invalid corpus graph provenance.');
 return d;
}
