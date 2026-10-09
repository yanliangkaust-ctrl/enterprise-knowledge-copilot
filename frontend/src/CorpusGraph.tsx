import {useEffect,useState} from 'react';
import {getCorpusGraph,type CorpusGraph as Graph} from './api';
export default function CorpusGraph(){
 const [graph,setGraph]=useState<Graph>();const [error,setError]=useState('');const [selected,setSelected]=useState('');
 useEffect(()=>{const controller=new AbortController();getCorpusGraph(controller.signal).then(g=>{setGraph(g);setSelected(g.nodes[0]?.entity_id || '');}).catch(()=>{if(!controller.signal.aborted)setError('Corpus graph unavailable. The deployed API may not yet include /graph.');});return()=>controller.abort();},[]);
 if(error)return <p className="unavailable">{error}</p>;
 if(!graph)return <p role="status">Loading real corpus graph…</p>;
 const node=graph.nodes.find(n=>n.entity_id===selected);
 const edges=graph.edges.filter(e=>e.source_entity===selected || e.target_entity===selected).slice(0,12);
 const neighbors=[...new Set(edges.flatMap(e=>[e.source_entity,e.target_entity]))].filter(id=>id!==selected);
 const positions=new Map([[selected,{x:150,y:45}],...neighbors.map((id,i)=>[id,{x:55+(i%2)*190,y:120+Math.floor(i/2)*90}] as const)]);
 return <div><p className="inspector-note">Bundled synthetic corpus overview, independent of this answer. Select a node to inspect up to 12 adjacent source-backed edges.{graph.truncated?' API bounds truncate the corpus.':''}</p><label htmlFor="graph-node">Select corpus entity</label><select id="graph-node" value={selected} onChange={e=>setSelected(e.target.value)}>{graph.nodes.map(n=><option key={n.entity_id} value={n.entity_id}>{n.name}</option>)}</select>
 {node && <><svg viewBox={`0 0 300 ${Math.max(160,170+Math.ceil(neighbors.length/2)*90)}`} role="img" aria-label={`Corpus relationships adjacent to ${node.name}`} className="corpus-svg">{edges.map(e=>{const a=positions.get(e.source_entity)!;const b=positions.get(e.target_entity)!;return <line key={e.relationship_id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#91ab93"/>;})}{[selected,...neighbors].map(id=>{const p=positions.get(id)!;const n=graph.nodes.find(n=>n.entity_id===id)!;return <g key={id} role="button" tabIndex={0} aria-label={`Select ${n.name}`} onClick={()=>setSelected(id)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setSelected(id);}}}><circle cx={p.x} cy={p.y} r="15" fill={id===selected?'#285640':'#6a8a69'}/><text x={p.x} y={p.y+30} textAnchor="middle" fontSize="9" fill="#294d38">{n.name}</text></g>;})}</svg><h4>{node.name} · {node.entity_type}</h4>{!edges.length && <p>No adjacent edges returned for this node.</p>}{edges.map(e=><details className="inspector-source" key={e.relationship_id}><summary>{graph.nodes.find(n=>n.entity_id===e.source_entity)?.name} → {e.relationship_type} → {graph.nodes.find(n=>n.entity_id===e.target_entity)?.name}</summary><p>{e.provenance.source_document} · {e.provenance.source_section}</p><p className="verbatim">{e.provenance.evidence_text}</p><small>{e.provenance.chunk_id}</small></details>)}</>}
 </div>;
}
