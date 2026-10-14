"""Bounded graph from bundled synthetic documents only, never uploaded knowledge."""
from functools import lru_cache
from pathlib import Path
from dataclasses import asdict
import re
from backend.document_ingestion import load_markdown_documents, chunk_documents
from backend.graph_builder import build_knowledge_graph

def visualization_flow(text, names):
    """Conservative positive sentence templates; no inferred dependencies.

    Anaphoric 'the gateway' is resolved only inside an explicit submit/send
    pattern. Ordering connects deployments, never a subsequent verification.
    This helper is used only by the public visualization, not retrieval.
    """
    observations=[]
    entity='(?:'+ '|'.join(re.escape(n) for n in sorted(names,key=len,reverse=True))+')'
    for sentence in re.split(r'(?<=[.!?])\s+',text):
        m=re.fullmatch(rf'Deploy ({entity} configuration) first, then ({entity} workers), then verify [^.]+\.',sentence)
        if m:
            observations.append((m[1],'DEPLOY_BEFORE',m[2]))
        m=re.fullmatch(rf'(?:The )?({entity}) stores the source and result in ({entity}) and publishes processing status for the ({entity})\.',sentence)
        if m:
            observations.extend([(m[1],'STORES_SOURCE_AND_RESULT_IN',m[2]),(m[1],'PUBLISHES_STATUS_FOR',m[3])])
    m=re.search(rf'(?:^|(?<=[.!?])\s+)A client submits a document through the ({entity})\. The gateway validates the request and sends the job to the ({entity})\.',text)
    if m and m[1].endswith('Gateway'):
        observations.extend([('Client','SUBMITS_DOCUMENT_THROUGH',m[1]),(m[1],'SENDS_JOB_TO',m[2])])
    return observations

@lru_cache(maxsize=1)
def corpus_graph():
    chunks=chunk_documents(load_markdown_documents(Path(__file__).resolve().parents[1]/"data"/"sample_docs"))
    graph=build_knowledge_graph(chunks)
    entities=graph.entities()[:80]
    ids={e.entity_id for e in entities}
    edges=sorted((r for r in graph.relationships() if r.source_entity in ids and r.target_entity in ids),key=lambda r:r.relationship_id)[:160]
    nodes={e.entity_id:asdict(e) for e in entities}
    # Visualization observations only. Do not alter the retrieval graph. The
    # legacy incident rule conflates platform with cause and infers UAT impact
    # from a summary that never mentions UAT; omit those edges from this API.
    visible=[asdict(e) for e in edges if not (
        e.relationship_type=='CAUSED_BY' or
        (e.relationship_type=='AFFECTS' and 'worker pods reached their memory limit' in e.provenance.evidence_text))]
    from backend.entity_extraction import normalize_entity_id
    for chunk in chunks:
        incidents=[e for e in entities if e.entity_type=='Incident' and any(
            p.chunk_id==chunk.chunk_id for p in e.provenance)]
        p=dict(source_document=chunk.document_name,source_section=chunk.section_heading,
               evidence_text=chunk.text,chunk_id=chunk.chunk_id)
        observations=visualization_flow(chunk.text,[e.name for e in entities])
        for match in re.finditer(r'([A-Z][\w-]* worker pods) reached their (memory limit)',chunk.text) if incidents else []:
            observations.extend([(incidents[0].name,'REPORTS',match[1]),(match[1],'REACHED',match[2])])
        if incidents and re.search(r'\bjobs stayed pending\b',chunk.text):
            observations.append((incidents[0].name,'REPORTS','pending jobs'))
        for match in re.finditer(r'\b([A-Z][\w-]*(?: [A-Z][\w-]*)*) remained available',chunk.text) if incidents else []:
            observations.append((incidents[0].name,'REPORTS_AVAILABLE',re.sub(r'^The ', '', match[1])))
        for source,kind,target in observations:
            for name in (source,target):
                identity=normalize_entity_id(name)
                nodes.setdefault(identity,dict(entity_id=identity,name=name,entity_type='Observation',provenance=[p]))
            visible.append(dict(relationship_id=f'{normalize_entity_id(source)}|{kind}|{normalize_entity_id(target)}|{chunk.chunk_id}',source_entity=normalize_entity_id(source),target_entity=normalize_entity_id(target),relationship_type=kind,provenance=p))
    bounded=list(nodes.values())[:80]
    allowed={n['entity_id'] for n in bounded}
    visible=sorted((e for e in visible if e['source_entity'] in allowed and e['target_entity'] in allowed),key=lambda e:e['relationship_id'])
    return {"scope":"bundled_synthetic_corpus", "nodes":bounded,"edges":visible[:160],"truncated":len(nodes)>80 or len(visible)>160}
