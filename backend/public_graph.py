"""Bounded graph from bundled synthetic documents only, never uploaded knowledge."""
from functools import lru_cache
from pathlib import Path
from dataclasses import asdict
from backend.document_ingestion import load_markdown_documents, chunk_documents
from backend.graph_builder import build_knowledge_graph

@lru_cache(maxsize=1)
def corpus_graph():
    graph=build_knowledge_graph(chunk_documents(load_markdown_documents(Path(__file__).resolve().parents[1]/"data"/"sample_docs")))
    entities=graph.entities()[:80]
    ids={e.entity_id for e in entities}
    edges=sorted((r for r in graph.relationships() if r.source_entity in ids and r.target_entity in ids),key=lambda r:r.relationship_id)[:160]
    return {"scope":"bundled_synthetic_corpus", "nodes":[asdict(e) for e in entities],"edges":[asdict(r) for r in edges],"truncated":len(graph.entities())>80 or graph.relationship_count>len(edges)}
