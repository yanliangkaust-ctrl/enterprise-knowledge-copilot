"""FastAPI adapter for the existing LangGraph-backed knowledge workflow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.agentic_orchestrator import ask_agentic
from backend.document_ingestion import DocumentRecord, chunk_documents, load_markdown_documents
from backend.graph_builder import build_knowledge_graph
from backend.knowledge_graph import KnowledgeGraph
from backend.rag_schema import EvidenceItem
from backend.semantic_retrieval import VectorIndex
from backend.session_context import InMemorySessionStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class ToolDecisionResponse(BaseModel):
    tool: str
    reason: str


class ExecutionTraceResponse(BaseModel):
    interpreted_intent: str
    tool_decisions: list[ToolDecisionResponse]
    retrieved_evidence_count: int
    evidence_sufficient: bool
    sufficiency_reason: str
    final_generation_mode: str
    tool_executions: list[dict[str, str]]


class QueryResponse(BaseModel):
    answer: str
    selected_retrieval_route: str | None
    evidence_sufficient: bool
    grounding_status: str
    sufficiency_reason: str
    sources: list[str]
    provenance: list[EvidenceItem]
    execution_trace: ExecutionTraceResponse
    session_id: str
    operational_data: list[dict[str, object]]


@lru_cache(maxsize=1)
def build_knowledge_base() -> tuple[list[DocumentRecord], list, VectorIndex, KnowledgeGraph]:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)
    vector_index = VectorIndex()
    vector_index.build(chunks)
    knowledge_graph = build_knowledge_graph(chunks)
    return documents, chunks, vector_index, knowledge_graph


app = FastAPI(title="Enterprise Knowledge Copilot API", version="0.1.0")
session_store = InMemorySessionStore()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="enterprise-knowledge-copilot")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    _, chunks, vector_index, knowledge_graph = build_knowledge_base()
    session_id = request.session_id or str(uuid4())
    result = ask_agentic(
        request.question,
        chunks,
        vector_index,
        knowledge_graph,
        session_context=session_store.get(session_id),
    )
    session_store.record(
        session_id,
        request.question,
        result.response,
        knowledge_graph,
        operational_references=tuple(
            str(item["incident_id"])
            for item in result.operational_data
            if item.get("incident_id")
        ),
    )
    trace = result.trace
    route = trace.tool_decisions[0].tool if trace.tool_decisions else None

    return QueryResponse(
        answer=result.response.answer,
        selected_retrieval_route=route,
        evidence_sufficient=trace.evidence_sufficient,
        grounding_status=result.response.grounding_status,
        sufficiency_reason=trace.sufficiency_reason,
        sources=result.response.sources,
        provenance=result.response.retrieved_evidence,
        execution_trace=ExecutionTraceResponse(
            interpreted_intent=trace.interpreted_intent,
            tool_decisions=[
                ToolDecisionResponse(tool=decision.tool, reason=decision.reason)
                for decision in trace.tool_decisions
            ],
            retrieved_evidence_count=trace.retrieved_evidence_count,
            evidence_sufficient=trace.evidence_sufficient,
            sufficiency_reason=trace.sufficiency_reason,
            final_generation_mode=trace.final_generation_mode,
            tool_executions=[execution.__dict__ for execution in trace.tool_executions],
        ),
        session_id=session_id,
        operational_data=result.operational_data,
    )