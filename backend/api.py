"""FastAPI adapter for the existing LangGraph-backed knowledge workflow."""

from __future__ import annotations

from contextlib import asynccontextmanager
import sqlite3
import os
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware
from backend.claim_verification import verify_answer, review_signal
from backend.public_graph import corpus_graph
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agentic_orchestrator import ask_agentic
from backend.document_ingestion import DocumentRecord
from backend.knowledge_base import get_shared_knowledge_base
from backend.knowledge_graph import KnowledgeGraph
from backend.rag_schema import EvidenceItem
from backend.semantic_retrieval import VectorIndex
from backend.resilience import ServiceFailure, boundary
from backend.observability import RequestObservabilityMiddleware, configure_json_logging, emit, stage
from backend.session_context import InMemorySessionStore
from backend.request_limits import ClientRateLimiter, QueryRateLimitMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)


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
    retrieval_checks: list[dict[str, object]] = Field(default_factory=list)


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
    verification: dict = Field(default_factory=dict)
    review: dict = Field(default_factory=dict)
    operational_data: list[dict[str, object]]


def build_knowledge_base() -> tuple[list[DocumentRecord], list, VectorIndex, KnowledgeGraph]:
    base = get_shared_knowledge_base()
    app.state.knowledge_base = base
    return base.snapshot()


@asynccontextmanager
async def lifespan(app):
    try:
        build_knowledge_base()
    except Exception:
        emit("application_error", stage="startup", status="error", error_code="startup_initialization_failed")
        raise ServiceFailure("startup_initialization_failed", 503) from None
    yield


app = FastAPI(title="Enterprise Knowledge Copilot API", version="0.1.0", lifespan=lifespan)
configure_json_logging()
# Explicit production origins only. Unset means cross-origin access is disabled.
origins = [v.strip() for v in os.getenv("FRONTEND_ORIGINS", "").split(",") if v.strip()]
if any(urlparse(v).scheme != "https" or not urlparse(v).netloc or urlparse(v).path or "*" in v or urlparse(v).query or urlparse(v).fragment or urlparse(v).username for v in origins):
    raise ValueError("FRONTEND_ORIGINS must contain exact HTTPS origins")
query_limiter = ClientRateLimiter()
app.add_middleware(QueryRateLimitMiddleware, limiter=query_limiter)
app.add_middleware(RequestObservabilityMiddleware)
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET","POST"], allow_headers=["Content-Type"], expose_headers=["X-Request-ID","Retry-After"], allow_credentials=False)

session_store = InMemorySessionStore()


@app.get("/graph")
def graph_overview():
    return corpus_graph()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="enterprise-knowledge-copilot")


@app.get("/ready")
def ready():
    base = getattr(app.state, "knowledge_base", None)
    if base is None:
        emit("readiness_check", status="error", reason_code="knowledge_uninitialized", initialized=False)
        return JSONResponse({"status": "not_ready", "reason_code": "knowledge_uninitialized"}, status_code=503)
    runtime_failure = getattr(base, "runtime_failure", None)
    if runtime_failure:
        emit("readiness_check", status="error", reason_code=runtime_failure, initialized=True)
        return JSONResponse({"status": "not_ready", "reason_code": runtime_failure}, status_code=503)
    available, code, metadata = base.readiness_status()
    emit("readiness_check", status="ok" if available else "error", reason_code=code, initialized=True, **metadata)
    return JSONResponse({"status": "ready" if available else "not_ready", "reason_code": code, **metadata},
                        status_code=200 if available else 503)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        with stage("knowledge_snapshot_request"):
            _, chunks, vector_index, knowledge_graph = build_knowledge_base()
    except (sqlite3.Error, OSError):
        raise ServiceFailure("storage_unavailable", 503) from None
    session_id = request.session_id or str(uuid4())
    try:
        result = ask_agentic(
            request.question,
            chunks,
            vector_index,
            knowledge_graph,
            session_context=session_store.get(session_id),
        )
    except ServiceFailure as exc:
        if exc.code in {"hybrid_retrieval_failed", "fallback_retrieval_failed", "generation_failed"}:
            base = getattr(app.state, "knowledge_base", None)
            if base is not None:
                base.runtime_failure = exc.code
        raise
    trace = result.trace
    route = trace.tool_decisions[0].tool if trace.tool_decisions else None

    response = QueryResponse(
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
            retrieval_checks=trace.retrieval_checks,
        ),
        session_id=session_id,
        verification=result.response.verification or verify_answer(result.response).model_dump(),
        review=review_signal(request.question),
        operational_data=result.operational_data,
    )

    response.model_dump_json()  # Validate serialization before committing conversational history.
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
    base = getattr(app.state, "knowledge_base", None)
    if base is not None:
        base.runtime_failure = None
    return response
