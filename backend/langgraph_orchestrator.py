"""LangGraph adapter around the existing deterministic agentic pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.answer_generation import _supported_evidence, evidence_items_from_results, generate_grounded_answer
from backend.factual_support import unsupported_facets
from backend.claim_verification import verify_answer
from backend.procedural_support import requested_operations, procedure_evidence
from backend.resilience import ServiceFailure, boundary, degraded
from backend.observability import decision, emit, record_metadata, stage, timed
from backend.agentic_orchestrator import (
    AgentExecutionTrace,
    AgenticResult,
    ToolDecision,
    ToolExecution,
    _evidence_sufficient,
    _run_tool,
    plan_tools,
)
from backend.document_ingestion import ChunkRecord
from backend.knowledge_graph import KnowledgeGraph
from backend.rag_schema import EvidenceItem, GroundedResponse
from backend.semantic_retrieval import VectorIndex
from backend.session_context import SessionContext, build_relevant_context
from backend.operational_service import (
    IncidentStatus,
    extract_incident_id,
    format_incident_status,
    get_incident_status,
)


class AgentGraphState(TypedDict, total=False):
    question: str
    effective_question: str
    session_context: SessionContext
    selected_context: str
    route_mode: str
    retrieval_requested: bool
    operational_requested: bool
    incident_id: str
    chunks: Sequence[ChunkRecord]
    vector_index: VectorIndex
    knowledge_graph: KnowledgeGraph
    generation_mode: str
    top_k: int
    interpreted_intent: str
    tool_decisions: list[ToolDecision]
    results: list[object]
    evidence: list[EvidenceItem]
    draft_response: GroundedResponse
    response: GroundedResponse
    evidence_sufficient: bool
    sufficiency_reason: str
    trace: AgentExecutionTrace
    tool_executions: list[ToolExecution]
    operational_results: list[IncidentStatus]
    retrieval_checks: list[dict[str, object]]


@timed("context")
def build_context(state: AgentGraphState) -> dict[str, Any]:
    selected_context = build_relevant_context(
        state["question"], state.get("session_context")
    )
    effective_question = state["question"]
    if selected_context:
        effective_question = f"{state['question']} {selected_context}"
    return {
        "effective_question": effective_question,
        "selected_context": selected_context,
    }


@timed("routing")
def route_query(state: AgentGraphState) -> dict[str, Any]:
    question = state["effective_question"]
    tokens = set(question.casefold().split())
    incident_id = extract_incident_id(question)
    operational_signal = bool(incident_id) and bool(
        tokens & {"status", "owner", "owns", "current", "incident", "action", "updated"}
    )
    combined_signal = operational_signal and bool(
        tokens & {"cause", "caused", "why", "impact", "root", "reason"}
    )

    if operational_signal:
        route_mode = "combined" if combined_signal else "operational"
        intent = "operational incident status question"
        decisions = [ToolDecision(
            "incident_status_tool",
            "The question identifies an incident and requests current operational status or ownership.",
        )]
        if route_mode == "combined":
            retrieval_intent, retrieval_decisions = plan_tools(state["question"], state["knowledge_graph"])
            intent = retrieval_intent + " plus operational incident status"
            decisions = retrieval_decisions + decisions
    else:
        route_mode = "retrieval"
        intent, decisions = plan_tools(state["question"], state["knowledge_graph"])
    return {
        "interpreted_intent": intent,
        "tool_decisions": decisions,
        "route_mode": route_mode,
        "retrieval_requested": route_mode in {"retrieval", "combined"},
        "operational_requested": route_mode in {"operational", "combined"},
        "incident_id": incident_id or "",
    }


def _graph_weakness(question: str, effective_question: str, results: Sequence[object], observation: dict | None = None) -> str | None:
    """Bounded pre-generation check using existing relevance and factual checks.

Nonempty graph matches can still lack text support. This is a fallback trigger,
not permission to answer: the unchanged final sufficiency gate always runs.
"""
    def checked(code, reason):
        if observation is not None:
            observation["reason_code"] = code
        emit("graph_support_check", reason_code=code, supported=reason is None)
        return reason

    if not results:
        return checked("graph_empty", "Graph search returned no direct evidence.")
    evidence = evidence_items_from_results(results)
    if requested_operations(question) and not procedure_evidence(question, evidence):
        return checked("graph_missing_procedural_support", "Graph evidence lacks support for the requested procedure.")
    if not _supported_evidence(effective_question, evidence):
        return checked("graph_weak_text_support", "Graph evidence has no question-relevant textual support.")
    missing = unsupported_facets(question, evidence)
    if missing:
        return checked("graph_missing_fact_support", "Graph evidence lacks requested factual support: " + ", ".join(missing) + ".")
    return checked("graph_support_adequate", None)


@timed("retrieval_orchestration")
def retrieve_tool(state: AgentGraphState) -> dict[str, Any]:
    decisions = list(state["tool_decisions"])
    executions: list[ToolExecution] = []
    operational_results: list[IncidentStatus] = []
    if state.get("operational_requested"):
        incident_id = state.get("incident_id", "")
        status = get_incident_status(incident_id) if incident_id else None
        if status is not None:
            operational_results.append(status)
        executions.append(ToolExecution(
            "get_incident_status",
            incident_id or "unknown",
            "found" if status else "not_found",
            "simulated operational data",
        ))

    if not state.get("retrieval_requested"):
        record_metadata(original_route="incident_status_tool", final_route="incident_status_tool", selected_count=0,
                        fallback_activated=False)
        return {
            "tool_decisions": decisions,
            "results": [],
            "evidence": [],
            "tool_executions": executions,
            "operational_results": operational_results,
        }

    primary = next(decision for decision in decisions if decision.tool != "incident_status_tool")
    retrieval_top_k = max(state["top_k"], 20) if state.get("selected_context") else state["top_k"]
    record_metadata(original_route=primary.tool, requested_k=state["top_k"], effective_k=retrieval_top_k)
    degraded_route = None
    try:
        with boundary({"semantic_search": "vector_retrieval_failed", "vector_search": "vector_retrieval_failed"}.get(primary.tool, primary.tool.replace("_search", "_retrieval_failed"))), stage("primary_retrieval"):
            results = _run_tool(
                primary.tool,
                state["effective_question"],
                state["chunks"],
                state["vector_index"],
                state["knowledge_graph"],
                retrieval_top_k,
            )
    except ServiceFailure as exc:
        if primary.tool not in {"graph_search", "semantic_search", "vector_search"}:
            raise
        degraded_route = "semantic_search" if primary.tool == "graph_search" else "graph_search"
        with boundary("fallback_retrieval_failed", 503), stage("fallback_retrieval"):
            results = _run_tool(degraded_route, state["effective_question"], state["chunks"],
                                state["vector_index"], state["knowledge_graph"], retrieval_top_k)
        if not results:
            raise ServiceFailure("fallback_retrieval_failed", 503)
        degraded(exc.code, "graph" if primary.tool == "graph_search" else "vector", degraded_route)
        decisions.append(ToolDecision(degraded_route, "Independent retriever used after component failure."))
    emit("retrieval_completed", retriever=degraded_route or primary.tool, selected_count=len(results),
         effective_k=retrieval_top_k, min_score=min((r.score for r in results), default=None),
         max_score=max((r.score for r in results), default=None))

    weakness = None
    retrieval_checks = []
    if primary.tool == "graph_search" and not degraded_route:
        observation = {}
        weakness = _graph_weakness(state["question"], state["effective_question"], results, observation)
        retrieval_checks.append({
            "reason_code": observation["reason_code"],
            "original_mode": "graph_search",
            "status": "empty" if not results else "weak_support" if weakness else "adequate_support",
            "reason": weakness or "Graph evidence passes existing text-relevance and requested-facet checks.",
            "original_evidence": [e.model_dump() for e in evidence_items_from_results(results)],
            "fallback_activated": bool(weakness),
            "final_mode": "hybrid_search" if weakness else "graph_search",
        })
    if weakness:
        emit("graph_fallback", reason_code=observation["reason_code"], fallback_activated=True,
             original_route=primary.tool, final_route="hybrid_search")
        decisions.append(
            ToolDecision(
                "hybrid_search",
                weakness + " Activating hybrid fallback while retaining graph signals; final evidence checks still apply.",
            )
        )
        with boundary("fallback_retrieval_failed", 503), stage("fallback_retrieval"):
            results = _run_tool(
                "hybrid_search",
                state["effective_question"],
                state["chunks"],
                state["vector_index"],
                state["knowledge_graph"],
                retrieval_top_k,
            )
        emit("retrieval_completed", retriever="hybrid_search", selected_count=len(results),
             effective_k=retrieval_top_k, min_score=min((r.score for r in results), default=None),
             max_score=max((r.score for r in results), default=None))

    record_metadata(final_route=degraded_route or ("hybrid_search" if weakness else primary.tool), selected_count=len(results),
                    fallback_activated=bool(weakness or degraded_route))
    return {
        "tool_decisions": decisions,
        "results": results,
        "evidence": evidence_items_from_results(results),
        "tool_executions": executions,
        "operational_results": operational_results,
        "retrieval_checks": retrieval_checks,
    }


@timed("evidence_evaluation")
def evaluate_evidence(state: AgentGraphState) -> dict[str, Any]:
    if state.get("retrieval_requested"):
        with boundary("generation_failed"):
            draft_response = generate_grounded_answer(
                state["effective_question"],
                state["evidence"],
                mode=state["generation_mode"],
            )
        with boundary("sufficiency_internal_error"):
            sufficient, reason = _evidence_sufficient(
                state["question"],
                draft_response,
                state["results"],
                state["knowledge_graph"],
            )
    else:
        draft_response = GroundedResponse(
            answer="",
            sources=[],
            retrieved_evidence=[],
            grounding_status="Operational data" if state.get("operational_results") else "Insufficient evidence",
            generation_mode="Simulated operational data",
        )
        sufficient = bool(state.get("operational_results"))
        reason = (
            "Simulated operational data was found for the requested incident."
            if sufficient
            else "No simulated operational record exists for the requested incident."
        )

    if state.get("route_mode") == "combined" and not state.get("operational_results"):
        sufficient = False
        reason = "No simulated operational record exists for the requested incident."
    if not state.get("retrieval_requested") or (state.get("route_mode") == "combined" and not state.get("operational_results")):
        decision("operational_record_found" if sufficient else "operational_record_not_found", sufficient)
    return {
        "draft_response": draft_response,
        "evidence_sufficient": sufficient,
        "sufficiency_reason": reason,
    }


def _evidence_route(state: AgentGraphState) -> str:
    return "generate_answer" if state["evidence_sufficient"] else "return_insufficient_evidence"


@timed("response_finalization")
def generate_answer(state: AgentGraphState) -> dict[str, Any]:
    response = state["draft_response"]
    operational = state.get("operational_results", [])
    operational_text = ""
    if operational:
        operational_text = "\n\n".join(format_incident_status(item) for item in operational)
        response = response.model_copy(update={
            "answer": f"{response.answer}\n\n{operational_text}".strip(),
            "generation_mode": "Demo / Local + Simulated operational data",
        })
    with boundary("verification_failed"):
        verification = verify_answer(response, operational_text)
    response = response.model_copy(update={"verification": verification.model_dump()})
    if verification.blocked:
        response = response.model_copy(update={"answer": "Insufficient evidence: critical answer claims could not be verified against cited evidence.", "sources": [], "grounding_status": "Insufficient evidence"})
        return {"response": response, "evidence_sufficient": False, "sufficiency_reason": "Post-generation verification blocked unsupported or unevaluated critical claims."}
    return {"response": response}


@timed("refusal_finalization")
def return_insufficient_evidence(state: AgentGraphState) -> dict[str, Any]:
    draft_response = state["draft_response"]
    response = draft_response
    if draft_response.grounding_status != "Insufficient evidence" or not draft_response.answer:
        response = GroundedResponse(
            answer="Insufficient evidence: the retrieved enterprise sources do not provide enough supported evidence for this question.",
            sources=[],
            retrieved_evidence=state["evidence"],
            grounding_status="Insufficient evidence",
            generation_mode=draft_response.generation_mode,
        )
    return {"response": response}


@timed("trace_finalization")
def finalize_trace(state: AgentGraphState) -> dict[str, Any]:
    response = state["response"]
    return {
        "trace": AgentExecutionTrace(
            interpreted_intent=state["interpreted_intent"],
            tool_decisions=state["tool_decisions"],
            retrieved_evidence_count=len(state["evidence"]),
            evidence_sufficient=state["evidence_sufficient"],
            sufficiency_reason=state["sufficiency_reason"],
            final_generation_mode=response.generation_mode,
            tool_executions=state.get("tool_executions", []),
            effective_question=state["effective_question"],
            retrieval_checks=state.get("retrieval_checks", []),
        )
    }


def build_agent_graph():
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("build_context", build_context)
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve_tool", retrieve_tool)
    workflow.add_node("evaluate_evidence", evaluate_evidence)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("return_insufficient_evidence", return_insufficient_evidence)
    workflow.add_node("finalize_trace", finalize_trace)

    workflow.add_edge(START, "build_context")
    workflow.add_edge("build_context", "route_query")
    workflow.add_edge("route_query", "retrieve_tool")
    workflow.add_edge("retrieve_tool", "evaluate_evidence")
    workflow.add_conditional_edges(
        "evaluate_evidence",
        _evidence_route,
        {
            "generate_answer": "generate_answer",
            "return_insufficient_evidence": "return_insufficient_evidence",
        },
    )
    workflow.add_edge("generate_answer", "finalize_trace")
    workflow.add_edge("return_insufficient_evidence", "finalize_trace")
    workflow.add_edge("finalize_trace", END)
    return workflow.compile()


@timed("orchestration")
def ask_agentic_langgraph(
    question: str,
    chunks: Sequence[ChunkRecord],
    vector_index: VectorIndex,
    knowledge_graph: KnowledgeGraph,
    generation_mode: str = "Demo / Local",
    top_k: int = 5,
    session_context: SessionContext | None = None,
) -> AgenticResult:
    state = build_agent_graph().invoke(
        {
            "question": question,
            "session_context": session_context,
            "chunks": chunks,
            "vector_index": vector_index,
            "knowledge_graph": knowledge_graph,
            "generation_mode": generation_mode,
            "top_k": top_k,
        }
    )
    return AgenticResult(
        response=state["response"],
        trace=state["trace"],
        operational_data=[item.__dict__ for item in state.get("operational_results", [])],
    )
