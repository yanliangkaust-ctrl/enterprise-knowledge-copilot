"""LangGraph adapter around the existing deterministic agentic pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.answer_generation import evidence_items_from_results, generate_grounded_answer
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
        return {
            "tool_decisions": decisions,
            "results": [],
            "evidence": [],
            "tool_executions": executions,
            "operational_results": operational_results,
        }

    primary = next(decision for decision in decisions if decision.tool != "incident_status_tool")
    retrieval_top_k = max(state["top_k"], 20) if state.get("selected_context") else state["top_k"]
    results = _run_tool(
        primary.tool,
        state["effective_question"],
        state["chunks"],
        state["vector_index"],
        state["knowledge_graph"],
        retrieval_top_k,
    )

    if primary.tool == "graph_search" and not results:
        decisions.append(
            ToolDecision(
                "hybrid_search",
                "Graph search returned no direct evidence, so hybrid retrieval broadens recall while retaining graph signals.",
            )
        )
        results = _run_tool(
            "hybrid_search",
            state["effective_question"],
            state["chunks"],
            state["vector_index"],
            state["knowledge_graph"],
            retrieval_top_k,
        )

    return {
        "tool_decisions": decisions,
        "results": results,
        "evidence": evidence_items_from_results(results),
        "tool_executions": executions,
        "operational_results": operational_results,
    }


def evaluate_evidence(state: AgentGraphState) -> dict[str, Any]:
    if state.get("retrieval_requested"):
        draft_response = generate_grounded_answer(
            state["effective_question"],
            state["evidence"],
            mode=state["generation_mode"],
        )
        sufficient, reason = _evidence_sufficient(
            state["question"],
            draft_response,
            state["results"],
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
    return {
        "draft_response": draft_response,
        "evidence_sufficient": sufficient,
        "sufficiency_reason": reason,
    }


def _evidence_route(state: AgentGraphState) -> str:
    return "generate_answer" if state["evidence_sufficient"] else "return_insufficient_evidence"


def generate_answer(state: AgentGraphState) -> dict[str, Any]:
    response = state["draft_response"]
    operational = state.get("operational_results", [])
    if operational:
        operational_text = "\n\n".join(format_incident_status(item) for item in operational)
        response = response.model_copy(update={
            "answer": f"{response.answer}\n\n{operational_text}".strip(),
            "generation_mode": "Demo / Local + Simulated operational data",
        })
    return {"response": response}


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
