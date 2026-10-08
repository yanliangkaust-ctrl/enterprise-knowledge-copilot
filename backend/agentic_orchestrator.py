"""Lightweight, explainable agentic orchestration over existing retrieval tools.

This module deliberately avoids exposing chain-of-thought.  It records a concise
execution trace containing routing decisions, tool reasons, evidence counts and
sufficiency outcomes suitable for a product UI and interview demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections.abc import Sequence

from backend.answer_generation import evidence_items_from_results, generate_grounded_answer
from backend.document_ingestion import ChunkRecord, search_chunks
from backend.factual_support import explicit_ownership_support, unsupported_facets
from backend.graph_retrieval import match_question_entities, search_graph
from backend.hybrid_retrieval import search_hybrid
from backend.knowledge_graph import KnowledgeGraph
from backend.rag_schema import GroundedResponse
from backend.semantic_retrieval import VectorIndex
from backend.session_context import SessionContext
from backend.observability import decision, emit, timed
from backend.procedural_support import requested_operations, procedure_evidence


@dataclass(frozen=True)
class ToolDecision:
    tool: str
    reason: str


@dataclass(frozen=True)
class ToolExecution:
    tool: str
    input: str
    status: str
    source: str


@dataclass
class AgentExecutionTrace:
    interpreted_intent: str
    tool_decisions: list[ToolDecision] = field(default_factory=list)
    retrieved_evidence_count: int = 0
    evidence_sufficient: bool = False
    sufficiency_reason: str = ""
    final_generation_mode: str = "Demo / Local"
    tool_executions: list[ToolExecution] = field(default_factory=list)
    effective_question: str = ""
    retrieval_checks: list[dict[str, object]] = field(default_factory=list)


@dataclass
class AgenticResult:
    response: GroundedResponse
    trace: AgentExecutionTrace
    operational_data: list[dict[str, object]] = field(default_factory=list)


_GRAPH_TERMS = {
    "depend", "depends", "dependency", "dependencies", "relationship", "relationships",
    "related", "connect", "connected", "team", "teams", "owner", "owners", "involved",
    "affect", "affects", "risk", "risks", "impact", "impacts",
}
_MULTI_HOP_TERMS = {"and", "which", "what else", "why", "evidence", "supports", "before", "production"}
_EXACT_TERMS = {"endpoint", "endpoints", "api", "r-001", "port", "status", "version"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.casefold()))


def plan_tools(question: str, graph: KnowledgeGraph) -> tuple[str, list[ToolDecision]]:
    """Choose retrieval tools using transparent, deterministic routing heuristics."""

    tokens = _tokens(question)
    named_entities = match_question_entities(question, graph)
    graph_signal = bool(tokens & _GRAPH_TERMS) or bool(named_entities)
    exact_signal = bool(tokens & _EXACT_TERMS)
    multi_signal = len(tokens & _MULTI_HOP_TERMS) >= 2 or (graph_signal and len(named_entities) >= 1 and "and" in tokens)

    if graph_signal and multi_signal:
        return (
            "multi-hop dependency / evidence question",
            [ToolDecision("hybrid_search", "Combines semantic relevance with explicit graph relationships for a multi-part question.")],
        )
    if graph_signal:
        return (
            "entity / dependency relationship question",
            [ToolDecision("graph_search", "The question asks about named entities, dependencies, risks, teams, or relationships.")],
        )
    if exact_signal:
        return (
            "exact technical fact question",
            [ToolDecision("keyword_search", "Exact API, endpoint, identifier, or technical terminology is prominent in the question.")],
        )
    return (
        "semantic knowledge question",
        [ToolDecision("semantic_search", "The question is best matched by meaning rather than an explicit graph relationship or exact identifier.")],
    )


def _run_tool(
    tool: str,
    question: str,
    chunks: Sequence[ChunkRecord],
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    top_k: int,
):
    if tool == "keyword_search":
        return search_chunks(list(chunks), question, top_k=top_k)
    if tool == "graph_search":
        return search_graph(question, graph, chunks, top_k=top_k)
    if tool == "hybrid_search":
        return search_hybrid(question, chunks, vector_index, graph, top_k=top_k)
    return vector_index.search(question, top_k=top_k)


def _sufficiency_result(code, sufficient, reason):
    decision(code, sufficient)
    return sufficient, reason


@timed("evidence_sufficiency")
def _evidence_sufficient(question: str, response: GroundedResponse, results: Sequence[object], knowledge_graph: KnowledgeGraph | None = None) -> tuple[bool, str]:
    """Apply a conservative, explainable evidence gate after retrieval/generation."""

    if not results:
        return _sufficiency_result("no_evidence", False, "No evidence was retrieved by the selected tool.")
    if requested_operations(question) and not procedure_evidence(question, response.retrieved_evidence):
        return _sufficiency_result("missing_procedural_support", False, "Retrieved evidence does not support the requested procedure.")
    if response.grounding_status == "Insufficient evidence":
        return _sufficiency_result("missing_question_support", False, "Retrieved chunks did not contain enough question-relevant support.")

    # Graph questions should have explicit graph evidence, not semantic coincidence alone.
    q_tokens = _tokens(question)
    graph_supported = any(getattr(item, "relationship_type", "") for item in results)
    hybrid_supported = any(getattr(item, "graph_score", 0.0) > 0 for item in results)
    relationship_support = "graph evidence" if graph_supported or hybrid_supported else ""
    textual_ownership = False
    graph_resolvable = True
    if knowledge_graph is not None:
        graph_resolvable = any(knowledge_graph.relationships(entity.entity_id)
                               for entity in match_question_entities(question, knowledge_graph))
        if not graph_resolvable:
            textual_ownership = explicit_ownership_support(question, response.retrieved_evidence)
    if q_tokens & _GRAPH_TERMS:
        if not (graph_supported or hybrid_supported):
            if not textual_ownership:
                return _sufficiency_result("missing_relationship_support", False, "A relationship-oriented question requires explicit graph-supported evidence.")
            relationship_support = "explicit textual fallback evidence (graph relationships unresolved)"
        else:
            relationship_support = "graph evidence"

    # Multi-part questions must have evidence for each requested facet before the
    # agent declares the answer sufficient. This prevents a partially correct
    # answer from passing the evidence gate.
    evidence_text = " ".join(item.text for item in response.retrieved_evidence).casefold()
    asks_risk = bool(q_tokens & {"risk", "risks", "affect", "affects", "impact", "impacts"})
    asks_recipient = (
        "who" in q_tokens
        and bool(q_tokens & {"notify", "notified", "contact", "contacted", "alert", "alerted", "inform", "informed"})
    ) or {"escalation", "contact"} <= q_tokens
    asks_team = bool(q_tokens & {"team", "teams", "owner", "owners", "involved"}) or (
        "who" in q_tokens and not asks_recipient
    )
    asks_checks = bool(q_tokens & {"check", "checked", "before", "promoting", "promote", "production"})

    if asks_risk and not any(term in evidence_text for term in ("risk", "impact", "r-001", "r-002", "r-003", "failure")):
        return _sufficiency_result("missing_risk_support", False, "The question asks about risks, but the retrieved evidence does not cover a supported risk facet.")
    recipient_patterns = (
        r"\b[\w][\w .'-]{1,80}\s+(?:must|should)\s+be\s+(?:notified|contacted|alerted|informed)\b",
        r"\b(?:notify|contact|alert|inform)\s+(?:the\s+)?[\w][\w .'-]{1,80}\b",
        r"\b[a-z][\w .'-]{1,80}\s+is\s+(?:the\s+)?(?:(?:primary|backup|escalation)\s+){1,2}contact\b",
    )
    if asks_recipient and not any(
        re.search(pattern, item.text.casefold())
        for item in response.retrieved_evidence
        for pattern in recipient_patterns
    ):
        return _sufficiency_result("missing_recipient_support", False, "The question asks for a notification or escalation contact, but retrieved evidence does not identify one.")
    if asks_team and not textual_ownership and not any(term in evidence_text for term in ("platform engineering team", "security team", "operations team")):
        return _sufficiency_result("missing_team_support", False, "The question asks which teams are involved, but the retrieved evidence does not identify supported team ownership.")
    if asks_checks and not any(term in evidence_text for term in ("before production", "pre-deployment", "approve", "validation", "uat environment", "release")):
        return _sufficiency_result("missing_release_check_support", False, "The question asks about production checks, but the retrieved evidence does not cover a supported release-check facet.")

    missing = unsupported_facets(question, response.retrieved_evidence)
    if textual_ownership:
        # Only ownership may be satisfied by the stricter text-policy alternative;
        # every other recognized factual facet retains its existing checks.
        missing = [facet for facet in missing if facet != "ownership"]
        emit("factual_support_check", facet="ownership", supported=True, reason_code="explicit_textual_ownership")
    if missing:
        return _sufficiency_result("missing_factual_support", False, "Retrieved evidence does not support the requested factual facet(s): " + ", ".join(missing) + ".")

    reason = "Retrieved enterprise evidence covers the requested question facets and supports a grounded response."
    if relationship_support:
        reason += " Relationship support: " + relationship_support + "."
    elif textual_ownership:
        reason += " Relationship support: explicit textual fallback evidence (graph relationships unresolved)."
    emit("relationship_support", relationship_support="textual" if textual_ownership and not (graph_supported or hybrid_supported) else "graph" if relationship_support else "none")
    return _sufficiency_result("evidence_sufficient", True, reason)


def ask_agentic(
    question: str,
    chunks: Sequence[ChunkRecord],
    vector_index: VectorIndex,
    knowledge_graph: KnowledgeGraph,
    generation_mode: str = "Demo / Local",
    top_k: int = 5,
    session_context: SessionContext | None = None,
) -> AgenticResult:
    """Plan, retrieve, verify evidence, and generate a grounded answer."""
    from backend.langgraph_orchestrator import ask_agentic_langgraph

    return ask_agentic_langgraph(
        question,
        chunks,
        vector_index,
        knowledge_graph,
        generation_mode=generation_mode,
        top_k=top_k,
        session_context=session_context,
    )
