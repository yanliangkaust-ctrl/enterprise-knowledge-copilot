"""Bounded in-memory conversational context for follow-up query resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.rag_schema import GroundedResponse
    from backend.knowledge_graph import KnowledgeGraph

from backend.entity_extraction import KNOWN_ENTITIES


_STOP_WORDS = {
    "a", "an", "and", "are", "be", "does", "for", "how", "in", "is", "of",
    "on", "or", "the", "to", "what", "when", "which", "with", "why", "who",
}
_REFERENCE_TERMS = {
    "that", "those", "this", "these", "it", "they", "them", "same", "previous",
}


@dataclass(frozen=True)
class EvidenceReference:
    source_document: str
    section: str
    chunk_id: str


@dataclass(frozen=True)
class SessionTurn:
    question: str
    answer: str
    grounding_status: str
    topics: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()
    operational_references: tuple[str, ...] = ()


@dataclass
class SessionContext:
    recent_turns: list[SessionTurn] = field(default_factory=list)
    current_topic: str = ""


def _terms(text: str) -> tuple[str, ...]:
    values = {
        token
        for token in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.casefold())
        if token not in _STOP_WORDS and len(token) > 2
    }
    return tuple(sorted(values))


def _known_entities(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    found = [name for name in KNOWN_ENTITIES if name.casefold() in lowered]
    found.extend(re.findall(r"\bR-\d{3}\b", text, flags=re.IGNORECASE))
    return tuple(dict.fromkeys(found))


def has_follow_up_reference(question: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", question.casefold()))
    return bool(tokens & _REFERENCE_TERMS) or "the risk" in question.casefold()


def build_relevant_context(question: str, context: SessionContext | None) -> str:
    """Return only compact prior anchors for an explicitly contextual question."""

    if context is None or not context.recent_turns or not has_follow_up_reference(question):
        return ""

    turn = context.recent_turns[-1]
    anchors = list(dict.fromkeys((*turn.entities, *turn.topics)))[:12]
    references = [
        f"{item.source_document} / {item.section} / {item.chunk_id}"
        for item in turn.evidence_references[:5]
    ]
    parts = [f"Prior conversational topic: {context.current_topic or 'recent enterprise question'}."]
    if anchors:
        parts.append("Prior entities and topics: " + ", ".join(anchors) + ".")
    if references:
        parts.append("Prior evidence references to revisit: " + "; ".join(references) + ".")
    if turn.operational_references:
        parts.append("Prior operational incident references: " + ", ".join(turn.operational_references) + ".")
    return " ".join(parts)


class InMemorySessionStore:
    """Small process-local store; history is context only, never enterprise evidence."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, SessionContext] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> SessionContext:
        with self._lock:
            context = self._sessions.get(session_id)
            if context is None:
                return SessionContext()
            return SessionContext(
                recent_turns=list(context.recent_turns),
                current_topic=context.current_topic,
            )

    def record(
        self,
        session_id: str,
        question: str,
        response: GroundedResponse,
        knowledge_graph: KnowledgeGraph | None = None,
        operational_references: tuple[str, ...] = (),
    ) -> None:
        evidence_text = " ".join(item.text for item in response.retrieved_evidence)
        entities = list(
            dict.fromkeys(
                (
                    *(
                        entity
                        for item in response.retrieved_evidence
                        for entity in item.entity_path
                    ),
                    *_known_entities(f"{question} {evidence_text}"),
                )
            )
        )
        if knowledge_graph is not None:
            for entity_name in tuple(entities):
                entity = knowledge_graph.entity(entity_name)
                if entity is None:
                    continue
                for relationship in knowledge_graph.relationships(entity.entity_id):
                    related_id = (
                        relationship.target_entity
                        if relationship.source_entity == entity.entity_id
                        else relationship.source_entity
                    )
                    related = knowledge_graph.entity(related_id)
                    if related is not None:
                        entities.append(related.name)
        turn = SessionTurn(
            question=question,
            answer=response.answer,
            grounding_status=response.grounding_status,
            topics=_terms(question),
            entities=tuple(dict.fromkeys(entities))[:12],
            evidence_references=tuple(
                EvidenceReference(item.source_document, item.section, item.chunk_id)
                for item in response.retrieved_evidence[:5]
            ),
            operational_references=operational_references,
        )
        with self._lock:
            context = self._sessions.setdefault(session_id, SessionContext())
            context.recent_turns.append(turn)
            context.recent_turns = context.recent_turns[-self.max_turns:]
            context.current_topic = ", ".join(turn.entities[:4] or turn.topics[:4])