"""Grounded answer generation with a deterministic local demo mode."""

import os
import re
from collections.abc import Sequence

from backend.prompt_builder import GROUNDED_SYSTEM_PROMPT, build_grounded_prompt
from backend.rag_schema import EvidenceItem, GroundedResponse
from backend.observability import emit, timed
from backend.procedural_support import requested_operations, procedure_evidence


STOP_WORDS = {
    "a", "an", "and", "are", "be", "does", "for", "how", "in", "is", "of",
    "on", "or", "the", "to", "what", "when", "which", "with", "why",
}


def evidence_items_from_results(results: Sequence[object]) -> list[EvidenceItem]:
    """Convert keyword or semantic retrieval results into the shared response schema."""

    items = []
    for result in results:
        chunk = result.chunk
        items.append(
            EvidenceItem(
                rank=result.rank,
                source_document=chunk.document_name,
                section=chunk.section_heading,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(result.score),
                selection_reason=getattr(result, "reason", ""),
                relationship_type=getattr(result, "relationship_type", ""),
                entity_path=list(getattr(result, "path", ())),
            )
        )
    return items


@timed("draft_generation")
def generate_grounded_answer(
    question: str,
    evidence: Sequence[EvidenceItem],
    mode: str = "Demo / Local",
    model: str = "gpt-4o-mini",
) -> GroundedResponse:
    """Generate a traceable answer locally or through the OpenAI Responses API."""

    evidence = list(evidence)
    if requested_operations(question):
        supported = procedure_evidence(question, evidence)
        if not supported:
            emit("generation_refusal", reason_code="missing_procedural_support")
            return GroundedResponse(answer="Insufficient evidence: the retrieved sources do not support the requested procedure.", sources=[], retrieved_evidence=evidence,
                                    grounding_status="Insufficient evidence", generation_mode="Demo / Local")
        # Extractive acceptance prevents a formatter/backend from substituting topical prose.
        return GroundedResponse(answer="\n\n".join(item.text for item in supported),
                                sources=list(dict.fromkeys(item.source_document for item in supported)),
                                retrieved_evidence=supported, grounding_status="Grounded in retrieved evidence", generation_mode="Demo / Local")
    if mode == "OpenAI" and os.getenv("OPENAI_API_KEY"):
        return _generate_openai_answer(question, evidence, model)
    return _generate_demo_answer(question, evidence, mode)


def _generate_demo_answer(
    question: str,
    evidence: list[EvidenceItem],
    mode: str,
) -> GroundedResponse:
    notification_result = _format_notification_answer(question, evidence)
    if notification_result is not None:
        answer, supporting_evidence = notification_result
        if not answer:
            emit("generation_refusal", reason_code="missing_recipient_support")
            return GroundedResponse(
                answer="Insufficient evidence: the retrieved enterprise sources do not identify a supported notification or escalation contact.",
                sources=[],
                retrieved_evidence=evidence,
                grounding_status="Insufficient evidence",
                generation_mode="Demo / Local",
            )
        return GroundedResponse(
            answer=answer,
            sources=list(dict.fromkeys(item.source_document for item in supporting_evidence)),
            retrieved_evidence=evidence,
            grounding_status="Grounded in retrieved evidence",
            generation_mode=mode,
        )

    supported = _supported_evidence(question, evidence)
    if not supported:
        emit("generation_refusal", reason_code="missing_question_support")
        return GroundedResponse(
            answer="Insufficient evidence: the retrieved enterprise sources do not support an answer to this question.",
            sources=[],
            retrieved_evidence=evidence,
            grounding_status="Insufficient evidence",
            generation_mode="Demo / Local",
        )

    answer = _format_demo_answer(question, supported)
    source_names = list(dict.fromkeys(item.source_document for item in supported))
    return GroundedResponse(
        answer=answer,
        sources=source_names,
        retrieved_evidence=evidence,
        grounding_status="Grounded in retrieved evidence",
        generation_mode=mode,
    )


def _format_notification_answer(
    question: str,
    evidence: list[EvidenceItem],
) -> tuple[str, list[EvidenceItem]] | None:
    """Answer recipient questions from evidence sentences naming the contact."""

    tokens = set(re.findall(r"[a-z0-9]+", question.casefold()))
    recipient_terms = {
        "notify", "notified", "notification", "contact", "contacted",
        "escalation", "escalate", "escalated", "alert", "alerted",
        "inform", "informed", "recipient",
    }
    if "who" not in tokens or not tokens.intersection(recipient_terms):
        return None
    prefers_backup = bool(tokens & {"backup", "unavailable"})
    prefers_primary = bool(tokens & {"first", "primary"})

    notification_pattern = re.compile(
        r"\b(?:the\s+)?(?P<recipient>[A-Z][\w&.'/-]*(?:\s+[A-Z][\w&.'/-]*){0,5})"
        r"\s+(?P<modal>must|should|needs to|has to)\s+be\s+"
        r"(?P<action>notified|contacted|alerted|informed|escalated)\b"
        r"(?P<detail>[^.!?]*)"
    )
    active_pattern = re.compile(
        r"\b(?P<action>notify|contact|alert|inform|escalate)\s+"
        r"(?P<recipient>(?:the\s+)?[A-Z][\w&.'/-]*(?:\s+[A-Z][\w&.'/-]*){0,5})"
        r"(?P<detail>[^.!?]*)",
        flags=re.IGNORECASE,
    )
    contact_pattern = re.compile(
        r"\b(?:(?:primary|backup|escalation)\s+){1,2}contact\b",
        flags=re.IGNORECASE,
    )

    notification_lines: list[tuple[str, EvidenceItem]] = []
    active_lines: list[tuple[str, EvidenceItem]] = []
    contact_lines: list[tuple[str, EvidenceItem]] = []
    for item in evidence:
        normalized_text = re.sub(r"\s*\n+\s*", " ", item.text.strip())
        sentences = re.split(r"(?<=[.!?])\s+", normalized_text)
        for sentence in sentences:
            sentence = _clean_line(sentence)
            if not sentence:
                continue
            match = notification_pattern.search(sentence)
            if match:
                line = (
                    f"The {match.group('recipient')} {match.group('modal')} be "
                    f"{match.group('action')}{match.group('detail')}"
                )
                if sentence.endswith((".", "!", "?")):
                    line += sentence[-1]
                notification_lines.append((line, item))
            else:
                match = active_pattern.search(sentence)
                if match:
                    line = (
                        f"{match.group('action').capitalize()} {match.group('recipient')}"
                        f"{match.group('detail')}"
                    )
                    active_lines.append((line, item))
                elif contact_pattern.search(sentence):
                    contact_lines.append((sentence, item))

    if prefers_backup and not prefers_primary:
        selected = contact_lines or notification_lines or active_lines
    else:
        selected = notification_lines or active_lines or contact_lines
    if not selected:
        return "", []

    lines = list(dict.fromkeys(line for line, _ in selected))
    supporting_evidence: list[EvidenceItem] = []
    seen_chunks: set[str] = set()
    for _, item in selected:
        if item.chunk_id not in seen_chunks:
            seen_chunks.add(item.chunk_id)
            supporting_evidence.append(item)
    return "\n".join(lines), supporting_evidence


def _clean_line(text: str) -> str:
    """Normalize markdown-ish evidence lines for clean bullet rendering."""
    text = text.strip()
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"^\d+\.\s*", "", text)
    text = text.replace("**", "")
    return text.strip()


def _matching_lines(evidence: list[EvidenceItem], patterns: tuple[str, ...], limit: int = 4) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        for raw in item.text.splitlines():
            line = _clean_line(raw)
            low = line.casefold()
            if line and any(pattern in low for pattern in patterns) and line not in seen:
                seen.add(line)
                matches.append(line)
                if len(matches) >= limit:
                    return matches
    return matches


def _format_demo_answer(question: str, evidence: list[EvidenceItem]) -> str:
    """Create a concise, scannable local-demo answer from retrieved evidence."""
    q = question.casefold()
    wants_risks = any(word in q for word in ("risk", "risks", "affect", "impact"))
    wants_teams = any(word in q for word in ("team", "teams", "owner", "owners", "involved", "who"))
    wants_actions = any(word in q for word in ("check", "checked", "before", "recommend", "action", "promot", "production"))

    sections: list[str] = []
    summary = _matching_lines(
        evidence,
        ("will deploy", "release path", "production deployment", "depends on", "platform flow"),
        limit=1,
    )
    if not summary:
        first = re.split(r"(?<=[.!?])\s+", evidence[0].text.strip())[0]
        summary = [_clean_line(first)] if first else []
    if summary:
        sections.append("**Summary**\n" + "\n".join(f"- {line}" for line in summary))

    if wants_risks:
        risks = _matching_lines(evidence, ("impact:", "risk", "pending", "failed", "capacity", "unauthorized", "alert"), limit=4)
        if risks:
            sections.append("**Key risks**\n" + "\n".join(f"- {line}" for line in risks))

    if wants_teams:
        teams = _matching_lines(evidence, ("platform engineering team", "security team", "operations team"), limit=5)
        if teams:
            sections.append("**Teams involved**\n" + "\n".join(f"- {line}" for line in teams))

    if wants_actions:
        actions = _matching_lines(evidence, ("must approve", "confirm ", "review ", "validate ", "resolved before", "rolls back", "mitigation:"), limit=4)
        if actions:
            sections.append("**Recommended checks / actions**\n" + "\n".join(f"- {line}" for line in actions))

    if len(sections) == 1:
        findings = []
        for item in evidence[:3]:
            sentence = re.split(r"(?<=[.!?])\s+", item.text.strip())[0]
            sentence = _clean_line(sentence)
            if sentence and sentence not in findings:
                findings.append(sentence)
        if findings:
            sections.append("**Key findings**\n" + "\n".join(f"- {line}" for line in findings))

    return "\n\n".join(sections)


def _supported_evidence(question: str, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    keywords = {
        word for word in re.findall(r"[a-z0-9]+", question.casefold()) if word not in STOP_WORDS and len(word) > 2
    }
    return [
        item
        for item in evidence
        if keywords.intersection(set(re.findall(r"[a-z0-9]+", item.text.casefold())))
    ]


def _generate_openai_answer(
    question: str,
    evidence: list[EvidenceItem],
    model: str,
) -> GroundedResponse:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=GROUNDED_SYSTEM_PROMPT,
        input=build_grounded_prompt(question, evidence),
    )
    return GroundedResponse(
        answer=response.output_text,
        sources=list(dict.fromkeys(item.source_document for item in evidence)),
        retrieved_evidence=evidence,
        grounding_status="Grounded in retrieved evidence" if evidence else "Insufficient evidence",
        generation_mode="OpenAI",
    )
