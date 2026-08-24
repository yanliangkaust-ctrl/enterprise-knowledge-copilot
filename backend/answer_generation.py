"""Grounded answer generation with a deterministic local demo mode."""

import os
import re
from collections.abc import Sequence

from backend.prompt_builder import GROUNDED_SYSTEM_PROMPT, build_grounded_prompt
from backend.rag_schema import EvidenceItem, GroundedResponse


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


def generate_grounded_answer(
    question: str,
    evidence: Sequence[EvidenceItem],
    mode: str = "Demo / Local",
    model: str = "gpt-4o-mini",
) -> GroundedResponse:
    """Generate a traceable answer locally or through the OpenAI Responses API."""

    evidence = list(evidence)
    if mode == "OpenAI" and os.getenv("OPENAI_API_KEY"):
        return _generate_openai_answer(question, evidence, model)
    return _generate_demo_answer(question, evidence, mode)


def _generate_demo_answer(
    question: str,
    evidence: list[EvidenceItem],
    mode: str,
) -> GroundedResponse:
    supported = _supported_evidence(question, evidence)
    if not supported:
        return GroundedResponse(
            answer="Insufficient evidence: the retrieved enterprise sources do not support an answer to this question.",
            sources=[],
            retrieved_evidence=evidence,
            grounding_status="Insufficient evidence",
            generation_mode="Demo / Local",
        )

    statements = []
    for item in supported[:2]:
        first_sentence = re.split(r"(?<=[.!?])\s+", item.text.strip())[0]
        statements.append(first_sentence)
    source_names = list(dict.fromkeys(item.source_document for item in supported))
    return GroundedResponse(
        answer=" ".join(statements),
        sources=source_names,
        retrieved_evidence=evidence,
        grounding_status="Grounded in retrieved evidence",
        generation_mode=mode,
    )


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
