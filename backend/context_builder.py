"""Build traceable, source-bounded context for grounded generation."""

from collections.abc import Sequence

from backend.rag_schema import EvidenceItem


def build_context(evidence: Sequence[EvidenceItem]) -> str:
    """Format evidence with explicit metadata and boundaries for an LLM."""

    if not evidence:
        return "No enterprise evidence was retrieved."

    blocks = []
    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"[SOURCE {item.rank}]",
                    f"Document: {item.source_document}",
                    f"Section: {item.section}",
                    f"Chunk ID: {item.chunk_id}",
                    f"Text: {item.text}",
                    "[/SOURCE]",
                ]
            )
        )
    return "\n\n".join(blocks)
