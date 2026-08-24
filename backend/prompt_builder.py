"""Reusable grounded-generation prompts."""

from backend.context_builder import build_context
from backend.rag_schema import EvidenceItem


GROUNDED_SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer the user's question using only the supplied enterprise evidence.
Do not invent unsupported facts. If the evidence is insufficient, say so clearly.
Distinguish directly supported facts from reasonable inference.
Be concise and professional. Preserve source traceability by naming the source
documents used for the answer. Never create a citation that is not in the evidence.
"""


def build_grounded_prompt(question: str, evidence: list[EvidenceItem]) -> str:
    """Create the user-facing prompt with a traceable evidence context."""

    return (
        f"Enterprise question:\n{question.strip()}\n\n"
        "Use only this retrieved evidence:\n"
        f"{build_context(evidence)}\n\n"
        "Return a concise answer. State 'Insufficient evidence' if the sources "
        "do not support an answer."
    )
