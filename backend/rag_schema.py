"""Typed response objects shared by the RAG pipeline and UI."""

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    rank: int
    source_document: str
    section: str
    chunk_id: str
    text: str
    score: float
    selection_reason: str = ""
    relationship_type: str = ""
    entity_path: list[str] = Field(default_factory=list)


class GroundedResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    retrieved_evidence: list[EvidenceItem] = Field(default_factory=list)
    grounding_status: str
    generation_mode: str
