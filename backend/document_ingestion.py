"""Small, dependency-free helpers for loading and searching Markdown documents."""

from dataclasses import dataclass
from pathlib import Path
import re

from backend.observability import emit, timed


DOCUMENT_TYPES = {
    "ADR_": "Architecture Decision Record",
    "API_": "API Specification",
    "Incident_": "Incident Report",
    "OCR_": "Architecture Document",
    "Risk_": "Risk Register",
    "Security_": "Security Requirements",
    "Deployment_": "Deployment Guide",
    "UAT_": "UAT Report",
}


@dataclass(frozen=True)
class DocumentRecord:
    """A Markdown document and the metadata needed by the document library."""

    name: str
    document_type: str
    raw_text: str
    section_headings: list[str]
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class ChunkRecord:
    """A section-aware retrieval unit with its source document reference."""

    chunk_id: str
    document_name: str
    document_type: str
    section_heading: str
    text: str
    source: str


@dataclass(frozen=True)
class KeywordChunkResult:
    """A keyword-ranked chunk used for comparison with vector retrieval."""

    chunk: ChunkRecord
    score: int
    rank: int



def _document_type(filename: str) -> str:
    for prefix, document_type in DOCUMENT_TYPES.items():
        if filename.startswith(prefix):
            return document_type
    return "Markdown Document"



def parse_markdown_document(path: Path) -> DocumentRecord:
    """Read one Markdown file and extract lightweight document metadata."""

    raw_text = path.read_text(encoding="utf-8")
    return parse_markdown_content(path.name, raw_text)


def parse_markdown_content(filename: str, raw_text: str) -> DocumentRecord:
    """Parse Markdown text while retaining its original filename as provenance."""

    section_headings = [
        match.group(1).strip()
        for match in re.finditer(r"^#{1,6}\s+(.+?)\s*$", raw_text, flags=re.MULTILINE)
    ]
    words = re.findall(r"\b\w+\b", raw_text)
    metadata: dict[str, str | int] = {
        "filename": filename,
        "file_size_bytes": len(raw_text.encode("utf-8")),
        "word_count": len(words),
        "character_count": len(raw_text),
        "section_count": len(section_headings),
    }
    return DocumentRecord(
        name=filename,
        document_type=_document_type(filename),
        raw_text=raw_text,
        section_headings=section_headings,
        metadata=metadata,
    )



def load_markdown_documents(directory: str | Path) -> list[DocumentRecord]:
    """Load all Markdown files in a directory in stable filename order."""

    directory_path = Path(directory)
    return [
        parse_markdown_document(path)
        for path in sorted(directory_path.glob("*.md"))
    ]


def chunk_documents(documents: list[DocumentRecord]) -> list[ChunkRecord]:
    """Create one or more chunks per Markdown section without slicing sentences."""

    chunks: list[ChunkRecord] = []
    for document in documents:
        document_chunk_number = 0
        sections = re.split(r"^(#{1,6})\s+(.+?)\s*$", document.raw_text, flags=re.MULTILINE)
        if len(sections) <= 1:
            sections = ["", "", document.raw_text]
        for index in range(1, len(sections), 3):
            heading = sections[index + 1].strip()
            text = sections[index + 2].strip()
            if not text:
                continue
            document_chunk_number += 1
            chunk_id = f"{document.name}::chunk-{len(chunks) + 1:03d}"
            if "document_id" in document.metadata:
                chunk_id = f"{document.metadata['document_id']}::v{document.metadata['version']}::chunk-{document_chunk_number:03d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_name=document.name,
                    document_type=document.document_type,
                    section_heading=heading,
                    text=text,
                    source=document.name,
                )
            )
    return chunks



def search_documents(documents: list[DocumentRecord], query: str) -> list[DocumentRecord]:
    """Return documents containing query keywords, ranked by keyword matches."""

    keywords = [
        keyword.casefold()
        for keyword in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", query, flags=re.IGNORECASE)
    ]
    if not keywords:
        return documents.copy()

    ranked: list[tuple[int, DocumentRecord]] = []
    for document in documents:
        searchable_text = document.raw_text.casefold()
        score = sum(searchable_text.count(keyword) for keyword in keywords)
        if score:
            ranked.append((score, document))

    ranked.sort(key=lambda item: (-item[0], item[1].name.casefold()))
    return [document for _, document in ranked]


@timed("keyword_retrieval")
def search_chunks(chunks: list[ChunkRecord], query: str, top_k: int = 5) -> list[KeywordChunkResult]:
    """Return keyword-ranked chunks using the same transparent baseline strategy."""

    keywords = [
        keyword.casefold()
        for keyword in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", query, flags=re.IGNORECASE)
    ]
    if not keywords or top_k <= 0:
        return []

    ranked = [
        (sum(chunk.text.casefold().count(keyword) for keyword in keywords), chunk)
        for chunk in chunks
    ]
    ranked = [(score, chunk) for score, chunk in ranked if score]
    ranked.sort(key=lambda item: (-item[0], item[1].document_name.casefold(), item[1].chunk_id))
    emit("retrieval_candidates", retriever="keyword_search", candidate_count=len(ranked),
         selected_count=min(len(ranked), top_k), effective_k=top_k)
    return [
        KeywordChunkResult(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(ranked[:top_k], start=1)
    ]
