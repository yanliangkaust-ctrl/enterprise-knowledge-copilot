# Enterprise Knowledge Copilot

An enterprise AI knowledge assistant that turns technical documents into **grounded, traceable answers** using semantic search, RAG, knowledge graphs, and hybrid retrieval.

## 🚀 Live Demo

[Open Enterprise Knowledge Copilot](https://enterprise-knowledge-copilot-2w95w5w5viaj6sbm5ltpvw.streamlit.app/)

> Click **Load Sample Knowledge Base** to initialize the demo, then explore the Document Library, Knowledge Explorer, Ask Knowledge, and Evaluation workspaces.

---

## What It Does

Enterprise knowledge is often scattered across requirements, architecture decisions, incident reports, risks, and deployment documentation.

Enterprise Knowledge Copilot brings that information into one evidence-focused workflow:

- 📚 Document ingestion and section-level indexing
- 🔎 Keyword and semantic retrieval
- 🧠 Retrieval-Augmented Generation (RAG)
- 🕸️ Knowledge graph exploration
- ⚡ Hybrid semantic + graph retrieval
- 🔍 Source and evidence provenance
- 📊 Quantitative retrieval evaluation
- 🚫 Insufficient-evidence handling

The core principle:

> **Retrieve evidence first. Generate answers second.**

---

## Architecture

```text
Enterprise Documents
        ↓
 Parsing & Chunking
        ↓
 ┌──────┴───────┐
 ↓              ↓
Semantic      Knowledge
Retrieval      Graph
 ↓              ↓
 └──────┬───────┘
        ↓
 Hybrid Retrieval
        ↓
 Retrieved Evidence
        ↓
 Context + Grounded Prompt
        ↓
       RAG
        ↓
 Grounded Answer
 + Sources + Evidence
```

The application supports four retrieval strategies:

**Keyword · Semantic · Graph · Hybrid**

---

## Evaluation Results

Retrieval strategies were evaluated against a 10-question ground-truth benchmark.

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword | 30% | 80% | 90% | 53% |
| **Semantic** | **70%** | 90% | 90% | **82%** |
| Graph | 40% | 60% | 60% | 48% |
| **Hybrid** | 40% | **100%** | **100%** | 68% |

### Key Findings

**Semantic retrieval improved Recall@1 from 30% to 70%** compared with the keyword baseline.

**Hybrid retrieval achieved 100% Recall@3 and Recall@5**, providing the strongest evidence coverage.

The results suggest that graph retrieval is most useful as a **complement to semantic retrieval**, rather than a replacement for it.

---

## Grounded Q&A

The RAG workflow keeps generated answers connected to retrieved evidence:

```text
Question
   ↓
Retrieve Evidence
   ↓
Build Context
   ↓
Grounded Prompt
   ↓
Generate Answer
   ↓
Answer + Sources + Evidence
```

When the knowledge base does not contain enough supporting information, the system can return:

> **Insufficient evidence**

rather than fabricate an answer.

---

## Knowledge Graph

The Knowledge Explorer supports:

- Entity and relationship extraction
- Graph visualization
- Neighbor exploration
- Relationship lookup
- Path traversal
- Source provenance

Example verified path:

```text
OCR Service → Kubernetes → R-001
```

Graph relationships remain traceable to their supporting evidence.

---

## Tech Stack

**AI / Retrieval:** Sentence Transformers, semantic vector search, RAG, hybrid retrieval

**Knowledge Graph:** NetworkX, entity and relationship extraction

**Application:** Python, Streamlit, Pydantic

**Generation:** Demo / Local mode with optional OpenAI integration

**Quality:** Pytest, Recall@K, MRR, provenance validation

**Deployment:** GitHub, Streamlit Community Cloud

---

## Testing

**25 automated tests passed**, covering:

- Document ingestion
- Keyword and semantic retrieval
- RAG behavior
- Insufficient-evidence handling
- Knowledge graph construction
- Graph traversal
- Graph and hybrid retrieval
- Provenance
- Evaluation metrics

The five application workspaces were also browser-tested.

---

## Product Design Principles

**Grounded** — answers originate from retrieved enterprise evidence.

**Traceable** — users can inspect supporting sources.

**Measurable** — retrieval quality is evaluated quantitatively.

**Explainable** — graph relationships and evidence remain inspectable.

**Safe failure** — unsupported questions should not produce fabricated evidence.

---

## Current Limitations

This is a portfolio MVP rather than a production enterprise platform.

Current limitations include:

- In-memory vector and graph indexes
- Deterministic graph extraction
- Simple hybrid weighting
- No persistent vector database
- No Neo4j persistence
- No advanced neural reranking
- Sample enterprise documents used for the demo

---

## Future Work

Potential extensions include:

- Persistent vector database
- Neo4j-backed knowledge graph
- Graph-RAG
- Cross-encoder reranking
- Production document ingestion
- Access-control-aware retrieval
- Retrieval observability
- Larger evaluation benchmarks

---

## Project Status

**Functional Portfolio MVP — Deployed**

`Document Ingestion → Semantic Retrieval → RAG → Knowledge Graph → Hybrid Retrieval → Evaluation → Cloud Deployment`

### Live Application

[Launch the Enterprise Knowledge Copilot](https://enterprise-knowledge-copilot-2w95w5w5viaj6sbm5ltpvw.streamlit.app/)