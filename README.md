# Enterprise Knowledge Copilot

An enterprise AI knowledge retrieval and grounded Q&A system that turns scattered technical documentation into **traceable, evidence-backed answers**.

The system combines **keyword search, semantic retrieval, knowledge graph retrieval, hybrid retrieval, and Retrieval-Augmented Generation (RAG)** in one interactive workspace.

---

## Why I Built This

Enterprise teams often have important knowledge spread across:

- Architecture decisions
- Requirements
- Incident reports
- Risk registers
- Deployment documentation
- Operational notes

Traditional keyword search can find exact terms, but it often struggles when:

- users describe the same concept using different words
- answers require information from multiple documents
- relationships between systems matter
- users need evidence behind an AI-generated answer

**Enterprise Knowledge Copilot** explores how semantic search, knowledge graphs, and RAG can work together to make enterprise knowledge easier to discover and verify.

---

## Product Capabilities

### Document Library

Loads and organizes an enterprise sample knowledge base.

The current demo contains:

- **8 documents**
- **46 sections**
- **9,520 characters**

Documents are processed into searchable sections while preserving source information.

### Semantic Retrieval

Uses sentence-transformer embeddings and vector similarity to find conceptually relevant evidence even when the user's wording differs from the source documents.

### Keyword Retrieval

Provides a transparent traditional search baseline for comparison with semantic retrieval.

### Knowledge Graph

Extracts entities and relationships from enterprise documents.

The graph supports:

- Entity lookup
- Relationship exploration
- Neighbor inspection
- Path traversal
- Source provenance
- Interactive visualization

Example:

```text
OCR Service
    ↓
Kubernetes
    ↓
R-001
```

Graph facts remain traceable to supporting source documents.

### Hybrid Retrieval

Combines semantic evidence with knowledge-graph evidence.

The application supports four retrieval modes:

- Keyword
- Semantic
- Graph
- Hybrid

### Grounded Q&A

The **Ask Knowledge** workspace retrieves evidence before generating an answer.

The pipeline follows:

```text
Question
   ↓
Retrieval
   ↓
Evidence
   ↓
Context Construction
   ↓
Grounded Prompt
   ↓
Answer Generation
   ↓
Answer + Sources + Evidence
```

If the available knowledge base does not contain enough information, the system can return:

> **Insufficient evidence**

instead of fabricating an answer.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
                         User Question
                               │
                    ┌──────────▼──────────┐
                    │ Retrieval Selector  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
      Keyword              Semantic              Graph
     Retrieval             Retrieval            Retrieval
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                         Hybrid Retrieval
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Retrieved Evidence  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Context Builder   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Prompt Builder    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer Generation   │
                    │ Demo / OpenAI Mode  │
                    └──────────┬──────────┘
                               │
                               ▼
                       Grounded Answer
                       + Sources
                       + Evidence
                       + Provenance
```

---

## Retrieval Evaluation

The retrieval strategies were evaluated against a **10-question ground-truth benchmark**.

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword | 30% | 80% | 90% | 53% |
| **Semantic** | **70%** | 90% | 90% | **82%** |
| Graph | 40% | 60% | 60% | 48% |
| **Hybrid** | 40% | **100%** | **100%** | 68% |

### Key Findings

Semantic retrieval substantially improved the ability to rank the correct evidence first:

```text
Keyword Recall@1:   30%
Semantic Recall@1:  70%
```

Hybrid retrieval achieved complete evidence coverage within the top results:

```text
Hybrid Recall@3: 100%
Hybrid Recall@5: 100%
```

The experiment revealed an important retrieval trade-off:

**Semantic retrieval** performed best at putting the correct evidence near the top.

**Hybrid retrieval** performed best at ensuring the correct evidence appeared somewhere within the top results.

**Graph retrieval** provided explicit relationship traversal and provenance, but did not outperform semantic retrieval as a general-purpose search method.

This suggests that knowledge graphs are most useful as a **complement to semantic retrieval rather than a replacement for it**.

---

## Evaluation by Question Type

Retrieval performance can also be examined across different information needs, including:

- Single-document retrieval
- Cross-document synthesis
- Relationship reasoning

This allows retrieval quality to be evaluated by task rather than relying only on one overall metric.

---

## Grounded RAG Design

The RAG architecture separates retrieval, context construction, prompting, and generation.

```text
RAG Pipeline
     │
     ├── Retrieval
     │
     ├── Context Builder
     │
     ├── Prompt Builder
     │
     ├── Answer Generation
     │
     └── Typed Response Schema
```

This modular structure makes the system easier to:

- Test
- Debug
- Evaluate
- Extend
- Explain

Generated answers are grounded in retrieved evidence rather than unrestricted model generation.

---

## Knowledge Graph

The Knowledge Explorer provides structured relationship discovery across the enterprise knowledge base.

The prototype supports:

- Deterministic entity extraction
- Relationship extraction
- Entity deduplication
- Graph construction
- Neighbor lookup
- Relationship lookup
- Path traversal
- Provenance inspection
- Graph visualization

A valid graph path was verified for:

```text
OCR Service → Kubernetes → R-001
```

Queries for nonexistent relationships return no fabricated graph paths.

---

## Product Design Principles

### Evidence Before Generation

The system retrieves evidence before generating an answer.

### Traceability

Users can inspect the documents and evidence supporting an answer.

### Multiple Retrieval Strategies

Different enterprise questions may benefit from different retrieval approaches.

### Graceful Failure

When evidence is insufficient, the system should say so rather than invent information.

### Measurable Quality

Retrieval strategies are evaluated using quantitative metrics instead of relying only on subjective answer quality.

---

## Understanding the Metrics

### Recall@K

Recall@K asks:

> Did the correct source appear somewhere within the top K retrieved results?

For example:

**Recall@3 = 100%**

means the correct evidence appeared within the first three results for every evaluated question.

### MRR — Mean Reciprocal Rank

MRR measures **how high the correct evidence appears in the ranking**.

A correct result at position 1 receives a better score than a correct result at position 3.

This helps distinguish:

- finding the correct evidence
- ranking the correct evidence first

---

## Tech Stack

### Application

- Python
- Streamlit

### Retrieval

- Keyword retrieval
- Sentence Transformers
- Vector similarity search
- Graph retrieval
- Hybrid retrieval

### Knowledge Graph

- NetworkX
- Deterministic entity extraction
- Rule-based relationship extraction

### RAG

- Modular context construction
- Grounded prompt generation
- Pydantic response schemas
- Deterministic Demo / Local generation
- Optional OpenAI integration

### Quality & Testing

- Pytest
- Retrieval benchmark
- Browser workflow validation
- Provenance validation

### Deployment

- Git
- GitHub
- Streamlit Community Cloud

---

## Testing

The project includes automated tests covering:

- Document ingestion
- Keyword retrieval
- Semantic retrieval
- Retrieval evaluation
- RAG behavior
- Insufficient-evidence handling
- Entity extraction
- Relationship extraction
- Knowledge graph construction
- Graph traversal
- Provenance
- Graph retrieval
- Hybrid retrieval
- Ranking behavior

Latest validated milestone:

> **25 automated tests passed**

Additional validation included:

- Python syntax checks
- All five application pages browser-tested
- Document Library loaded successfully
- Supported questions returned grounded answers
- Unsupported questions returned insufficient evidence
- Source/evidence traceability verified
- Graph provenance verified
- Valid graph path verified
- Unsupported graph query produced no fabricated match
- Semantic, Keyword, Graph, and Hybrid retrieval modes tested

---

## Project Structure

```text
EnterpriseKnowledgeCopilot/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── answer_generation.py
│   ├── context_builder.py
│   ├── prompt_builder.py
│   ├── rag_pipeline.py
│   ├── rag_schema.py
│   └── ...
│
├── data/
│   └── ...
│
├── frontend/
│
└── tests/
    ├── test_document_ingestion.py
    ├── test_graph.py
    ├── test_rag.py
    ├── test_retrieval.py
    └── test_sprint5.py
```

---

## Current Limitations

This project is an MVP / portfolio implementation rather than a production enterprise platform.

Current limitations include:

- Vector indexes are stored in memory
- Knowledge graph is stored in memory
- Graph extraction uses deterministic rules
- Graph retrieval relies on explicit entity names and aliases
- Hybrid weighting is intentionally simple
- No persistent vector database
- No Neo4j persistence layer
- No advanced neural reranking
- No production document-access controls
- Sample enterprise documents are used for the demonstration

These limitations are intentionally visible rather than hidden behind the demo.

---

## Future Roadmap

Potential production extensions include:

1. Persistent vector database
2. Neo4j-backed knowledge graph
3. Graph-RAG retrieval
4. Cross-encoder reranking
5. Production PDF and document ingestion
6. OCR ingestion pipelines
7. Access-control-aware retrieval
8. Retrieval observability and tracing
9. Larger evaluation datasets
10. Human feedback and answer-quality evaluation

---

## What This Project Demonstrates

This project demonstrates the end-to-end development of an AI knowledge product rather than only an LLM API integration.

### Product Thinking

Defining an enterprise knowledge problem, user workflow, evidence requirements, and measurable success criteria.

### Information Retrieval

Designing and comparing keyword, semantic, graph, and hybrid retrieval strategies.

### AI Engineering

Building an evidence-grounded RAG pipeline with controlled answer generation.

### Knowledge Representation

Representing enterprise entities, relationships, graph paths, and provenance.

### Scientific Evaluation

Comparing retrieval strategies using Recall@K, MRR, ground-truth questions, and question-type analysis.

### Software Engineering

Building a modular Python architecture with typed schemas, automated tests, Git version control, and cloud deployment.

---

## Status

**Portfolio MVP — Functional and Deployed**

Core retrieval, RAG, knowledge graph, hybrid search, evaluation, provenance, automated testing, and Streamlit deployment are implemented.