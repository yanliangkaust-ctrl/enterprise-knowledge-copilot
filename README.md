# Enterprise Knowledge Copilot

A portfolio-ready foundation for grounded enterprise knowledge retrieval with RAG and knowledge graphs.

## Current scope

The current MVP includes:

- Home
- Document Library with eight interconnected Markdown sample documents
- Knowledge Explorer with a provenance-backed in-memory graph
- Ask Knowledge
- Evaluation with Keyword, Semantic, Graph, and Hybrid Recall@1/3/5 and MRR comparison

Document Library supports section-aware chunking, local vector retrieval, and a transparent keyword-search baseline. The preferred embedding model is the local `all-MiniLM-L6-v2` sentence-transformer.

Knowledge Explorer uses deterministic terminology and explicit phrase rules to build a local NetworkX graph. Ask Knowledge can retrieve vector, graph, or merged Hybrid evidence before generating a grounded Demo / Local response.

Neo4j, managed vector infrastructure, graph agents, and production persistence are intentionally out of scope.

The graph and vector indexes are intentionally in-memory and rebuilt for a fresh process. Graph extraction is rule-based and only emits relationships supported by the sample corpus; Graph-RAG, hybrid graph/vector answer strategies beyond retrieval, and production persistence are not included.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first Semantic retrieval run downloads `all-MiniLM-L6-v2` from Hugging Face and caches it locally. An explainable NumPy hashing fallback is used when the model package or model files are unavailable offline.

Run the application:

```powershell
streamlit run app.py
```
