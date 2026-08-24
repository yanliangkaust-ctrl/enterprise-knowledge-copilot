import streamlit as st
import json
import os
from pathlib import Path

from backend.document_ingestion import (
    DocumentRecord,
    chunk_documents,
    load_markdown_documents,
    search_chunks,
    search_documents,
)
from backend.retrieval_evaluation import evaluate_retrieval
from backend.rag_pipeline import ask_knowledge
from backend.semantic_retrieval import RetrievalResult, VectorIndex
from backend.graph_builder import build_knowledge_graph
from backend.knowledge_graph import KnowledgeGraph
from backend.graph_retrieval import search_graph
from backend.hybrid_retrieval import search_hybrid


SAMPLE_DOCS_DIR = Path(__file__).parent / "data" / "sample_docs"
GROUND_TRUTH_PATH = Path(__file__).parent / "data" / "evaluation" / "ground_truth_questions.json"


@st.cache_resource(show_spinner="Preparing local evidence index...")
def build_knowledge_base() -> tuple[list[DocumentRecord], list, VectorIndex, KnowledgeGraph]:
    """Build the local knowledge base once per Streamlit process."""

    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)
    vector_index = VectorIndex()
    vector_index.build(chunks)
    knowledge_graph = build_knowledge_graph(chunks)
    return documents, chunks, vector_index, knowledge_graph


def load_knowledge_base() -> None:
    """Load the cached knowledge base into the current Streamlit session."""

    documents, chunks, vector_index, knowledge_graph = build_knowledge_base()
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.vector_index = vector_index
    st.session_state.knowledge_graph = knowledge_graph


st.set_page_config(
    page_title="Enterprise Knowledge Copilot",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --navy: #10253f;
        --navy-deep: #0b1b30;
        --ink: #172b3f;
        --muted: #66788a;
        --line: #dce5eb;
        --paper: #f5f7f8;
        --white: #ffffff;
        --teal: #159a9c;
        --blue: #3478b9;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: var(--navy-deep);
        min-width: 290px;
        max-width: 290px;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 30px 20px 22px; }
    [data-testid="stSidebar"] * { color: #dbe8f0; }
    [data-testid="stSidebar"] .stRadio > label { display: none; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin: 0; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
    [data-testid="stSidebar"] [role="radio"] {
        padding: 11px 14px;
        border-radius: 7px;
        border: 1px solid transparent;
        transition: background 120ms ease;
    }
    [data-testid="stSidebar"] [role="radio"]:hover { background: #173552; }
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
        background: #1d4663;
        border-color: #31627c;
        box-shadow: inset 3px 0 0 var(--teal);
    }
    [data-testid="stSidebar"] [role="radio"] > div:first-child { display: none; }
    [data-testid="stSidebar"] [role="radio"] p { font-size: 14px; font-weight: 600; }
    .brand { display: flex; gap: 12px; align-items: center; margin-bottom: 34px; }
    .brand-mark {
        width: 35px; height: 35px; border: 1px solid #57b5b0; border-radius: 8px;
        display: grid; place-items: center; background: #163c55; position: relative;
    }
    .brand-mark:before, .brand-mark:after { content: ''; position: absolute; background: #70c8bd; border-radius: 2px; }
    .brand-mark:before { width: 17px; height: 5px; transform: rotate(-35deg); }
    .brand-mark:after { width: 5px; height: 17px; transform: rotate(-35deg); }
    .brand-name { font-family: 'Manrope', sans-serif; font-size: 14px; font-weight: 800; line-height: 1.2; color: #f3f8fa; }
    .brand-subtitle { color: #91adbd; font-size: 11px; line-height: 1.3; margin-top: 3px; }
    .nav-label { color: #6f8da1; font-size: 10px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 10px; }
    .status-panel { border: 1px solid #29455b; border-radius: 8px; background: #102a43; padding: 14px; margin-top: 30px; }
    .status-title { color: #eff8f9; font-size: 12px; font-weight: 700; margin-bottom: 10px; }
    .status-row { display: flex; align-items: center; gap: 7px; color: #91adbd; font-size: 11px; margin-top: 7px; }
    .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #55c2a8; }
    .main-wrap { max-width: 1190px; margin: 0 auto; padding: 40px 44px 64px; }
    .eyebrow { color: var(--teal); font-size: 11px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: 10px; }
    .page-title { font-family: 'Manrope', sans-serif; color: var(--navy); font-size: 31px; font-weight: 800; letter-spacing: -.025em; line-height: 1.15; margin-bottom: 10px; }
    .page-lede { color: var(--muted); font-size: 15px; line-height: 1.6; max-width: 690px; margin-bottom: 30px; }
    .hero { background: var(--white); border: 1px solid var(--line); border-radius: 10px; padding: 31px 34px; box-shadow: 0 7px 22px rgba(23, 43, 63, .045); margin-bottom: 22px; }
    .hero-kicker { color: var(--blue); font-size: 12px; font-weight: 700; margin-bottom: 11px; }
    .hero h2 { font-family: 'Manrope', sans-serif; color: var(--navy); font-size: 25px; margin: 0 0 10px; letter-spacing: -.02em; }
    .hero p { color: var(--muted); font-size: 14px; line-height: 1.55; max-width: 620px; margin: 0; }
    .section-heading { font-family: 'Manrope', sans-serif; color: var(--navy); font-size: 17px; font-weight: 700; margin: 28px 0 13px; }
    .metric-card, .workflow-card { background: var(--white); border: 1px solid var(--line); border-radius: 8px; padding: 18px; min-height: 112px; box-shadow: 0 4px 15px rgba(23, 43, 63, .035); }
    .metric-label { color: var(--muted); font-size: 12px; margin-bottom: 11px; }
    .metric-value { color: var(--navy); font-family: 'Manrope', sans-serif; font-size: 24px; font-weight: 800; }
    .metric-note { color: #81909d; font-size: 11px; margin-top: 7px; }
    .workflow-card { min-height: 145px; position: relative; }
    .step { color: var(--teal); font-family: 'Manrope', sans-serif; font-size: 12px; font-weight: 800; margin-bottom: 13px; }
    .workflow-title { color: var(--navy); font-size: 14px; font-weight: 700; margin-bottom: 7px; }
    .workflow-copy { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .future-note { border-left: 3px solid #72b8b7; background: #edf6f5; color: #416d70; padding: 12px 15px; border-radius: 0 6px 6px 0; font-size: 12px; margin-top: 25px; }
    .module-heading { font-family: 'Manrope', sans-serif; color: var(--navy); font-size: 27px; margin: 0 0 8px; }
    .module-copy { color: var(--muted); margin-bottom: 25px; }
    @media (max-width: 720px) { .main-wrap { padding: 28px 20px 48px; } .page-title { font-size: 26px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark"></div>
            <div>
                <div class="brand-name">Enterprise Knowledge<br>Copilot</div>
                <div class="brand-subtitle">Grounded enterprise knowledge retrieval</div>
            </div>
        </div>
        <div class="nav-label">Workspace</div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Home",
        "Document Library",
        "Knowledge Explorer",
        "Ask Knowledge",
        "Evaluation",
    ]
    selected_page = st.radio("Workspace navigation", pages, key="workspace_navigation", label_visibility="collapsed")

    st.markdown(
        """
        <div class="status-panel">
            <div class="status-title">Environment status</div>
            <div class="status-row"><span class="status-dot"></span>Portfolio MVP</div>
            <div class="status-row"><span class="status-dot"></span>Local / Demo Mode</div>
            <div class="status-row"><span class="status-dot"></span>No external LLM calls yet</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

if selected_page == "Home":
    st.markdown('<div class="eyebrow">Workspace overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Turn scattered knowledge into confident decisions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-lede">A single workspace for finding the evidence behind technical and business decisions. Start by understanding the current knowledge base, then move from source material to grounded answers.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">GET STARTED</div>
            <h2>Build your evidence foundation</h2>
            <p>Your next best step is to review the document library. As the product evolves, this space will connect source documents, relationships, and answers into one traceable workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Knowledge base snapshot</div>', unsafe_allow_html=True)
    metric_columns = st.columns(3)
    metrics = [
        ("Documents indexed", "0", "Ready for your first source"),
        ("Knowledge entities", "0", "Relationships will appear here"),
        ("Evaluated answers", "0", "Quality tracking is not active yet"),
    ]
    for column, (label, value, note) in zip(metric_columns, metrics):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Your path through the product</div>', unsafe_allow_html=True)
    workflow_columns = st.columns(4)
    workflow = [
        ("01", "Prepare evidence", "Bring the right documents into a shared knowledge base."),
        ("02", "Explore context", "See how topics, systems, and teams connect."),
        ("03", "Ask with confidence", "Get answers grounded in the sources that matter."),
        ("04", "Improve quality", "Measure usefulness, coverage, and trust over time."),
    ]
    for column, (step, title, copy) in zip(workflow_columns, workflow):
        with column:
            st.markdown(f'<div class="workflow-card"><div class="step">{step}</div><div class="workflow-title">{title}</div><div class="workflow-copy">{copy}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="future-note">Demo mode is active. The interface is ready for document ingestion, retrieval, and evaluation capabilities in a future iteration.</div>', unsafe_allow_html=True)
elif selected_page == "Document Library":
    st.markdown('<div class="eyebrow">Evidence workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-heading">Document Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-copy">Review the source material that will ground future enterprise answers. This demo uses local Markdown documents only.</div>', unsafe_allow_html=True)

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if st.button("Load Sample Knowledge Base", type="primary"):
        load_knowledge_base()
        st.rerun()

    documents: list[DocumentRecord] = st.session_state.documents
    if documents:
        chunks = st.session_state.chunks
        vector_index: VectorIndex = st.session_state.vector_index
        knowledge_graph: KnowledgeGraph = st.session_state.knowledge_graph
        metric_columns = st.columns(3)
        library_metrics = [
            ("Total documents", str(len(documents)), "Markdown sources loaded"),
            ("Total text length", f"{sum(len(document.raw_text) for document in documents):,}", "Characters across sources"),
            ("Sections discovered", str(sum(len(document.section_headings) for document in documents)), "Headings available for later retrieval"),
        ]
        for column, (label, value, note) in zip(metric_columns, library_metrics):
            with column:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Retrieve evidence</div>', unsafe_allow_html=True)
        controls = st.columns([1, 2, 1])
        with controls[0]:
            retrieval_mode = st.radio("Retrieval mode", ["Keyword", "Semantic"], key="library_retrieval_mode", horizontal=True)
        with controls[1]:
            query = st.text_input("Search query", placeholder="Try: How is batch capacity monitored?", label_visibility="collapsed")
        with controls[2]:
            top_k = st.selectbox("Top K", [1, 3, 5], index=2, key="library_top_k")

        if retrieval_mode == "Keyword":
            retrieved_results = search_chunks(chunks, query, top_k=top_k)
            visible_documents = search_documents(documents, query)
            score_label = "keyword matches"
        else:
            retrieved_results = vector_index.search(query, top_k=top_k)
            visible_documents = [document for document in documents if document.name in {result.chunk.document_name for result in retrieved_results}]
            score_label = "cosine similarity"

        st.caption(f"{len(visible_documents)} documents with evidence" if query else "Enter a question to retrieve evidence")
        if retrieved_results:
            st.markdown(f'<div class="section-heading">Retrieved evidence · {score_label}</div>', unsafe_allow_html=True)
            for result in retrieved_results:
                score = result.score if isinstance(result, RetrievalResult) else result.score
                chunk = result.chunk
                st.markdown(f"**{result.rank}. {chunk.document_name}** · `{chunk.section_heading}` · score `{score:.3f}`")
                st.write(chunk.text)
                st.divider()

        document_options = {document.name: document for document in visible_documents}
        if document_options:
            st.markdown('<div class="section-heading">Source document preview</div>', unsafe_allow_html=True)
            selected_name = st.selectbox("Preview document", list(document_options), key="library_preview_document", label_visibility="collapsed")
            selected_document = document_options[selected_name]
            preview_columns = st.columns([1, 2])
            with preview_columns[0]:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Document type</div><div class="metric-value" style="font-size:16px">{selected_document.document_type}</div><div class="metric-note">Status: Loaded</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-heading" style="margin-top:20px">Sections</div>', unsafe_allow_html=True)
                for heading in selected_document.section_headings:
                    st.markdown(f"- {heading}")
            with preview_columns[1]:
                st.markdown('<div class="section-heading" style="margin-top:0">Document preview</div>', unsafe_allow_html=True)
                st.code(selected_document.raw_text, language="markdown")
        else:
            st.warning("No documents match that keyword search.")
    else:
        st.markdown('<div class="future-note">No documents loaded yet. Load the sample knowledge base to begin reviewing interconnected platform evidence.</div>', unsafe_allow_html=True)
elif selected_page == "Knowledge Explorer":
    st.markdown('<div class="eyebrow">Explainable graph workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-heading">Knowledge Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-copy">Inspect how systems, teams, risks, and documents connect. Every graph fact links back to source evidence.</div>', unsafe_allow_html=True)

    if "documents" not in st.session_state or not st.session_state.documents:
        if st.button("Load Sample Knowledge Base", type="primary"):
            load_knowledge_base()
            st.rerun()
        st.info("Load the sample knowledge base to explore the knowledge graph.")
    else:
        knowledge_graph: KnowledgeGraph = st.session_state.knowledge_graph
        metric_columns = st.columns(3)
        graph_metrics = [
            ("Total entities", str(len(knowledge_graph)), "Deduplicated graph nodes"),
            ("Total relationships", str(knowledge_graph.relationship_count), "Explicit corpus relationships"),
            ("Provenance records", str(sum(len(entity.provenance) for entity in knowledge_graph.entities())), "Source references retained"),
        ]
        for column, (label, value, note) in zip(metric_columns, graph_metrics):
            with column:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Entity search</div>', unsafe_allow_html=True)
        entity_query = st.text_input("Find an entity", placeholder="Try: OCR Service, Kubernetes, or R-001", label_visibility="collapsed")
        matching_entities = [
            entity for entity in knowledge_graph.entities()
            if entity_query.casefold() in entity.name.casefold()
        ]
        if matching_entities:
            selected_entity_name = st.selectbox("Select entity", [entity.name for entity in matching_entities], key="graph_selected_entity", label_visibility="collapsed")
            selected_entity = knowledge_graph.entity(selected_entity_name)
            if selected_entity:
                detail_columns = st.columns([1, 2])
                with detail_columns[0]:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Entity type</div><div class="metric-value" style="font-size:18px">{selected_entity.entity_type}</div><div class="metric-note">{len(selected_entity.provenance)} provenance records</div></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-heading" style="margin-top:20px">Direct neighbors</div>', unsafe_allow_html=True)
                    for neighbor in knowledge_graph.neighbors(selected_entity.entity_id):
                        st.markdown(f"- **{neighbor.name}** · {neighbor.entity_type}")
                with detail_columns[1]:
                    st.markdown('<div class="section-heading" style="margin-top:0">Relationships</div>', unsafe_allow_html=True)
                    relationship_rows = []
                    for relationship in knowledge_graph.relationships(selected_entity.entity_id):
                        other_id = relationship.target_entity if relationship.source_entity == selected_entity.entity_id else relationship.source_entity
                        other = knowledge_graph.entity(other_id)
                        relationship_rows.append({
                            "Direction": "outbound" if relationship.source_entity == selected_entity.entity_id else "inbound",
                            "Relationship": relationship.relationship_type,
                            "Entity": other.name if other else other_id,
                            "Source": relationship.provenance.source_document,
                            "Section": relationship.provenance.source_section,
                        })
                    st.dataframe(relationship_rows, hide_index=True, use_container_width=True)
                    st.markdown('<div class="section-heading">Evidence</div>', unsafe_allow_html=True)
                    for provenance in selected_entity.provenance[:5]:
                        with st.expander(f"{provenance.source_document} · {provenance.source_section}"):
                            st.write(provenance.evidence_text)

                graph_lines = ["graph knowledge_graph {", '  graph [overlap=false, splines=true];', "  node [shape=box, style=rounded];"]
                graph_lines.append(f'  "{selected_entity.name}" [style="rounded,filled", fillcolor="#d9f0ee"];')
                for relationship in knowledge_graph.relationships(selected_entity.entity_id):
                    other_id = relationship.target_entity if relationship.source_entity == selected_entity.entity_id else relationship.source_entity
                    other = knowledge_graph.entity(other_id)
                    if other:
                        graph_lines.append(f'  "{selected_entity.name}" -- "{other.name}" [label="{relationship.relationship_type}"];')
                graph_lines.append("}")
                st.graphviz_chart("\n".join(graph_lines), use_container_width=True)
        else:
            st.info("No matching entities. Try a system, team, environment, risk, or document name.")

        st.markdown('<div class="section-heading">Path exploration</div>', unsafe_allow_html=True)
        entity_names = [entity.name for entity in knowledge_graph.entities()]
        path_columns = st.columns(2)
        with path_columns[0]:
            path_source = st.selectbox("From entity", entity_names, index=entity_names.index("OCR Service") if "OCR Service" in entity_names else 0, key="graph_path_source")
        with path_columns[1]:
            path_target = st.selectbox("To entity", entity_names, index=entity_names.index("R-001") if "R-001" in entity_names else 0, key="graph_path_target")
        paths = knowledge_graph.paths(path_source, path_target)
        if paths:
            path = paths[0]
            path_nodes = [path_source] + [knowledge_graph.entity(relationship.target_entity).name for relationship in path]
            st.success("Supported path found: " + " → ".join(path_nodes))
            for relationship in path:
                st.markdown(f"`{relationship.relationship_type}` · {relationship.provenance.source_document} · {relationship.provenance.source_section}")
                st.caption(relationship.provenance.evidence_text)
        else:
            st.info(f"No supported path found between {path_source} and {path_target}.")
elif selected_page == "Ask Knowledge":
    st.markdown('<div class="eyebrow">Grounded Q&A workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-heading">Ask Knowledge</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-copy">Ask a question and inspect the evidence behind the answer. Every response stays connected to retrieved source chunks.</div>', unsafe_allow_html=True)

    if "documents" not in st.session_state or not st.session_state.documents:
        if st.button("Load Sample Knowledge Base", type="primary"):
            load_knowledge_base()
            st.rerun()
        st.info("Load the sample knowledge base to ask grounded questions.")
    else:
        chunks = st.session_state.chunks
        vector_index: VectorIndex = st.session_state.vector_index
        knowledge_graph: KnowledgeGraph = st.session_state.knowledge_graph
        suggested_questions = [
            "What endpoints does the OCR Platform API expose?",
            "What caused the batch OCR incident and how was it resolved?",
            "What must be checked before promoting a release to production?",
            "What is the company's 2028 international expansion budget?",
        ]
        st.markdown("### Ask a question")

if "ask_question" not in st.session_state:
    st.session_state.ask_question = ""

def use_example(question_text):
    st.session_state.ask_question = question_text

example_cols = st.columns(3)

with example_cols[0]:
    st.button(
        "Production release checks",
        on_click=use_example,
        args=("What must be checked before promoting a release to production?",),
        use_container_width=True,
    )

with example_cols[1]:
    st.button(
        "OCR service risks",
        on_click=use_example,
        args=("What risks affect the OCR service?",),
        use_container_width=True,
    )

with example_cols[2]:
    st.button(
        "Kubernetes dependencies",
        on_click=use_example,
        args=("What depends on Kubernetes?",),
        use_container_width=True,
    )

question = st.text_area(
    "Question",
    placeholder="Ask about the Government OCR Platform...",
    height=90,
    key="ask_question",
)
controls = st.columns(3)
with controls[0]:
            retrieval_mode = st.radio("Retrieval", ["Semantic", "Keyword", "Graph", "Hybrid"], key="ask_retrieval_mode", horizontal=True)
with controls[1]:
            generation_mode = st.radio("Generation", ["Demo / Local", "OpenAI"], key="ask_generation_mode", horizontal=True)
with controls[2]:
            top_k = st.selectbox("Evidence Top-K", [1, 3, 5], index=2, key="ask_top_k")

if generation_mode == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
    st.caption("OpenAI is unavailable without an environment key. Demo / Local fallback will be used.")

if st.button("Ask with evidence", type="primary", disabled=not question.strip()):
    response = ask_knowledge(
        question,
        chunks,
        vector_index,
        knowledge_graph,
        retrieval_mode=retrieval_mode,
        generation_mode=generation_mode,
        top_k=top_k,
    )
    st.session_state.ask_response = response

response = st.session_state.get("ask_response")
if response:
            st.markdown('<div class="section-heading">Grounded answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hero"><div class="hero-kicker">{response.generation_mode.upper()}</div><h2>{response.answer}</h2><p>Grounding status: {response.grounding_status}</p></div>', unsafe_allow_html=True)
            if response.sources:
                st.markdown('<div class="section-heading">Sources used</div>', unsafe_allow_html=True)
                for source in response.sources:
                    st.markdown(f"- `{source}`")
            st.markdown('<div class="section-heading">Retrieved evidence</div>', unsafe_allow_html=True)
            for item in response.retrieved_evidence:
                st.markdown(f"**{item.rank}. {item.source_document}** · `{item.section}` · score `{item.score:.3f}`")
                if item.selection_reason:
                    st.caption(item.selection_reason)
                if item.relationship_type:
                    st.caption(f"Graph fact: {' → '.join(item.entity_path)} · {item.relationship_type}")
                st.write(item.text)
                st.divider()
elif selected_page == "Evaluation":
    st.markdown('<div class="eyebrow">Retrieval quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-heading">Retrieval Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-copy">Compare the transparent keyword baseline with local semantic retrieval against the 10-question ground-truth set.</div>', unsafe_allow_html=True)

    if "documents" not in st.session_state or not st.session_state.documents:
        if st.button("Load Sample Knowledge Base", type="primary"):
            load_knowledge_base()
        st.info("Load the sample knowledge base to run the retrieval evaluation.")
    else:
        documents = st.session_state.documents
        vector_index: VectorIndex = st.session_state.vector_index
        chunks = st.session_state.chunks
        knowledge_graph: KnowledgeGraph = st.session_state.knowledge_graph
        questions = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

        keyword_metrics = evaluate_retrieval(
            questions,
            lambda question, top_k: search_documents(documents, question)[:top_k],
        )
        semantic_metrics = evaluate_retrieval(
            questions,
            lambda question, top_k: vector_index.search(question, top_k),
        )
        graph_metrics = evaluate_retrieval(
            questions,
            lambda question, top_k: search_graph(question, knowledge_graph, chunks, top_k),
        )
        hybrid_metrics = evaluate_retrieval(
            questions,
            lambda question, top_k: search_hybrid(question, chunks, vector_index, knowledge_graph, top_k),
        )
        from backend.retrieval_evaluation import mean_reciprocal_rank, evaluate_by_question_type
        strategies = {
            "Keyword": lambda question, top_k: search_documents(documents, question)[:top_k],
            "Semantic": lambda question, top_k: vector_index.search(question, top_k),
            "Graph": lambda question, top_k: search_graph(question, knowledge_graph, chunks, top_k),
            "Hybrid": lambda question, top_k: search_hybrid(question, chunks, vector_index, knowledge_graph, top_k),
        }
        st.markdown('<div class="section-heading">Recall comparison</div>', unsafe_allow_html=True)
        st.dataframe(
            [{"Strategy": name, "Recall@1": f"{metrics[1]:.0%}", "Recall@3": f"{metrics[3]:.0%}", "Recall@5": f"{metrics[5]:.0%}", "MRR": f"{mean_reciprocal_rank(questions, retrieve):.0%}"} for name, metrics, retrieve in [
                ("Keyword", keyword_metrics, strategies["Keyword"]),
                ("Semantic", semantic_metrics, strategies["Semantic"]),
                ("Graph", graph_metrics, strategies["Graph"]),
                ("Hybrid", hybrid_metrics, strategies["Hybrid"]),
            ]],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown('<div class="section-heading">By question type</div>', unsafe_allow_html=True)
        breakdown_rows = []
        for question_type in sorted({question["question_type"] for question in questions}):
            row = {"Question type": question_type}
            for name, retrieve in strategies.items():
                row[name] = f"{evaluate_by_question_type(questions, retrieve)[question_type]['Recall@3']:.0%} R@3"
            breakdown_rows.append(row)
        st.dataframe(breakdown_rows, hide_index=True, use_container_width=True)
        st.caption(f"Evaluated {len(questions)} ground-truth questions. Semantic backend: {vector_index.embedding_model.backend}. Graph retrieval uses explicit entity relationships only.")
else:
    st.markdown(f'<div class="eyebrow">Workspace module</div><div class="module-heading">{selected_page}</div><div class="module-copy">A focused space for the next stage of the enterprise knowledge workflow.</div>', unsafe_allow_html=True)
    st.info("This module is planned for a future product iteration. No backend functionality is connected yet.")

st.markdown('</div>', unsafe_allow_html=True)
